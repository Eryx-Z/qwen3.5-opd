"""Fixed greedy dev/test evaluation of native CQ LoRA checkpoints. No tuning on test."""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from baseline.verifier import compute_score, extract
from .model_adapter import load_student, tokenizer_metadata
from .rollout import generate
from .trainer import digest, load_splits, prompt_ids, write_json, guarded_arguments


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--confirm-final-test',action='store_true')
    parser.add_argument('--max-new-tokens',type=int,default=2048)
    args=parser.parse_args(guarded_arguments())
    if not os.environ.get('VERL_OPD_GUARD_RUN'):
        raise RuntimeError('evaluation requires external guard')
    if args.split=='test' and not args.confirm_final_test:
        raise ValueError('test is final-only: explicitly pass --confirm-final-test')
    if args.max_new_tokens<=0:
        raise ValueError('positive decode budget required')
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path('/home/eryx/qwen3.5-opd')
    sp='/home/eryx/models/Qwen3.5-2B'
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)  # trusted only
    precision=checkpoint['config'].get('student_dtype','bf16')
    tokenizer=AutoTokenizer.from_pretrained(sp,local_files_only=True)
    student=load_student(sp,root/'baselines/gsm8k_k1/lora_manifest.json',base_dtype=precision)
    names=student.lora_named_parameters()
    if set(checkpoint['lora'])!={name for name,_ in names}:
        raise ValueError('checkpoint LoRA scope mismatch')
    meta=checkpoint['metadata']
    if student.metadata()!=meta['student'] or student.manifest_hash!=meta['lora_manifest_hash']:
        raise ValueError('checkpoint Student model/weight/LoRA revision mismatch')
    if tokenizer_metadata(tokenizer,thinking=False)!=meta['student_tokenizer'] or meta['thinking'] is not False:
        raise ValueError('checkpoint tokenizer/template/thinking mismatch')
    import hashlib
    actual_data={name:hashlib.sha256((root/'data/gsm8k'/f'{name}.parquet').read_bytes()).hexdigest()
                 for name in ('train','probe','dev','test')}
    if actual_data!=meta['dataset_sha256']:
        raise ValueError('checkpoint dataset revision mismatch')
    write_json(args.output/'manifest.json',dict(checkpoint=str(args.checkpoint),step=checkpoint['step'],
        split=args.split,max_new_tokens=args.max_new_tokens,decode='greedy',thinking=False,
        seed=42,student_dtype=precision,checkpoint_metadata_hash=digest(checkpoint['metadata'])))
    rows=load_splits(root/'data/gsm8k')[args.split]
    scores={}; summary={}
    for name in ('base','cq'):
        if name=='cq':
            with torch.no_grad():
                for key,p in names:
                    value=checkpoint['lora'][key]
                    if not torch.isfinite(value).all() or value.shape!=p.shape:
                        raise ValueError('invalid adapter checkpoint')
                    p.copy_(value.to(p.device))
        records=[]
        with (args.output/f'{name}.jsonl').open('x') as file:
            for row in rows:
                rollout=generate(student,tokenizer,prompt_ids(tokenizer,row),args.max_new_tokens,42,
                                 id=row['extra_info']['id'],do_sample=False)
                score=compute_score(row['data_source'],rollout.response_text,row['reward_model']['ground_truth'])
                try:
                    parsed=extract(rollout.response_text)
                except (ValueError,ZeroDivisionError):
                    parsed=None
                record=dict(id=rollout.id,response=rollout.response_text,correct=score,
                    parseable=parsed is not None,truncated=rollout.truncated,output_tokens=len(rollout.response_ids))
                records.append(record);file.write(json.dumps(record,ensure_ascii=False)+'\n');file.flush()
        scores[name]={r['id']:r['correct'] for r in records}
        n=len(records)
        summary[name]=dict(count=n,correct=sum(r['correct'] for r in records),
            accuracy=sum(r['correct'] for r in records)/n,
            truncation_rate=sum(r['truncated'] for r in records)/n,
            unparseable_rate=sum(not r['parseable'] for r in records)/n,
            mean_output_tokens=sum(r['output_tokens'] for r in records)/n)
        write_json(args.output/'summary.json',summary)
    summary['paired']=dict(improved=sum(scores['base'][k]==0 and scores['cq'][k]==1 for k in scores['base']),
        regressed=sum(scores['base'][k]==1 and scores['cq'][k]==0 for k in scores['base']))
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
