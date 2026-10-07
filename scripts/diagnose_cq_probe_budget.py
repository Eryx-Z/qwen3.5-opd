"""Replay an unsuccessful startup probe, comparing fixed seeds at two budgets.

Student-only diagnostic; no Teacher substitution, calibration or optimizer update.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from cq_opd import trainer
from cq_opd.model_adapter import load_student, require_gpu_guard
from baseline.verifier import compute_score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--failed-start',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(trainer.guarded_arguments());require_gpu_guard()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=SimpleNamespace(**json.loads((args.failed_start/'config.json').read_text()))
    entropy=json.loads((args.failed_start/'entropy_calibration.json').read_text())
    metadata=json.loads((args.failed_start/'model-manifest.json').read_text())
    root=Path('/home/eryx/qwen3.5-opd')
    splits=trainer.load_splits(root/'data/gsm8k')
    # Reconstruct exactly the Python RNG state at failed startup. Here every
    # generated entropy position was retained (<8192), so total count is known.
    assert len(entropy['positions'])<8192 and cfg.max_new_tokens==256
    rng=random.Random(cfg.seed)
    chosen=rng.sample(splits['train'],cfg.entropy_questions)
    for _ in chosen:rng.randrange(2**31)
    assert list(dict.fromkeys(p[0] for p in entropy['positions']))==[r['extra_info']['id'] for r in chosen]
    rng.sample(range(len(entropy['positions'])),len(entropy['positions']))
    torch.manual_seed(cfg.seed)
    tokenizer=AutoTokenizer.from_pretrained('/home/eryx/models/Qwen3.5-2B',local_files_only=True)
    student=load_student('/home/eryx/models/Qwen3.5-2B',root/'baselines/gsm8k_k1/lora_manifest.json',base_dtype='fp32')
    initial=hashlib.sha256()
    for name,p in student.lora_named_parameters():
        initial.update(name.encode());initial.update(p.detach().cpu().contiguous().numpy().tobytes())
    assert initial.hexdigest()==metadata['initial_lora_sha256']
    sample=trainer.sample;records=[]
    reference={row['extra_info']['id']:row for row in splits['probe']}
    def enrich(stats):
        rewards=[compute_score(reference[r.id]['data_source'],r.response_text,
                               reference[r.id]['reward_model']['ground_truth']) for r in records]
        stats.update(probe_rewards=[rewards[i:i+cfg.probe_answers] for i in range(0,len(rewards),cfg.probe_answers)],
            probe_response_lengths=[len(r.response_ids) for r in records],
            probe_truncated_fraction=sum(r.truncated for r in records)/len(records))
        return stats
    def recorded(*a,**k):
        result=sample(*a,**k);records.append(result);return result
    trainer.sample=recorded
    native_batch=trainer.generate_batch
    def recorded_batch(*a,**k):
        results=native_batch(*a,**k);records.extend(results);return results
    trainer.generate_batch=recorded_batch
    report=dict(scope='fixed_seed_probe_budget_diagnostic_only_no_updates',source=str(args.failed_start),results={})
    previous=[]
    for cap in (256,2048):
        records.clear();configuration=copy.copy(cfg);configuration.max_new_tokens=cap
        d,stats=trainer.direction(student,tokenizer,splits['probe'],configuration,0,copy.deepcopy(rng))
        enrich(stats)
        texts=[dict(id=r.id,seed=r.seed,response=r.response_text,length=len(r.response_ids),truncated=r.truncated)
               for r in records]
        if previous:
            assert [(r.id,r.seed) for r in records]==[(r.id,r.seed) for r in previous]
            assert all(b.response_ids[:len(a.response_ids)]==a.response_ids for a,b in zip(previous,records))
        report['results'][str(cap)]=dict(direction_valid=d.valid,grad_norm=d.grad_norm,stats=stats,responses=texts)
        trainer.write_json(args.output/'probe_budget.json',report)
        print(json.dumps(dict(cap=cap,direction_valid=d.valid,stats=stats)),flush=True)
        if cap==256:
            assert not d.valid, 'failed startup did not reproduce; investigate before proceeding'
        previous=list(records)
    # Fixed existing training stream, two PREDECLARED groups at unchanged theta0.
    # This diagnoses settings; it is not seed search or training permission.
    named_rng=random.Random(cfg.seed+2001)
    configuration=copy.copy(cfg);configuration.max_new_tokens=2048;configuration.rollout_batch_size=16
    report['named_training_probe_seed']=cfg.seed+2001
    report['named_training_sampling_rng']='native_batch_seed_plus_row'
    report['named_training_theta_version']=0
    report['named_training_groups']=[]
    for round_index in range(2):
        records.clear()
        d,stats=trainer.direction(student,tokenizer,splits['probe'],configuration,0,named_rng)
        enrich(stats)
        report['named_training_groups'].append(dict(direction_valid=d.valid,stats=stats,
            responses=[dict(id=r.id,seed=r.seed,response=r.response_text,length=len(r.response_ids),truncated=r.truncated)
                       for r in records]))
        trainer.write_json(args.output/'probe_budget.json',report)
        print(json.dumps(dict(named_group=round_index,direction_valid=d.valid,stats=stats)),flush=True)
    assert all(p.grad is None for _,p in student.lora_named_parameters())
    trainer.write_json(args.output/'probe_budget.json',report)


if __name__=='__main__':main()
