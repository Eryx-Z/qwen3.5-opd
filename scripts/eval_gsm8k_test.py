"""Frozen full-test comparison: base Student versus step-300 language LoRA."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/eryx/qwen3.5-opd')
sys.path.insert(0, str(ROOT))
from baseline.verifier import compute_score, extract


def export_adapter(checkpoint, output):
    import torch
    from safetensors.torch import save_file
    from peft import LoraConfig
    meta = json.loads((checkpoint / 'lora_train_meta.json').read_text())
    assert meta['r'] == 8 and meta['lora_alpha'] == 16
    state = torch.load(checkpoint / 'model_world_size_1_rank_0.pt', map_location='cpu', mmap=True, weights_only=False)
    weights = {}
    modules = set()
    for key, value in state.items():
        if 'lora_' not in key:
            continue
        assert '.language_model.layers.' in key
        if hasattr(value, 'to_local'):
            local = value.to_local()
            assert tuple(local.shape) == tuple(value.shape), 'not a single-rank full tensor'
        else:
            local = value
        assert torch.isfinite(local).all()
        weights[key.replace('.default.weight', '.weight')] = local.contiguous().clone()
        modules.add(key.split('.lora_')[0].removeprefix('base_model.model.'))
    assert len(weights) == 372 and len(modules) == 186
    assert all(torch.count_nonzero(v).item() > 0 for k, v in weights.items() if 'lora_B' in k)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0, target_modules=sorted(modules), task_type='CAUSAL_LM').to_dict()
    config['target_modules'] = sorted(modules)
    output.mkdir()
    (output / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    save_file(weights, str(output / 'adapter_model.safetensors'))
    print('Exported and verified 372 adapter tensors', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('VERL_OPD_GUARD_RUN'), 'run under guard'
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'runs/gsm8k_k1_300_2048_20261005_194228/checkpoints/global_step_300/actor'
    adapter = args.output / 'adapter'
    export_adapter(checkpoint, adapter)
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    source = ROOT / 'data/gsm8k/test.parquet'
    data = pd.read_parquet(source)
    assert len(data) == 1319
    assert len({r['id'] for r in data.extra_info}) == 1319
    model = '/home/eryx/models/Qwen3.5-2B'
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [tokenizer.apply_chat_template(list(row), tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in data.prompt]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    assert max(lengths) <= 512
    manifest = dict(dataset='official GSM8K main test',count=len(data),data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint=str(checkpoint),base_model=model,max_new_tokens=2048,max_prompt_tokens=max(lengths),temperature=0,seed=42,
        enable_thinking=False,verifier='baseline/verifier.py; strict #### or boxed final number',
        max_num_seqs=16,kv_cache_bytes=4294967296,test_used_for_training=False)
    (args.output / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    llm = LLM(model=model, dtype='bfloat16', tensor_parallel_size=1, trust_remote_code=False,
        enable_lora=True,max_lora_rank=8,max_loras=1,max_model_len=2561,
        max_num_seqs=16,max_num_batched_tokens=4096,kv_cache_memory_bytes=4294967296,
        enforce_eager=True,enable_sleep_mode=False,seed=42,generation_config='vllm',
        limit_mm_per_prompt={'image':0,'video':0},skip_mm_profiling=True)
    sampling = SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=2048,n=1,seed=42)
    summaries = {}
    scores = {}
    for name, request in [('base',None),('step300',LoRARequest('step300',1,str(adapter)))]:
        records = []
        # Persist results incrementally, bounded scheduling batches independent of engine concurrency.
        with (args.output / f'{name}.jsonl').open('x') as f:
            for start in range(0,len(data),64):
                outputs = llm.generate(prompts[start:start+64],sampling,lora_request=request,use_tqdm=False)
                assert len(outputs) == len(data.iloc[start:start+64])
                for index, result in enumerate(outputs,start):
                    row = data.iloc[index]; response = result.outputs[0]
                    assert result.prompt == prompts[index]
                    correct = compute_score(row.data_source,response.text,row.reward_model['ground_truth'])
                    try:
                        parsed = extract(response.text)
                    except (ValueError,ZeroDivisionError):
                        parsed = None
                    record = dict(id=row.extra_info['id'],question_hash=row.extra_info['question_hash'],
                        response=response.text,reference=row.reward_model['ground_truth'],correct=correct,
                        parseable=parsed is not None,output_tokens=len(response.token_ids),finish_reason=response.finish_reason,
                        truncated=response.finish_reason=='length')
                    records.append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n')
                f.flush()
                print(f'{name}: {len(records)}/1319, correct={sum(r["correct"] for r in records)}',flush=True)
        summaries[name] = dict(count=len(records),correct=sum(r['correct'] for r in records),
            accuracy=sum(r['correct'] for r in records)/len(records),
            truncation_rate=sum(r['truncated'] for r in records)/len(records),
            unparseable_rate=sum(not r['parseable'] for r in records)/len(records),
            mean_output_tokens=sum(r['output_tokens'] for r in records)/len(records))
        scores[name] = {r['id']:r['correct'] for r in records}
        (args.output / 'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    summaries['paired'] = dict(accuracy_delta=summaries['step300']['accuracy']-summaries['base']['accuracy'],
        improved=sum(scores['base'][k]==0 and scores['step300'][k]==1 for k in scores['base']),
        regressed=sum(scores['base'][k]==1 and scores['step300'][k]==0 for k in scores['base']))
    (args.output / 'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    main()
