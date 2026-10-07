"""GSM8K evaluation on test set for custom checkpoint."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
import pandas as pd
from safetensors.torch import save_file
from peft import LoraConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROOT = Path('/home/eryx/qwen3.5-opd')
sys.path.insert(0, str(ROOT))
from baseline.verifier import compute_score, extract


def export_adapter_from_pt(checkpoint_path, output_dir):
    data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    lora = data['lora']
    weights = {}
    modules = set()
    for key, value in lora.items():
        # key format: model.language_model.layers.0.linear_attn.out_proj.lora_A
        # peft format: base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.weight
        target_mod = key.rsplit('.', 1)[0].replace('model.language_model.', '')
        modules.add(target_mod)
        peft_key = f"base_model.model.{key}.weight"
        weights[peft_key] = value.contiguous().clone().to(torch.bfloat16)
    
    assert len(weights) == 372
    assert len(modules) == 186
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=sorted(modules),
        task_type='CAUSAL_LM'
    ).to_dict()
    config['target_modules'] = sorted(modules)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    save_file(weights, str(output_dir / 'adapter_model.safetensors'))
    print(f"Exported adapter to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    
    assert os.environ.get('VERL_OPD_GUARD_RUN'), 'run under guard'
    args.output.mkdir(parents=True, exist_ok=False)
    
    adapter_dir = args.output / 'adapter'
    export_adapter_from_pt(args.checkpoint, adapter_dir)
    
    source = ROOT / 'data/gsm8k/test.parquet'
    data = pd.read_parquet(source)
    assert len(data) == 1319
    
    model = '/home/eryx/models/Qwen3.5-2B'
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [tokenizer.apply_chat_template(list(row), tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in data.prompt]
    
    manifest = dict(
        dataset='official GSM8K main test',
        count=len(data),
        checkpoint=str(args.checkpoint),
        base_model=model,
        max_new_tokens=2048,
        temperature=0,
        seed=42
    )
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    
    llm = LLM(
        model=model, dtype='bfloat16', tensor_parallel_size=1, trust_remote_code=False,
        enable_lora=True, max_lora_rank=8, max_loras=1, max_model_len=2561,
        max_num_seqs=32, max_num_batched_tokens=4096, kv_cache_memory_bytes=4294967296,
        enforce_eager=True, enable_sleep_mode=False, seed=42, generation_config='vllm',
        limit_mm_per_prompt={'image': 0, 'video': 0}, skip_mm_profiling=True
    )
    sampling = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=2048, n=1, seed=42)
    
    summaries = {}
    scores = {}
    
    for name, request in [('base', None), ('step300', LoRARequest('step300', 1, str(adapter_dir)))]:
        records = []
        with (args.output / f'{name}.jsonl').open('x') as f:
            for start in range(0, len(data), 64):
                batch_prompts = prompts[start:start+64]
                outputs = llm.generate(batch_prompts, sampling, lora_request=request, use_tqdm=False)
                for index, result in enumerate(outputs, start):
                    row = data.iloc[index]
                    response = result.outputs[0]
                    correct = compute_score(row.data_source, response.text, row.reward_model['ground_truth'])
                    try:
                        parsed = extract(response.text)
                    except (ValueError, ZeroDivisionError):
                        parsed = None
                    record = dict(
                        id=row.extra_info['id'],
                        question_hash=row.extra_info['question_hash'],
                        response=response.text,
                        reference=row.reward_model['ground_truth'],
                        correct=correct,
                        parseable=parsed is not None,
                        output_tokens=len(response.token_ids),
                        finish_reason=response.finish_reason,
                        truncated=response.finish_reason == 'length'
                    )
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                print(f'{name}: {len(records)}/{len(data)}, correct={sum(r["correct"] for r in records)}', flush=True)
                
        summaries[name] = dict(
            count=len(records),
            correct=sum(r['correct'] for r in records),
            accuracy=sum(r['correct'] for r in records) / len(records),
            truncation_rate=sum(r['truncated'] for r in records) / len(records),
            unparseable_rate=sum(not r['parseable'] for r in records) / len(records),
            mean_output_tokens=sum(r['output_tokens'] for r in records) / len(records)
        )
        scores[name] = {r['id']: r['correct'] for r in records}
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        
    summaries['paired'] = dict(
        accuracy_delta=summaries['step300']['accuracy'] - summaries['base']['accuracy'],
        improved=sum(scores['base'][k] == 0 and scores['step300'][k] == 1 for k in scores['base']),
        regressed=sum(scores['base'][k] == 1 and scores['step300'][k] == 0 for k in scores['base'])
    )
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print("FINAL SUMMARY:\n", json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
