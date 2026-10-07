"""Enumerate the pinned real architecture on meta; no model weights or GPU load."""
import json
from pathlib import Path
import torch
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

path = '/home/eryx/models/Qwen3.5-2B'
config = AutoConfig.from_pretrained(path, local_files_only=True)
config._attn_implementation = 'sdpa'
with torch.device('meta'):
    model = Qwen3_5ForConditionalGeneration(config)
    targets = [name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith('model.language_model.layers.')
        and name.rsplit('.', 1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
                                      'in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'}]
    if not targets:
        raise ValueError('no actual language LoRA targets')
    adapted = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0, bias='none', task_type='CAUSAL_LM'))
trainable = [name for name, p in adapted.named_parameters() if p.requires_grad]
assert trainable and all('lora_' in name and '.language_model.layers.' in name for name in trainable)
assert not any(p.requires_grad for p in adapted.get_input_embeddings().parameters())
assert not any(p.requires_grad for p in adapted.get_output_embeddings().parameters())
manifest = dict(target_modules=targets, trainable_parameters=trainable,
                rank=8, alpha=16, dropout=0, trainable_count=sum(p.numel() for p in adapted.parameters() if p.requires_grad))
output = Path('/home/eryx/qwen3.5-opd/baselines/gsm8k_k1/lora_manifest.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(target_count=len(targets), trainable_count=manifest['trainable_count'])))
