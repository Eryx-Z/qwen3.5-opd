"""Exact named BF16 receiver verification, independent of probability agreement."""
import hashlib
import torch

PACKED = {'in_proj_qkvz': ('in_proj_qkv','in_proj_z'),
          'in_proj_ba': ('in_proj_b','in_proj_a'),
          'gate_up_proj': ('gate_proj','up_proj'),
          'qkv_proj': ('q_proj','k_proj','v_proj')}


def fingerprint(t):
    t=t.detach().cpu().contiguous()
    if not torch.isfinite(t).all():raise RuntimeError('nonfinite received LoRA tensor')
    return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())


def expected_fingerprints(tensors,config):
    result={}
    scale=config['lora_alpha']/config['r']
    for name,t in tensors.items():
        prefix='base_model.model.model.language_model.'
        if not name.startswith(prefix) or not name.endswith('.weight'):raise ValueError('unsupported LoRA payload name')
        source='language_model.model.'+name[len(prefix):-len('.weight')]
        kind=source.rsplit('.',1)[1]
        if kind not in ('lora_A','lora_B'):raise ValueError('invalid LoRA payload kind')
        value=t.to(torch.bfloat16)
        if kind=='lora_B':value=value*scale
        result[source]=fingerprint(value)
    return result


def verify_receiver(records,tensors,config):
    expected=expected_fingerprints(tensors,config);actual={}
    for record in records:
        name=record['name'];parent,leaf=name.rsplit('.',1);index=record['index']
        parts=PACKED.get(leaf,(leaf,))
        if type(index)!=int or not 0<=index<len(parts):raise RuntimeError('invalid receiver packed index')
        if record['kind'] not in ('lora_a','lora_b'):raise RuntimeError('invalid receiver tensor kind')
        key=parent+'.'+parts[index]+'.'+('lora_A' if record['kind']=='lora_a' else 'lora_B')
        if key in actual:raise RuntimeError('duplicate receiver LoRA tensor')
        actual[key]={k:record[k] for k in ('shape','dtype','sha256')}
    if actual!=expected:raise RuntimeError('received LoRA weights differ from current named BF16 payload')
    return dict(passed=True,checked_tensors=len(actual),scope='named_receiver_BF16_storage_with_folded_B_scaling')
