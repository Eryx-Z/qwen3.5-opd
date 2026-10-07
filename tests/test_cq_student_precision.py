"""Explicit Student compute precision; original Teacher contract is unchanged."""
import json

import pytest
import torch

from cq_opd.model_adapter import FP32LoRALinear, load_student, check_logits_equivalence, student_compute_dtype
from cq_opd.rollout import build_batch, generate
from test_cq_adapter import tiny, records, Tokenizer


@pytest.mark.parametrize('precision,dtype',[('bf16',torch.bfloat16),('fp32',torch.float32)])
def test_native_loader_precision_frozen_base_master_and_generation(tmp_path,precision,dtype):
    torch.set_num_threads(1)
    original=tiny().model
    targets=[]
    # Save an actual tiny native checkpoint, without an adapter wrapper layout.
    for name,module in list(original.named_modules()):
        if isinstance(module,FP32LoRALinear):
            targets.append(name)
            parent,leaf=name.rsplit('.',1)
            setattr(original.get_submodule(parent),leaf,module.base)
    model_path=tmp_path/'model';original.save_pretrained(model_path)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'target_modules':targets}))
    adapter=load_student(model_path,manifest,device='cpu',base_dtype=precision,rank=2,alpha=4)
    assert adapter.language_model.embed_tokens.weight.dtype==dtype
    assert adapter.model.lm_head.weight.dtype==dtype
    originals=dict(original.named_parameters())
    for name,p in adapter.model.named_parameters():
        if name.endswith('.linear_attn.A_log'):
            assert p.dtype==torch.float32
            assert torch.equal(p,originals[name].float())
    assert all(p.dtype==torch.float32 and p.requires_grad for _,p in adapter.lora_named_parameters())
    assert all(not p.requires_grad for n,p in adapter.model.named_parameters()
               if not n.endswith(('.lora_A','.lora_B')))
    batch=build_batch(records(),'cpu',2)
    proof=check_logits_equivalence(adapter,batch)
    assert proof['mode_hidden_max_abs']<.02
    hidden=adapter.last_hidden(batch,with_grad=True)
    gradients=torch.autograd.grad(adapter.project(hidden).square().mean(),
                                  [p for _,p in adapter.lora_named_parameters()])
    assert all(torch.isfinite(g).all() for g in gradients)
    report=adapter.metadata()
    assert report['precision']['embedding_dtype']==str(dtype)
    assert report['precision']['head_dtype']==str(dtype)
    state=torch.get_rng_state().clone()
    result=generate(adapter,Tokenizer(),[4,5],2,123)
    assert result.prompt_len==2 and len(result.response_ids)>0
    assert torch.equal(state,torch.get_rng_state())


def test_evaluation_reads_fp32_checkpoint_contract_before_loading(monkeypatch,tmp_path):
    from cq_opd import evaluate
    monkeypatch.setenv('VERL_OPD_GUARD_RUN','cpu-entry-test')
    monkeypatch.setattr('sys.argv',['evaluate','--checkpoint',str(tmp_path/'trusted.pt'),
                                  '--output',str(tmp_path/'eval')])
    checkpoint={'config':{'student_dtype':'fp32'}}
    monkeypatch.setattr(torch,'load',lambda *a,**k:checkpoint)
    monkeypatch.setattr(evaluate.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    calls=[]
    class StopAfterLoad(RuntimeError):pass
    def load(*a,**k):
        calls.append(k['base_dtype'])
        assert student_compute_dtype(k['base_dtype'])==torch.float32
        assert not torch.backends.cuda.matmul.allow_tf32
        assert torch.get_float32_matmul_precision()=='highest'
        raise StopAfterLoad()
    monkeypatch.setattr(evaluate,'load_student',load)
    with pytest.raises(StopAfterLoad):evaluate.main()
    assert calls==['fp32']
