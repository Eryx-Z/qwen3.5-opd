import pytest
import torch
from types import SimpleNamespace
from cq_opd.model_adapter import temporary_scoring_precision
from cq_opd.trainer import fd_precision,scoring_protocol


def test_promotion_restores_exact_storage_values_and_lora_on_failure():
    m=torch.nn.Module();m.base=torch.nn.Linear(4,3).bfloat16();m.base.requires_grad_(False)
    m.master=torch.nn.Parameter(torch.randn(3,4));m.register_buffer('b',torch.randn(2).bfloat16())
    a=SimpleNamespace(model=m);before=[(p,p.data,p.detach().clone()) for p in m.parameters()]
    b=m.b
    with pytest.raises(RuntimeError,match='injected'):
        with temporary_scoring_precision(a,torch.float32):
            assert m.base.weight.dtype==m.b.dtype==torch.float32 and m.master.dtype==torch.float32
            assert m.master.data_ptr()==before[0][1].data_ptr()
            raise RuntimeError('injected')
    for p,data,value in before:
        assert p.data_ptr()==data.data_ptr() and p.dtype==data.dtype
        assert torch.equal(p,value) and p.grad is None
    assert m.b is b


def test_nested_scoring_context_keeps_original_storage():
    m=torch.nn.Linear(4,3).bfloat16().requires_grad_(False);a=SimpleNamespace(model=m)
    ptr=m.weight.data_ptr()
    with temporary_scoring_precision(a,torch.float32):
        promoted=m.weight.data_ptr()
        with temporary_scoring_precision(a,torch.float32):assert m.weight.data_ptr()==promoted
        assert m.weight.dtype==torch.float32 and m.weight.data_ptr()==promoted
    assert m.weight.dtype==torch.bfloat16 and m.weight.data_ptr()==ptr


def test_protocol_binds_scoring_dtype_and_backend():
    c=SimpleNamespace(max_new_tokens=2048,block_len=16,keep_ratio=.2,chunk_size=64,
                      probe_questions=4,probe_answers=4,seed=42,student_dtype='bf16',
                      rollout_backend='vllm',fd_scoring_dtype='fp32',rollout_batch_size=16)
    p=scoring_protocol(c)
    assert fd_precision(c)==torch.float32 and p['delta_choice_rule']=='fp32_interior_margin_v1'
    assert p['rollout_backend']=='vllm' and p['sampling_rng']=='vllm_per_request_seed'
    c.fd_scoring_dtype='native';assert scoring_protocol(c)!=p
    c.rollout_backend='native';assert scoring_protocol(c)['sampling_rng']=='native_batch_seed_plus_row'
