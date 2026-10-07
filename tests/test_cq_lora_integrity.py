import copy
import pytest
import torch
from cq_opd.lora_integrity import expected_fingerprints,verify_receiver,fingerprint


def fixture():
    config={'r':2,'lora_alpha':4}
    tensors={f'base_model.model.model.language_model.layers.0.mlp.{p}.lora_{k}.weight':torch.randn(2,2) for p in ('gate_proj','up_proj') for k in ('A','B')}
    expected=expected_fingerprints(tensors,config)
    records=[dict(name='language_model.model.layers.0.mlp.gate_up_proj',kind='lora_'+k.lower(),index=i,**expected[f'language_model.model.layers.0.mlp.{p}.lora_{k}']) for i,p in enumerate(('gate_proj','up_proj')) for k in ('A','B')]
    return tensors,config,records


def test_named_packed_receiver_exact():
    t,c,r=fixture()
    assert verify_receiver(r,t,c)['checked_tensors']==4


@pytest.mark.parametrize('case',['missing','hash','name','duplicate','dtype','index'])
def test_corruption_rejected(case):
    t,c,r=fixture();r=copy.deepcopy(r)
    if case=='missing':r.pop()
    if case=='hash':r[0]['sha256']='bad'
    if case=='name':r[0]['name']='language_model.model.layers.1.mlp.gate_up_proj'
    if case=='duplicate':r.append(r[0])
    if case=='dtype':r[0]['dtype']='torch.float32'
    if case=='index':r[0]['index']=9
    with pytest.raises(RuntimeError):verify_receiver(r,t,c)


def test_nonfinite_fingerprint_refuses():
    with pytest.raises(RuntimeError):fingerprint(torch.tensor([float('nan')]))
