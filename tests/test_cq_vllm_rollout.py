import sys
from types import SimpleNamespace,ModuleType
import pytest
import torch
from cq_opd.vllm_rollout import baseline_engine_kwargs,tensor_lora_payload,BaselineVLLMRollout


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(8,2).bfloat16()
    def get_input_embeddings(self):return self.emb


def adapter():
    a=torch.nn.Parameter(torch.randn(2,3));b=torch.nn.Parameter(torch.zeros(4,2))
    return SimpleNamespace(model=Tiny(),device=torch.device('cpu'),source_path='fixture',
        lora_config={'rank':2,'alpha':4},lora_named_parameters=lambda:[
        ('model.language_model.layers.0.self_attn.q_proj.lora_A',a),
        ('model.language_model.layers.0.self_attn.q_proj.lora_B',b)])


def test_payload_matches_baseline_peft_contract_preserves_master():
    a=adapter();t,c,sha=tensor_lora_payload(a)
    assert list(t)[0]=='base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight'
    assert c['target_modules']==['q_proj'] and c['r']==2 and c['lora_alpha']==4
    master=a.lora_named_parameters()[0][1];snapshot=master.detach().clone()
    next(iter(t.values())).zero_();assert torch.equal(snapshot,master)
    assert tensor_lora_payload(a)[2]==sha
    with torch.no_grad():master.add_(.1)
    assert tensor_lora_payload(a)[2]!=sha


@pytest.mark.parametrize('bad',['visual','dtype','nonfinite'])
def test_bad_payload_refused(bad):
    a=adapter();pairs=a.lora_named_parameters()
    if bad=='visual':pairs[0]=('model.visual.q_proj.lora_A',pairs[0][1])
    elif bad=='dtype':pairs[0]=(pairs[0][0],torch.nn.Parameter(pairs[0][1].bfloat16()))
    else:
        with torch.no_grad():pairs[0][1].fill_(float('nan'))
    a.lora_named_parameters=lambda:pairs
    with pytest.raises(ValueError):tensor_lora_payload(a)


def test_baseline_parameters_not_small_smoke_substitution():
    k=baseline_engine_kwargs('model')
    assert k['dtype']=='bfloat16' and k['max_model_len']==2561
    assert k['max_num_seqs']==16 and k['max_num_batched_tokens']==4096
    assert k['kv_cache_memory_bytes']==4*2**30 and k['max_lora_rank']==8
    assert k['gpu_memory_utilization']==.4
    assert k['enforce_eager'] and not k['enable_prefix_caching']
    assert k['worker_extension_cls']=='cq_opd.vllm_integrity_worker.IntegrityWorker'


@pytest.mark.parametrize('size',[16,32])
def test_concurrency_changes_only_scheduler_sequence_limit(size):
    default=baseline_engine_kwargs('model')
    configured=baseline_engine_kwargs('model',max_num_seqs=size)
    assert configured==dict(default,max_num_seqs=size)


@pytest.mark.parametrize('size',[0,-1,1.5,16.0,'32',None,True,False])
def test_invalid_concurrency_refused_before_engine_creation(size):
    with pytest.raises(ValueError,match='max_num_seqs must be a positive integer'):
        baseline_engine_kwargs('model',max_num_seqs=size)
    with pytest.raises(ValueError,match='max_num_seqs must be a positive integer'):
        BaselineVLLMRollout(adapter(),None,max_num_seqs=size)


@pytest.mark.parametrize('size',[None,16,32])
def test_constructor_passes_concurrency_to_vllm_without_changing_memory_limits(monkeypatch,size):
    monkeypatch.setenv('VERL_OPD_GUARD_RUN','cpu-only-unit')
    tokenizer=SimpleNamespace(get_vocab=lambda:{'x':0},all_special_ids=[0])
    calls=[]
    def llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tokenizer=lambda:tokenizer)
    module=ModuleType('vllm');module.LLM=llm;module.SamplingParams=object
    monkeypatch.setitem(sys.modules,'vllm',module)
    a=adapter();a.device=torch.device('cuda')  # fake engine; never allocate CUDA
    kwargs={} if size is None else {'max_num_seqs':size}
    BaselineVLLMRollout(a,tokenizer,**kwargs)
    assert calls==[dict(baseline_engine_kwargs('fixture'),max_num_seqs=16 if size is None else size)]


def test_sync_reuses_existing_baseline_flow_and_refuses_failed_parity(monkeypatch):
    monkeypatch.setenv('VERL_OPD_GUARD_RUN','cpu-only-unit')
    module=ModuleType('verl.workers.rollout.vllm_rollout.utils');module.VLLM_LORA_INT_ID=123
    monkeypatch.setitem(sys.modules,module.__name__,module)
    tokenizer=SimpleNamespace(get_vocab=lambda:{'x':0},all_special_ids=[0]);events=[];active=set()
    def remove(i):events.append('remove');active.discard(i)
    def transfer(engine,tensors,config):
        assert all(isinstance(t,torch.Tensor) for t in tensors.values()) and config['r']==2
        events.append('sync');active.add(123);return [None]
    monkeypatch.setattr('cq_opd.vllm_rollout.baseline_transfer',transfer)
    engine=SimpleNamespace(get_tokenizer=lambda:tokenizer,
        llm_engine=SimpleNamespace(remove_lora=remove,list_loras=lambda:active),
        reset_prefix_cache=lambda:events.append('reset') or True,
        collective_rpc=lambda name:[[]])
    monkeypatch.setattr('cq_opd.vllm_rollout.verify_receiver',lambda *a:dict(passed=True))
    def parity(self):events.append('parity')
    monkeypatch.setattr(BaselineVLLMRollout,'validate_policy',parity)
    a=adapter();r=BaselineVLLMRollout(a,tokenizer,engine=engine,sampling_cls=None)
    assert r.sync_policy() and events==['remove','sync','reset','parity'] and r.ready
    assert not r.sync_policy()
    with torch.no_grad():a.lora_named_parameters()[1][1].add_(.1)
    def bad(self):raise RuntimeError('parity failed')
    monkeypatch.setattr(BaselineVLLMRollout,'validate_policy',bad)
    with pytest.raises(RuntimeError,match='parity failed'):r.sync_policy()
    assert not r.ready
