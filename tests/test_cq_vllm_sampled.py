"""Sampled parity scope and actual rollout logprob provenance, CPU only."""
from types import SimpleNamespace as S
import pytest
import torch
from cq_opd.vllm_rollout import BaselineVLLMRollout


def fixture():
    r=object.__new__(BaselineVLLMRollout)
    r.parity_scope='sampled';r.policy_hash='current';r.ready=True;r.policy_checks=[]
    r.sampling_cls=lambda **kw:S(**kw)
    r._request=lambda:None
    r.tokenizer=S(apply_chat_template=lambda *a,**k:[1,2],decode=lambda *a,**k:'answer')
    logits=torch.tensor([[1.,2.,3.,4.]])
    r.adapter=S(device=torch.device('cpu'),last_hidden=lambda *a,**k:torch.zeros(1,2,1),
        project=lambda h:logits,model=S(config=S(text_config=S(eos_token_id=3))))
    lp=torch.log_softmax(logits,-1)[0]
    r.sync_policy=lambda:False
    return r,lp


def test_sampled_parity_checks_returned_token_not_other_vocabulary():
    r,lp=fixture();calls=[]
    def generate(prompts,params,**kw):
        calls.append(params)
        return [S(outputs=[S(token_ids=[2],logprobs=[{2:S(logprob=float(lp[2]))}])])]
    r.engine=S(generate=generate)
    r.validate_policy()
    assert calls[0].logprobs==0
    assert r.policy_checks[-1]['checked_tokens']==1
    assert r.policy_checks[-1]['scope']=='sampled'
    assert r.policy_checks[-1]['vocab_size']==4
    r.parity_scope='full'
    with pytest.raises(RuntimeError,match='entire actual vocabulary'):r.validate_policy()


@pytest.mark.parametrize('missing',[False,True])
def test_sampled_parity_refuses_nonfinite_or_missing_probability(missing):
    r,lp=fixture()
    entries={} if missing else {2:S(logprob=float('nan'))}
    r.engine=S(generate=lambda *a,**k:[S(outputs=[S(token_ids=[2],logprobs=[entries])])])
    with pytest.raises((RuntimeError,AssertionError)):r.validate_policy()
    assert not r.policy_checks


def test_finite_probability_drift_is_recorded_not_enforced():
    r,lp=fixture()
    r.engine=S(generate=lambda *a,**k:[S(outputs=[S(token_ids=[2],logprobs=[{2:S(logprob=float(lp[2])+1.)}])])])
    r.validate_policy()
    check=r.policy_checks[-1]
    assert not check['passed'] and not check['within_reference_tolerance']
    assert check['enforcement']=='diagnostic_only'
    assert check['max_abs']==pytest.approx(1.)


def test_generated_probability_is_bound_to_each_token_and_policy():
    r,lp=fixture();calls=[]
    def generate(prompts,params,**kw):
        calls.extend(params)
        return [S(prompt_token_ids=p['prompt_token_ids'],outputs=[S(token_ids=[2,3],
            logprobs=[{2:S(logprob=-.5)},{3:S(logprob=-.25)}],finish_reason='stop')]) for p in prompts]
    r.engine=S(generate=generate)
    results=r.generate([[1,2],[1]],2048,42,['a','b'])
    assert [p.max_tokens for p in calls]==[2048,2048]
    assert [p.seed for p in calls]==[42,43]
    assert all(p.logprobs==0 for p in calls)
    assert all(x.sampled_log_probs==[-.5,-.25] and x.policy_sha256=='current' for x in results)
    assert all(not x.truncated for x in results)


@pytest.mark.parametrize('size',[16,32])
def test_rollout_batch_keeps_response_cap_and_row_provenance(size):
    r,_=fixture();calls=[]
    def generate(prompts,params,**kw):
        calls.append((prompts,params))
        return [S(prompt_token_ids=p['prompt_token_ids'],outputs=[S(token_ids=[2,3],
            logprobs=[{2:S(logprob=-.5)},{3:S(logprob=-.25)}],finish_reason='stop')]) for p in prompts]
    r.engine=S(generate=generate)
    ids=[str(i) for i in range(size)]
    results=r.generate([[1]]*size,2048,42,ids)
    assert len(calls)==1 and len(calls[0][0])==size
    assert len(results)==size
    assert all(p.max_tokens==2048 and p.logprobs==0 for p in calls[0][1])
    assert [p.seed for p in calls[0][1]]==list(range(42,42+size))
    assert [row.sample_index for row in results]==list(range(size))
    assert all(row.generation_batch_size==size and row.policy_sha256=='current' for row in results)
    assert all(row.sampled_log_probs==[-.5,-.25] for row in results)


@pytest.mark.parametrize('logps',[None,[],[{}],[{2:S(logprob=float('nan'))}]])
def test_generation_rejects_invalid_sampled_logprobs(logps):
    r,lp=fixture()
    r.engine=S(generate=lambda *a,**k:[S(prompt_token_ids=[1],outputs=[S(token_ids=[2],
        logprobs=logps,finish_reason='length')])])
    with pytest.raises(RuntimeError):r.generate([[1]],1,42,['a'])
