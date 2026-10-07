import copy
import json
import pytest
import torch
from cq_opd import *
from cq_opd.losses import log_probs, soft_ce, chunked_hard_label_hidden_grad
from cq_opd.probe import probe_loss
from cq_opd.selectors import teachability_scores

DT = torch.float64


def random(shape, seed=1):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=DT)


def head():
    m = torch.nn.Linear(3, 7, dtype=DT)
    with torch.no_grad():
        m.weight.copy_(random((7, 3)))
        m.bias.copy_(random((7,), 2))
    m.requires_grad_(False)
    return m


def test_shift_blocks_eos_and_padding():
    # EOS and padding may share the same token ID; masks alone decide inclusion.
    response = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    valid = shifted_valid_mask(response, attention)
    assert valid[0].tolist() == [False, False, True, True, False]
    blocks = make_blocks(valid, 2)
    assert blocks == [Block(0, (2, 3)), Block(1, (0, 1)), Block(1, (2, 3)), Block(1, (4,))]


@pytest.mark.parametrize('a', [0., .37, 1.])
def test_full_kl_gradients_endpoints(a):
    s = random((4, 7)).requires_grad_()
    teacher = random((4, 7), 2).requires_grad_()
    alpha = torch.full((4,), a, dtype=DT, requires_grad=True)
    lq = log_probs(teacher)
    f, r = kl_components(s, lq)
    loss = mixed_kl(s, lq, alpha)
    torch.testing.assert_close(loss, a * f + (1-a) * r)
    assert loss.dtype == DT
    loss.sum().backward()
    assert teacher.grad is None and alpha.grad is None
    lp, p, q = log_probs(s), log_probs(s).exp(), lq.detach().exp()
    rv = lp - lq.detach()
    analytic = a * (p-q) + (1-a)*p*(rv-(p*rv).sum(-1, keepdim=True))
    torch.testing.assert_close(s.grad, analytic, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(q.sum(-1), torch.ones(4, dtype=DT))
    if a == 1:
        torch.testing.assert_close(torch.autograd.grad(soft_ce(s, lq).sum(), s)[0], s.grad)
    if a == 0:
        detached_p_wrong = (p.detach() * (lp-lq.detach())).sum()
        wrong = torch.autograd.grad(detached_p_wrong, s)[0]
        assert not torch.allclose(wrong, s.grad)


def test_entropy_calibration_mask_and_serialization():
    project = head()
    hidden = random((2, 6, 3))
    valid = torch.tensor([[0,1,1,0,0],[1,1,0,1,0]], dtype=torch.bool)
    cal = calibrate_entropy(torch.tensor([.1, .2, .6, 1.2, 1.8], dtype=DT))
    restored = json.loads(json.dumps(cal))
    meta = prepare_entropy_alpha(hidden, valid, cal, project, 2)
    other = prepare_entropy_alpha(hidden, valid, restored, project, 3)
    torch.testing.assert_close(meta.alpha, other.alpha)
    assert (meta.alpha[~valid] == 0).all() and not meta.alpha.requires_grad
    e, a = entropy_alpha(log_probs(torch.tensor([[0.,0.,0.],[10.,0.,0.]], dtype=DT)), cal)
    assert e[0] > e[1] and a[0] > a[1]
    assert cal['tau'] == .6 and cal['scale'] == 1.0
    assert calibrate_entropy(torch.ones(3, dtype=DT))['scale'] == .1
    with pytest.raises(ValueError):
        calibrate_entropy(torch.empty(0))
    with pytest.raises(FloatingPointError):
        calibrate_entropy(torch.tensor([float('nan')]))


def test_token_mix_not_block_average_mix():
    s = random((3,7))
    lq = log_probs(random((3,7), 2))
    alpha = torch.tensor([0., .2, 1.], dtype=DT)
    f,r = kl_components(s,lq)
    assert not torch.allclose(mixed_kl(s,lq,alpha).mean(), alpha.mean()*f.mean()+(1-alpha.mean())*r.mean())


def test_signed_stable_per_response_budget_and_r1():
    valid = torch.ones((2, 5), dtype=torch.bool)
    blocks = make_blocks(valid,2)
    scores = torch.tensor([[-5.,-5.,-1.,-1.,-1.],[-4.,-4.,-3.,-3.,-8.]],dtype=DT)
    selected = select_blocks(scores, blocks, .2)
    assert selected.tolist() == [[False,False,True,True,False],[False,False,True,True,False]]
    assert torch.equal(select_blocks(scores, blocks,1),valid)
    assert torch.equal(select_by_method('full',valid,blocks),valid)
    rng = torch.Generator().manual_seed(3)
    assert select_by_method('random',valid,blocks,generator=rng).sum() >= 2


@pytest.mark.parametrize('chunk_size', [1,2,64])
def test_chunk_vjp_matches_full_dynamic_and_r1(chunk_size):
    project = head()
    p = torch.nn.Parameter(random((3,3)))
    x = random((2,6,3),2)
    teacher = random((2,6,3),3)
    valid = torch.tensor([[0,1,1,1,0],[1,1,0,1,1]],dtype=torch.bool)
    h = x @ p
    alpha = prepare_entropy_alpha(teacher,valid,{'tau':1.,'scale':.5},project).alpha
    selected = select_blocks(torch.zeros_like(alpha),make_blocks(valid,2),1)
    direct = mixed_kl(project(h[:,:-1]),log_probs(project(teacher[:,:-1])),alpha)[valid].sum()
    reference = torch.autograd.grad(direct,p)[0]
    h = x @ p
    dh,total,count = chunked_mixed_kl_hidden_grad(h,teacher,alpha,selected,project,project,chunk_size)
    actual = torch.autograd.grad(h,p,grad_outputs=dh)[0]
    torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(total,direct.detach())
    assert count == int(valid.sum()) and not dh.requires_grad
    assert (dh[:,-1] == 0).all() and p.grad is None


def test_loo_probe_length_and_zero_signal():
    rewards = torch.tensor([[1,0,1,0],[0,0,0,0],[1,1,1,1]],dtype=DT)
    a = leave_one_out_advantages(rewards)
    torch.testing.assert_close(a[0],torch.tensor([2/3,-2/3,2/3,-2/3],dtype=DT))
    assert (a[1:] == 0).all()
    p = torch.nn.Parameter(torch.tensor(2.,dtype=DT))
    p.grad = torch.tensor(99.,dtype=DT)
    unused = torch.nn.Parameter(torch.tensor(1.,dtype=DT))
    calls = []
    def callback(i,k):
        calls.append((i,k))
        return p * (k+1)  # negative summed logprob, variable length
    direction = compute_probe_direction([('p',p),('unused',unused)],a,callback,source_step=7)
    expected = sum(float(a[0,k])*(k+1)/a.numel() for k in range(4))
    assert len(calls) == 4 and direction.source_step == 7
    assert direction.unused_names == ('unused',)
    assert direction.grad_norm == pytest.approx(abs(expected))
    assert float(direction.tensors[0]) > 0 and float(p.grad) == 99
    zero = compute_probe_direction([('p',p)],a[1:],lambda i,k: pytest.fail('zero signal must skip'))
    assert not zero.valid and zero.reason == 'no_probe_signal'
    with pytest.raises(FloatingPointError):
        direction_from_gradients([('p',p)],[torch.tensor(float('nan'))])
    logits = torch.zeros((2,3,4),dtype=DT,requires_grad=True)
    valid = torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
    loss = probe_loss(logits,torch.zeros((2,3),dtype=torch.long),valid,torch.tensor([[1.,-1.]],dtype=DT))
    assert float(loss.detach()) == pytest.approx(torch.log(torch.tensor(4.)).item(),rel=1e-7)


def test_chunked_hard_probe_callback_equivalence_without_dot_grad():
    project = head()
    p = torch.nn.Parameter(random((3,3)))
    p.grad = torch.ones_like(p)
    x = random((1,5,3))
    valid = torch.tensor([[1,1,0,1]],dtype=torch.bool)
    ids = torch.tensor([[0,1,2,3]])
    h = x @ p
    direct = -log_probs(project(h[:,:-1])).gather(-1,ids[...,None]).squeeze(-1)[valid].sum()
    expected = torch.autograd.grad(direct,p)[0]
    actual = chunked_probe_gradients(x@p,ids,valid,project,[('p',p)],chunk_size=2)[0]
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    a = torch.tensor([[1.,-1.]],dtype=DT)
    def callback(i,k):
        return chunked_probe_gradients((x*(k+1))@p,ids,valid,project,[('p',p)],chunk_size=1)
    d = compute_probe_direction([('p',p)],a,callback)
    gradients = [callback(0,k)[0] for k in range(2)]
    reference = direction_from_gradients([('p',p)],[(gradients[0]-gradients[1])/2])
    torch.testing.assert_close(d.tensors[0],reference.tensors[0])
    assert torch.equal(p.grad,torch.ones_like(p))


def test_finite_difference_stable_formula_explicit_inner_product_restore():
    p = torch.nn.Parameter(random((2,4)))
    teacher = random((2,4),2).requires_grad_()
    lq = log_probs(teacher)
    a = torch.tensor([.1,.8],dtype=DT,requires_grad=True)
    g = random((2,4),3)
    d = direction_from_gradients([('p',p)],[g])
    original = p.detach().clone()
    p.grad = torch.ones_like(p)
    def scoring(): return p.detach().clone()
    good,bad = score_symmetric_offsets([('p',p)],d,1e-5,scoring)
    stable = mixed_utility_from_logits(good,bad,lq,a,1e-5)
    subtraction = (mixed_kl(bad,lq,a)-mixed_kl(good,lq,a))/(2e-5)
    explicit = torch.stack([(torch.autograd.grad(mixed_kl(p,lq,a)[i],p)[0] * (-d.tensors[0])).sum() for i in range(2)])
    torch.testing.assert_close(stable,subtraction,atol=1e-10,rtol=1e-8)
    torch.testing.assert_close(stable,explicit,atol=1e-9,rtol=1e-8)
    assert torch.equal(p,original) and torch.equal(p.grad,torch.ones_like(p))
    assert teacher.grad is None and a.grad is None
    n = 0
    def failing():
        nonlocal n
        n += 1
        if n == 2: raise RuntimeError('second forward failed')
        return p.clone()
    with pytest.raises(RuntimeError):
        score_symmetric_offsets([('p',p)],d,.1,failing)
    assert torch.equal(p,original) and torch.equal(p.grad,torch.ones_like(p))


def test_utility_sign_and_delta_convergence():
    p = torch.nn.Parameter(torch.tensor(.4,dtype=DT))
    direction = direction_from_gradients([('p',p)],[torch.ones_like(p)])
    good,bad = score_symmetric_offsets([('p',p)],direction,.01,lambda: torch.stack([p,p.neg()]).clone())
    utility = (bad-good)/.02
    assert utility[0] > 0 and utility[1] < 0
    exact = 3*float(p.detach())**2
    errors = []
    for delta in [.1,.01,.001]:
        good,bad = score_symmetric_offsets([('p',p)],direction,delta,lambda: p.clone()**3)
        errors.append(abs(float((bad-good)/(2*delta))-exact))
    assert errors[2] < errors[1] < errors[0]
    assert delta_from_rho([('p',p)],.01) == pytest.approx(.004)


def test_unequal_microbatch_raw_sums_equivalence():
    project = head()
    p = torch.nn.Parameter(random((3,3)))
    xs = [random((1,4,3),2),random((1,7,3),3)]
    teachers = [random(x.shape,4) for x in xs]
    masks = [torch.tensor([[1,0,1]],dtype=torch.bool),torch.tensor([[1,1,1,1,0,1]],dtype=torch.bool)]
    alphas = [prepare_entropy_alpha(t,m,{'tau':1.,'scale':.5},project).alpha for t,m in zip(teachers,masks)]
    total_tokens = sum(int(m.sum()) for m in masks)
    full_loss = sum(mixed_kl(project(x@p)[:,:-1],log_probs(project(t[:,:-1])),a)[m].sum() for x,t,a,m in zip(xs,teachers,alphas,masks))/total_tokens
    expected = torch.autograd.grad(full_loss,p)[0]
    for x,t,a,m in zip(xs,teachers,alphas,masks):
        h = x@p
        dh,_,_ = chunked_mixed_kl_hidden_grad(h,t,a,m,project,project,2)
        torch.autograd.backward(h,dh)
    divide_accumulated_gradients_by([p],total_tokens)
    torch.testing.assert_close(p.grad,expected,atol=1e-12,rtol=1e-12)


def test_chunk_utility_matches_logits_and_no_graph():
    project = head()
    good,bad,teacher = [random((1,5,3),s) for s in [1,2,3]]
    valid = torch.tensor([[1,0,1,1]],dtype=torch.bool)
    alpha = prepare_entropy_alpha(teacher,valid,{'tau':1.,'scale':.5},project).alpha
    actual = chunked_token_utility(good,bad,teacher,alpha,valid,.01,project,project,2)
    expected = mixed_utility_from_logits(project(good[:,:-1]),project(bad[:,:-1]),log_probs(project(teacher[:,:-1])),alpha,.01)
    torch.testing.assert_close(actual[valid],expected[valid])
    assert (actual[~valid] == 0).all() and not actual.requires_grad


def test_fail_fast_nonfinite_not_clipped():
    with pytest.raises(FloatingPointError):
        mixed_kl(torch.tensor([[float('nan'),1.]]),torch.tensor([[-1.,-1.]]),torch.ones(1))
    s = torch.tensor([[0.,0.]],dtype=DT)
    # No clamping of tiny negative KL: exact full-vocab expression is retained.
    lq = log_probs(s)
    torch.testing.assert_close(mixed_kl(s,lq,torch.ones(1)),torch.zeros(1,dtype=DT))


@pytest.mark.parametrize('a', [0., .5, 1.])
def test_constant_alpha_cache_and_frozen_teacher(a):
    project = head()
    teacher = random((1,4,3)).requires_grad_()
    valid = torch.tensor([[1,0,1]],dtype=torch.bool)
    meta = prepare_entropy_alpha(teacher,valid,{'mode':'constant','constant_alpha':a},project,1)
    assert (meta.alpha[valid] == a).all() and (meta.alpha[~valid] == 0).all()
    h = random((1,4,3),2).requires_grad_()
    dh,total,count = chunked_mixed_kl_hidden_grad(h,teacher,meta.alpha,valid,project,project,1)
    torch.autograd.backward(h,dh)
    assert teacher.grad is None and project.weight.grad is None and project.bias.grad is None
    assert count == 2 and not total.requires_grad


def test_low_precision_logits_promote_fp32():
    s = random((2,7)).to(torch.bfloat16).requires_grad_()
    lq = log_probs(random((2,7),2).to(torch.bfloat16))
    loss = mixed_kl(s,lq,torch.full((2,),.4))
    assert lq.dtype == torch.float32 and loss.dtype == torch.float32
    loss.sum().backward()
    assert torch.isfinite(s.grad).all()


def test_optimizer_state_restored_without_touching_gradients():
    p = torch.nn.Parameter(random((2,3)))
    opt = torch.optim.AdamW([p], lr=.01)
    p.square().sum().backward()
    opt.step()
    saved_parameter = p.detach().clone()
    saved_grad = p.grad.clone()
    saved_state = copy.deepcopy(opt.state_dict())
    d = direction_from_gradients([('p',p)],[random(p.shape,2)])
    score_symmetric_offsets([('p',p)],d,.1,lambda: p.square().sum())
    assert torch.equal(p,saved_parameter) and torch.equal(p.grad,saved_grad)
    current = opt.state_dict()
    assert current['param_groups'] == saved_state['param_groups']
    for key,value in saved_state['state'][0].items():
        torch.testing.assert_close(current['state'][0][key],value,rtol=0,atol=0)
    assert all(not hasattr(x,'grad_fn') or x.grad_fn is None for x in d.tensors)


def test_oversized_finite_gradient_norm_fails():
    p = torch.nn.Parameter(torch.ones(1,dtype=DT))
    with pytest.raises(FloatingPointError):
        direction_from_gradients([('p',p)],[torch.tensor([1e300],dtype=DT)])
