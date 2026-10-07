import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verifier', ROOT / 'baseline/verifier.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

@pytest.mark.parametrize('response,answer,score', [
    ('reasoning 12\n#### 1,200', '1200', 1),
    ('#### 2.0', '2', 1), ('#### -1/2', '-0.5', 1),
    (r'Answer: \boxed{\frac{1}{2}}', '0.5', 1),
    ('there are 2', '2', 0), ('#### 3', '2', 0),
    ('#### 1/0', '1', 0), ('#### __import__("os")', '2', 0),
    ('#### 2\n#### 3', '2', 0), ('#### 2 apples', '2', 0),
])
def test_verifier(response, answer, score):
    assert verifier.compute_score('openai/gsm8k', response, answer) == score


def test_bad_reference_fails():
    with pytest.raises(ValueError):
        verifier.compute_score('openai/gsm8k', '#### 1', 'bad')


def test_sampling_gradient_identity():
    import torch
    # Conditional on one fixed prefix: expected k1 policy gradient equals reverse KL gradient.
    z = torch.tensor([0.2, -0.4, 1.3], dtype=torch.float64, requires_grad=True)
    lq = torch.log_softmax(torch.tensor([0.8, 0.1, -0.2], dtype=torch.float64), 0)
    lp = z.log_softmax(0)
    exact = (lp.exp() * (lp - lq)).sum()
    g = torch.autograd.grad(exact, z, retain_graph=True)[0]
    surrogate = (lp.exp().detach() * (lp - lq).detach() * lp).sum()
    torch.testing.assert_close(torch.autograd.grad(surrogate, z)[0], g)


def test_splits_disjoint():
    import pandas as pd
    splits = {name: pd.read_parquet(ROOT / f'data/gsm8k/{name}.parquet') for name in ['train','probe','dev','test']}
    seen = set()
    for name, frame in splits.items():
        ids = {row['question_hash'] for row in frame.extra_info}
        assert len(ids) == len(frame)
        assert not ids & seen
        seen.update(ids)
    assert [len(splits[n]) for n in ['train','probe','dev','test']] == [6000,500,973,1319]
