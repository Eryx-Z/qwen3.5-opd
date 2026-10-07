"""Production probe gradient and calibration protocol guards on CPU."""
from types import SimpleNamespace
import random

import pandas as pd
import pytest
import torch

from cq_opd import trainer
from cq_opd.blocks import make_blocks
from test_cq_trainer import TinyAdapter, configuration


def test_production_probe_gradient_preserves_existing_dot_grad(monkeypatch):
    counter=iter(range(100))
    def sample(student,tokenizer,row,cap,seed):
        correct=next(counter)%2==0
        response=[3,4,5,6] if correct else [7,8,9,10]
        return SimpleNamespace(input_ids=[1,2]+response,prompt_len=2,response_ids=response,
            response_text='#### 1' if correct else '#### 2',id=row['extra_info']['id'],
            truncated=False,seed=seed)
    monkeypatch.setattr(trainer,'sample',sample)
    student=TinyAdapter(1,True)
    parameter=student.model.adapter.weight
    parameter.grad=torch.ones_like(parameter)
    before=parameter.grad.clone()
    cfg=configuration();cfg.probe_questions=2;cfg.probe_answers=4
    rows=[dict(data_source='openai/gsm8k',reward_model={'ground_truth':'1'},extra_info={'id':str(i)}) for i in range(4)]
    result,stats=trainer.direction(student,SimpleNamespace(pad_token_id=0),rows,cfg,3,random.Random(42))
    assert result.valid and result.source_step==3 and stats['probe_success_rate']==.5
    assert stats['informative_question_fraction']==1
    assert torch.isfinite(result.tensors[0]).all()
    torch.testing.assert_close(parameter.grad,before,rtol=0,atol=0)
    torch.testing.assert_close(torch.sqrt(sum(t.double().square().sum() for t in result.tensors)),
                               torch.tensor(1.,dtype=torch.float64),rtol=1e-6,atol=1e-6)


def test_native_chat_template_returns_token_ids_not_batchencoding():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'<unk>':0,'<pad>':1,'hello':2},unk_token='<unk>')),unk_token='<unk>',pad_token='<pad>',
        chat_template='{% for message in messages %}{{ message["content"] }}{% endfor %}')
    ids=trainer.prompt_ids(tokenizer,{'prompt':[{'role':'user','content':'hello'}]})
    assert ids==[2] and all(isinstance(token,int) for token in ids)


def test_only_exact_guard_transport_argument_is_removed(monkeypatch):
    monkeypatch.setenv('VERL_OPD_GUARD_RUN','test')
    monkeypatch.setenv('RAY_TMPDIR','/tmp/cq-ray')
    monkeypatch.setattr('sys.argv',['trainer','--steps','2',
        '+ray_kwargs.ray_init._temp_dir=/tmp/cq-ray','+ray_kwargs.ray_init._temp_dir=/other'])
    assert trainer.guarded_arguments()==['--steps','2','+ray_kwargs.ray_init._temp_dir=/other']
    monkeypatch.delenv('VERL_OPD_GUARD_RUN')
    assert len(trainer.guarded_arguments())==4


def test_global_ids_disjoint_even_with_distinct_content(tmp_path):
    for i,name in enumerate(('train','probe','dev','test')):
        pd.DataFrame([{'extra_info':{'id':'same' if i<2 else name,'question_hash':str(i)}}]).to_parquet(tmp_path/f'{name}.parquet')
    with pytest.raises(ValueError,match='overlap'):
        trainer.load_splits(tmp_path)


def test_overlap_ranks_all_blocks_including_negative_and_noise():
    valid=torch.ones(1,10,dtype=torch.bool)
    blocks=make_blocks(valid,1)
    a=torch.tensor([-1.,-2.,-3.,-4.,-5.,-6.,-7.,-8.,-9.,-10.])
    b=torch.tensor([-2.,-1.,-3.,-4.,-5.,-6.,-7.,-8.,-9.,-10.])
    corr,overlap=trainer.rank_agreement(a,b,blocks,.2)
    assert corr>.8 and overlap==1
    c,o=trainer.rank_agreement(torch.zeros(10),torch.zeros(10),blocks,.2)
    assert c is None and o==1  # ties are reproducible but NOT evidence of a valid FD signal
