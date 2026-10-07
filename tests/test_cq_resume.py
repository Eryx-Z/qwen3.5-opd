"""Resume with production probe refresh, named RNGs, variable lengths and zero-signal fallback."""
import json
import random
from types import SimpleNamespace

import pytest
import torch

from cq_opd import trainer
from test_cq_trainer import TinyAdapter, configuration


@pytest.mark.parametrize('no_signal',[False,True])
def test_resume_with_real_probe_and_variable_trajectories(monkeypatch,tmp_path,no_signal):
    # Stub only token generation; all probe gradient/FD/selector/update code is production.
    def sample(student,tokenizer,row,cap,seed):
        correct=no_signal or seed%2==0
        response=([3,4,5,6,3] if correct else [7,8,9,10,7])[:3+seed%3]
        return SimpleNamespace(input_ids=[1,2]+response,prompt_len=2,response_ids=response,
            response_text='#### 1' if correct else '#### 2',id=row['extra_info']['id'],
            truncated=False,seed=seed)
    monkeypatch.setattr(trainer,'sample',sample)
    # Resume tests use real probe gradients but isolate FD gate acceptance.
    monkeypatch.setattr(trainer,'check_delta_gate',lambda *args,**kwargs:dict(
        report={'candidates':[{'passed':True}]},rollouts=[]))
    rows=[dict(data_source='openai/gsm8k',reward_model={'ground_truth':'1'},
               extra_info={'id':str(i)}) for i in range(10)]
    splits=dict(train=rows[:6],probe=rows[6:],dev=[])
    cfg=configuration();cfg.steps=3;cfg.batch_size=3;cfg.probe_questions=2;cfg.probe_answers=4
    calibration={'mode':'teacher_entropy','tau':2.,'scale':.5,'k':2.}
    delta={'cq_valid':True,'delta':.01}
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    trainer.train(student,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,
        calibration,delta,{},random.Random(42))
    expected=student.model.adapter.weight.detach().clone()
    checkpoint=torch.load(tmp_path/'checkpoint-1.pt',weights_only=False)
    resumed=TinyAdapter(1,True)
    output=tmp_path/'resumed';output.mkdir()
    trainer.train(resumed,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,output,
        calibration,delta,{},random.Random(1),checkpoint)
    torch.testing.assert_close(resumed.model.adapter.weight,expected,rtol=0,atol=0)
    original=[json.loads(x) for x in (tmp_path/'metrics.jsonl').read_text().splitlines()][1:]
    repeated=[json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
    for a,b in zip(original,repeated):
        for key in ('train_mixed_kl','selected_tokens','available_tokens','probe_grad_norm','fallback_reason'):
            assert a[key]==b[key]
    if no_signal:
        assert all(m['fallback_reason']=='no_probe_signal' for m in repeated)
