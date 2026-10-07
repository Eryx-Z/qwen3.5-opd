"""Cumulative time includes checkpoint writes and survives trusted resume."""
import json
import random
from types import SimpleNamespace

import pytest
import torch

from cq_opd import trainer
from test_cq_trainer import TinyAdapter, configuration, setup


def test_calibration_job_cost_includes_load_gate_and_binds_artifacts(tmp_path):
    cal={'elapsed_time':2.};delta={'elapsed_time':3.};metadata={'version':'test'}
    path=tmp_path/'calibration_job.json'
    job=dict(total_elapsed_time=15.,includes_model_load_and_mode_gate=True,
        metadata_hash=trainer.digest(metadata),entropy_calibration_hash=trainer.digest(cal),
        delta_calibration_hash=trainer.digest(delta))
    path.write_text(json.dumps(job))
    assert trainer.load_calibration_job(path,cal,delta,metadata)==15.
    job['total_elapsed_time']=4.;path.write_text(json.dumps(job))
    with pytest.raises(ValueError,match='binding mismatch'):
        trainer.load_calibration_job(path,cal,delta,metadata)
    job['total_elapsed_time']=15.;job['delta_calibration_hash']='other';path.write_text(json.dumps(job))
    with pytest.raises(ValueError,match='binding mismatch'):
        trainer.load_calibration_job(path,cal,delta,metadata)


@pytest.mark.parametrize('interrupt',['sidecar_write','checkpoint_publish'])
def test_checkpoint_not_published_until_sidecar_is_ready(monkeypatch,tmp_path,interrupt):
    from pathlib import Path
    splits=setup(monkeypatch);cfg=configuration();cfg.selector='full';cfg.steps=1
    if interrupt=='sidecar_write':
        write=trainer.write_json
        def interrupted(path,value):
            if str(path).endswith('.timing.json.tmp'):
                raise OSError('simulated interruption')
            return write(path,value)
        monkeypatch.setattr(trainer,'write_json',interrupted)
    else:
        replace=Path.replace
        def interrupted(self,target):
            if str(target).endswith('checkpoint-1.pt'):
                raise OSError('simulated interruption')
            return replace(self,target)
        monkeypatch.setattr(Path,'replace',interrupted)
    with pytest.raises(OSError,match='simulated interruption'):
        trainer.train(TinyAdapter(1,True),TinyAdapter(2,False),SimpleNamespace(pad_token_id=0),
            splits,cfg,tmp_path,{'mode':'constant','constant_alpha':.5},
            {'cq_valid':False,'delta':None},{},random.Random(42))
    assert not (tmp_path/'checkpoint-1.pt').exists()  # .tmp or orphan timing is not a committed checkpoint


def test_resume_carries_time_and_final_summary_includes_last_save(monkeypatch,tmp_path):
    splits=setup(monkeypatch)
    cfg=configuration();cfg.selector='full'
    now=[100.]
    def monotonic():
        now[0]+=.001
        return now[0]
    monkeypatch.setattr(trainer.time,'monotonic',monotonic)
    save=torch.save
    def slow_save(*args,**kwargs):
        save(*args,**kwargs)
        now[0]+=7.  # deterministic expensive serialization
    monkeypatch.setattr(torch,'save',slow_save)
    cal={'mode':'constant','constant_alpha':.5,'elapsed_time':2.}
    delta={'cq_valid':False,'delta':None,'elapsed_time':3.}
    student,teacher=TinyAdapter(1,True),TinyAdapter(2,False)
    trainer.train(student,teacher,SimpleNamespace(pad_token_id=0),splits,cfg,tmp_path,
        cal,delta,{},random.Random(42),startup_elapsed_time=2.,imported_calibration_cost=15.)
    summary=json.loads((tmp_path/'training_summary.json').read_text())
    raw=torch.load(tmp_path/'checkpoint-2.pt',weights_only=False)
    completed=trainer.load_checkpoint(tmp_path/'checkpoint-2.pt')
    assert completed['cumulative_elapsed_time']>=raw['cumulative_elapsed_time']+7.
    assert summary['total_elapsed_time']>=completed['cumulative_elapsed_time']
    assert summary['checkpoint_elapsed_time']>=14. and summary['final_checkpoint_included']
    assert summary['imported_calibration_elapsed_time']==15.
    checkpoint=trainer.load_checkpoint(tmp_path/'checkpoint-1.pt')
    output=tmp_path/'resumed';output.mkdir()
    trainer.train(TinyAdapter(1,True),teacher,SimpleNamespace(pad_token_id=0),splits,cfg,output,
        cal,delta,{},random.Random(1),checkpoint,startup_elapsed_time=3.,imported_calibration_cost=15.)
    resumed=json.loads((output/'training_summary.json').read_text())
    assert resumed['previous_elapsed_time']==checkpoint['cumulative_elapsed_time']
    assert resumed['imported_calibration_elapsed_time']==0.  # no duplicate attribution
    assert resumed['total_elapsed_time']>=checkpoint['cumulative_elapsed_time']+3.+7.


def test_checkpoint_timing_sidecar_cannot_be_mixed(monkeypatch,tmp_path):
    splits=setup(monkeypatch);cfg=configuration();cfg.selector='full';cfg.steps=1
    trainer.train(TinyAdapter(1,True),TinyAdapter(2,False),SimpleNamespace(pad_token_id=0),
        splits,cfg,tmp_path,{'mode':'constant','constant_alpha':.5},
        {'cq_valid':False,'delta':None},{},random.Random(42))
    path=tmp_path/'checkpoint-1.pt.timing.json'
    timing=json.loads(path.read_text());timing['checkpoint_id']='not-the-same-save'
    path.write_text(json.dumps(timing))
    with pytest.raises(ValueError,match='sidecar mismatch'):
        trainer.load_checkpoint(tmp_path/'checkpoint-1.pt')
