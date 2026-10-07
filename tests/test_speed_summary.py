"""CPU-only proof for the timing report, not a GPU acceptance substitute."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/summarize_gsm8k_speed.py'


def test_warmup_and_checkpoint_excluded(tmp_path):
    case = tmp_path / 'eager16'; guard = case / 'guard'; guard.mkdir(parents=True)
    lines = []
    for step in range(1, 6):
        time = 100 if step == 1 else (50 if step == 5 else 20)
        line = f'training/global_step:{step} - timing_s/step:{time} - timing_s/gen:10 - response_length/mean:5 - actor/loss:0.2 - actor/grad_norm:1'
        if step == 5: line += ' - timing_s/save_checkpoint:30'
        lines.append(line)
    (guard / 'smoke.log').write_text('\n'.join(lines))
    (guard / 'outcome.json').write_text(json.dumps({'reason':'child_exit','child_returncode':0,'cleanup':{'ok':True}}))
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], check=True, capture_output=True)
    x = json.loads((tmp_path / 'comparison.json').read_text())['eager16']
    assert x['step_excluding_save_mean'] == 20
    assert x['checkpoint_save_seconds'] == 30
    assert x['whole_step_output_tokens_per_sec'] == 8
    assert x['generation_stage_output_tokens_per_sec'] == 16
    assert x['finite_loss_grad']
    assert not x['checkpoint_saved']


def test_pending_does_not_claim_success(tmp_path):
    subprocess.run([sys.executable,str(SCRIPT),str(tmp_path)],check=True,capture_output=True)
    x=json.loads((tmp_path/'comparison.json').read_text())
    assert all(r['status']=='pending/running' and r['completed_steps']==0 for r in x.values())


def test_two_step_stress_metrics(tmp_path):
    c=tmp_path/'graph32';g=c/'guard';g.mkdir(parents=True)
    (g/'smoke.log').write_text(
        'training/global_step:1 - timing_s/step:100 - timing_s/gen:50 - response_length/mean:2048 - actor/loss:0.2 - actor/grad_norm:1\n'
        'training/global_step:2 - timing_s/step:90 - timing_s/save_checkpoint:10 - timing_s/gen:60 - response_length/mean:2048 - actor/loss:0.1 - actor/grad_norm:1\n'
    )
    subprocess.run([sys.executable,str(SCRIPT),str(tmp_path),'--steps','2','--cases','graph32'],check=True,capture_output=True)
    x=json.loads((tmp_path/'comparison.json').read_text())
    assert list(x)==['graph32']
    assert x['graph32']['step_excluding_save_mean']==80
    assert x['graph32']['whole_step_output_tokens_per_sec']==819.2


def test_failure_remains_failure(tmp_path):
    g=tmp_path/'graph32/guard';g.mkdir(parents=True)
    (g/'outcome.json').write_text(json.dumps({'reason':'child_exit','child_returncode':1,'cleanup':{'ok':True}}))
    subprocess.run([sys.executable,str(SCRIPT),str(tmp_path)],check=True,capture_output=True)
    x=json.loads((tmp_path/'comparison.json').read_text())['graph32']
    assert x['status']=='failed/incomplete'
