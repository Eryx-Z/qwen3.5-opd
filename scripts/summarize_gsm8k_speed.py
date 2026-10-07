"""Report measured results; exclude first-step warmup from steady-state means."""
import argparse,json,math,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path)
p.add_argument('--steps',type=int,default=5)
p.add_argument('--cases',nargs='+',default=['eager16','eager32','eager32_workers4','graph32'])
args=p.parse_args()
results={}
for name in args.cases:
    root=args.root/name;guard=root/'guard'; rows=[]
    if (guard/'smoke.log').exists():
        for line in (guard/'smoke.log').read_text(errors='replace').splitlines():
            if 'timing_s/step:' not in line:continue
            d={k:float(v) for k,v in re.findall(r'(?:^| - )([\w/.-]+):([-+\d.eE]+)',line)}
            if 'training/global_step' in d:rows.append(d)
    r={'completed_steps':len(rows),'status':'pending/running'}
    if (guard/'outcome.json').exists():
        o=json.loads((guard/'outcome.json').read_text());r.update(reason=o['reason'],child_returncode=o.get('child_returncode'),cleanup=o.get('cleanup'))
        r['status']='passed' if len(rows)==args.steps and o.get('child_returncode')==0 and o.get('cleanup',{}).get('ok') else 'failed/incomplete'
    if rows:
        r['finite_loss_grad']=all(math.isfinite(d[k]) for d in rows for k in ['actor/loss','actor/grad_norm'])
    steady=rows[1:]
    if steady:
        for key in ['timing_s/step','timing_s/gen','timing_s/update_actor','timing_s/update_weights','response_length/mean','response_length/clip_ratio','training/rollout_actor_probs_pearson_corr']:
            values=[d[key] for d in steady if key in d]
            if values:r[key+'_mean']=statistics.mean(values)
        compute_seconds=sum(d['timing_s/step']-d.get('timing_s/save_checkpoint',0) for d in steady)
        r['step_excluding_save_mean']=compute_seconds/len(steady)
        r['checkpoint_save_seconds']=sum(d.get('timing_s/save_checkpoint',0) for d in rows)
        r['whole_step_output_tokens_per_sec']=sum(d['response_length/mean']*32 for d in steady)/compute_seconds
        r['generation_stage_output_tokens_per_sec']=sum(d['response_length/mean']*32 for d in steady)/sum(d['timing_s/gen'] for d in steady)
        r['timing_note']='first step excluded; step_excluding_save removes checkpoint time; gen includes Teacher scoring/postprocessing, not pure Student decode'
    if (guard/'guard.jsonl').exists():
        samples=[json.loads(l) for l in (guard/'guard.jsonl').read_text().splitlines()]
        values=[s['memory']['MemAvailable']/2**30 for s in samples if 'memory' in s]
        if values:r['min_available_gib']=min(values)
    r['checkpoint_saved']=(root/f'output/checkpoints/global_step_{args.steps}/actor/model_world_size_1_rank_0.pt').exists()
    if r['status']=='passed' and (not r['checkpoint_saved'] or not r.get('finite_loss_grad')):
        r['status']='failed/incomplete'
    results[name]=r
print(json.dumps(results,indent=2))
(args.root/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
