"""CPU checkpoint and matched-prompt audit after completed speed cases."""
import argparse,hashlib,json,math,re
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path)
p.add_argument('--steps',type=int,default=5)
p.add_argument('--cases',nargs='+',default=['eager16','eager32','eager32_workers4','graph32'])
args=p.parse_args()
result={};reference={}
for name in args.cases:
    root=args.root/name; outcome_path=root/'guard/outcome.json'
    if not outcome_path.exists():continue
    outcome=json.loads(outcome_path.read_text())
    if outcome.get('child_returncode')!=0:
        result[name]={'not_accepted':True,'reason':outcome['reason']};continue
    assert outcome['cleanup']['ok'] and not outcome['cleanup']['remaining']
    c=root/f'output/checkpoints/global_step_{args.steps}/actor'
    s=torch.load(c/'model_world_size_1_rank_0.pt',map_location='cpu',mmap=True,weights_only=False)
    lora={k:v.to_local() if hasattr(v,'to_local') else v for k,v in s.items() if 'lora_' in k}
    assert len(lora)==372 and all('.language_model.layers.' in k for k in lora)
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all().item() for v in lora.values())
    assert all(torch.count_nonzero(v).item()>0 for k,v in lora.items() if 'lora_B' in k)
    opt=torch.load(c/'optim_world_size_1_rank_0.pt',map_location='cpu',mmap=True,weights_only=False)
    assert len(opt['state'])==372 and {int(v['step']) for v in opt['state'].values()}=={args.steps}
    optimizer_finite=all(torch.isfinite(v.to_local() if hasattr(v,'to_local') else v).all().item() for st in opt['state'].values() for v in st.values() if torch.is_tensor(v))
    assert optimizer_finite
    prompt_hashes=[]
    for step in range(1,args.steps+1):
        rows=[json.loads(l) for l in (root/f'output/rollouts/{step}.jsonl').read_text().splitlines()]
        assert len(rows)==32 and all(int(r['step'])==step for r in rows)
        hashes=sorted(hashlib.sha256(r['input'].encode()).hexdigest() for r in rows)
        prompt_hashes.append(hashes)
        if step not in reference:reference[step]=hashes
        assert hashes==reference[step],f'prompt mismatch {name} step{step}'
    metrics=[]
    for l in (root/'guard/smoke.log').read_text(errors='replace').splitlines():
        if 'timing_s/step:' in l:
            d={k:float(v) for k,v in re.findall(r'(?:^| - )([\w/.-]+):([-+\d.eE]+)',l)}
            if 'training/global_step' in d:metrics.append(d)
    assert [d['training/global_step'] for d in metrics]==list(range(1,args.steps+1))
    assert all(math.isfinite(d[k]) and d['actor/grad_norm']>0 for d in metrics for k in ['actor/loss','actor/grad_norm'])
    assert all(d['training/rollout_probs_diff_valid']==1 for d in metrics)
    result[name]={'steps':args.steps,'lora_tensors':372,'language_only':True,'all_finite_fp32':True,'nonzero_B':186,'optimizer_states':372,'optimizer_step':args.steps,'optimizer_finite':True,'rollout_count':args.steps*32,'prompt_batches_match':True,'prompt_batches_sha256':hashlib.sha256(json.dumps(prompt_hashes).encode()).hexdigest(),'rollout_probs_diff_valid_all':True,'clean_exit':True}
    del s,lora,opt
(args.root/'checkpoint-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
