#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, subprocess, time, tempfile, shutil
from pathlib import Path

REPO=Path('/Users/rishi/cs288/atlas')
MODAL='/Users/rishi/miniconda3/envs/atlas/bin/modal'
PY='/Users/rishi/miniconda3/envs/atlas/bin/python'
PROFILE='rishipython'

def run(cmd, check=True):
    env={**os.environ,'MODAL_PROFILE':PROFILE}
    p=subprocess.run(cmd,cwd=str(REPO),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if check and p.returncode!=0:
        raise RuntimeError(f"failed {' '.join(cmd)}\n{p.stdout}")
    return p.stdout

def load_records(trace):
    return [json.loads(l) for l in Path(trace).read_text().splitlines() if l.strip()]

def write_shards(records, n, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    shards=[[] for _ in range(n)]
    for i,r in enumerate(records):
        shards[i % n].append(r)
    files=[]
    for i,s in enumerate(shards):
        f=outdir/f'shard_{i}.jsonl'
        with f.open('w') as w:
            for r in s:
                w.write(json.dumps(r)+'\n')
        files.append((i,f,len(s)))
    return files

def remote_count(out_name):
    out=run([MODAL,'volume','ls','atlas-openevolve-outputs',f'/synth/{out_name}'],check=False)
    if 'Error' in out or 'No such file' in out:
        return 0
    return sum(1 for ln in out.splitlines() if re.search(r'iter_\d+_content\.txt',ln))

jobs=[
 ('sft-best-traj',4,'runs/stream_perm_v1_auto_chain/sft-best-traj/affine_transform_2d/evolution_trace.jsonl','stream_perm_v1_sft-best-traj_affine_transform_2d_synth_fixv1'),
 ('dpo-weighted-sft',3,'runs/stream_perm_v1_auto_chain/dpo-weighted-sft/affine_transform_2d/evolution_trace.jsonl','stream_perm_v1_dpo-weighted-sft_affine_transform_2d_synth_fixv1'),
 ('rlm',3,'runs/stream_perm_v1_auto_chain/rlm/affine_transform_2d/evolution_trace.jsonl','stream_perm_v1_rlm_affine_transform_2d_synth_fixv1'),
]

launch=[]
for method,n,trace,outbase in jobs:
    recs=load_records(REPO/trace)
    shard_files=write_shards(recs,n,REPO/f'runs/stream_perm_v1_auto_chain/{method}/affine_transform_2d/synth_fix_shards')
    for idx,fp,sz in shard_files:
        out_name=f'{outbase}_sh{idx}'
        cmd=[MODAL,'run','-d','experiment/synth_reasoning.py::main_algotune','--trace-path',str(fp),'--problem-id','affine_transform_2d','--out-name',out_name]
        print(run(cmd))
        launch.append((method,out_name,sz))

print('launched',len(launch),'shard jobs')
# poll
pending={(m,o):sz for m,o,sz in launch}
while pending:
    done=[]
    for (m,o),exp in list(pending.items()):
        c=remote_count(o)
        print(f'{m} {o} {c}/{exp}')
        if c>=exp:
            done.append((m,o))
    for k in done:
        pending.pop(k,None)
    if pending:
        time.sleep(60)

print('all shard synth complete')

# merge into canonical synth dirs
for method,n,trace,outbase in jobs:
    synth_dir=REPO/f'runs/stream_perm_v1_auto_chain/{method}/affine_transform_2d/synth'
    synth_dir.mkdir(parents=True,exist_ok=True)
    for p in synth_dir.glob('iter_*_content.txt'): p.unlink()
    for p in synth_dir.glob('iter_*_context.json'): p.unlink()
    for idx in range(n):
        out_name=f'{outbase}_sh{idx}'
        with tempfile.TemporaryDirectory(prefix='synthdl_') as td:
            local=Path(td)/'payload'
            run([MODAL,'volume','get','atlas-openevolve-outputs',f'/synth/{out_name}',str(local),'--force'])
            for f in Path(td).rglob('iter_*_content.txt'):
                shutil.copy2(f,synth_dir/f.name)
            for f in Path(td).rglob('iter_*_context.json'):
                shutil.copy2(f,synth_dir/f.name)
    samples=[]
    for f in sorted(synth_dir.glob('iter_*_content.txt')):
        it=int(re.match(r'iter_(\d+)_content\.txt$',f.name).group(1))
        samples.append({'task_id':f'iter_{it:03d}','content':f.read_text()})
    (synth_dir/'all_samples.json').write_text(json.dumps(samples,indent=2))
    print(method,'merged',len(samples))

print('merge done')
