#!/usr/bin/env bash
set -euo pipefail

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
MODAL_PROFILE="${MODAL_PROFILE:-rishipython}"
POLL_SECS="${POLL_SECS:-30}"
ONCE="${1:-}"

python_cmd=/Users/rishi/miniconda3/envs/atlas/bin/python

while true; do
  SNAPSHOT="$($python_cmd - <<'PY'
import json, os, subprocess, tempfile, pathlib
from statistics import mean

modal_bin=os.environ.get('MODAL_BIN','/Users/rishi/miniconda3/envs/atlas/bin/modal')
modal_profile=os.environ.get('MODAL_PROFILE','rishipython')

def sh(args):
    env=os.environ.copy(); env['MODAL_PROFILE']=modal_profile
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

def vol_get_outputs(remote_path):
    with tempfile.TemporaryDirectory(prefix='mon_fixed_') as td:
        local=pathlib.Path(td)/'x'
        p=sh([modal_bin,'volume','get','atlas-openevolve-outputs',remote_path,str(local)])
        if p.returncode!=0 or not local.exists():
            return None
        return local.read_text()

def vol_get_models(remote_path):
    with tempfile.TemporaryDirectory(prefix='mon_fixed_') as td:
        local=pathlib.Path(td)/'x'
        p=sh([modal_bin,'volume','get','atlas-models',remote_path,str(local)])
        if p.returncode!=0 or not local.exists():
            return None
        return local.read_text()

def vol_ls_models(remote_path):
    p=sh([modal_bin,'volume','ls','atlas-models',remote_path])
    if p.returncode!=0:
        return []
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]

def parse_oe(run):
    txt=vol_get_outputs(f'/{run}/oe/evolution_trace.jsonl')
    if not txt:
        return ('n/a','n/a','n/a','n/a','waiting')
    recs=[]
    for ln in txt.splitlines():
        ln=ln.strip()
        if not ln: continue
        try: recs.append(json.loads(ln))
        except: pass
    n=len(recs)
    corr=[]
    for r in recs:
        m=r.get('child_metrics') or {}
        c=float(m.get('correctness', m.get('correctness_score',0.0)) or 0.0)
        s=float(m.get('speedup', m.get('speedup_score',0.0)) or 0.0)
        if c>=0.99:
            corr.append(s)
    passk=(len(corr)/n) if n else 0.0
    best=max(corr) if corr else 0.0
    mean_s=mean(corr) if corr else 0.0
    status='done' if n>=40 else 'running'
    return (f'{n}/40', str(len(corr)), f'{passk:.3f}', f'{best:.4f}/{mean_s:.4f}', status)

def parse_distill(run):
    s=vol_get_models(f'/{run}/training_summary.json')
    if s:
        return ('done','done')
    ents=vol_ls_models(f'/{run}')
    ck=[]
    for e in ents:
        if '/checkpoint-' in e:
            try: ck.append(int(e.rsplit('checkpoint-',1)[1]))
            except: pass
    if not ck:
        return ('n/a','waiting')
    step=max(ck)
    ts=vol_get_models(f'/{run}/checkpoint-{step}/trainer_state.json')
    if ts:
        try:
            d=json.loads(ts)
            step=int(d.get('global_step',step) or step)
            mx=d.get('max_steps')
            if mx is not None:
                return (f'{step}/{int(mx)}','training')
        except: pass
    return (str(step),'training')

rows=[]
# Hardcoded OE(base) rows
rows.append(('base','OE','fft_convolution',*parse_oe('stream_oe5_methods_v1_base_fft_convolution_oe')))
rows.append(('base','OE','affine_transform_2d',*parse_oe('stream_easyv6_base_affine_transform_2d_oe')))
rows.append(('base','OE','base64_encoding',*parse_oe('stream_easyv9_base_base64_encoding_oe')))
rows.append(('base','OE','sha256_hashing',*parse_oe('stream_easyv9_base_sha256_hashing_oe')))
rows.append(('base','OE','rotate_2d',*parse_oe('stream_easyv9_base_rotate_2d_oe')))

# Hardcoded distillation rows (fft_conv)
prog,st=parse_distill('atlas_fftconv_bestspeed_only_v3')
rows.append(('sft-best-traj','distill','fft_convolution',prog,'n/a','n/a','n/a',st))
prog,st=parse_distill('atlas_fftconv_dpo_phase1_v3')
rows.append(('dpo-weighted-sft','distill','fft_convolution',prog,'n/a','n/a','n/a',st))
prog,st=parse_distill('dpo-weighted-sft','distill','fft_convolution') if False else (None,None)
prog,st=parse_distill('atlas_fftconv_dpo_p2bridge_v3')
rows.append(('dpo-weighted-sft-p2','distill','fft_convolution',prog,'n/a','n/a','n/a',st))
prog,st=parse_distill('atlas_fftconv_phase1_only_v3')
rows.append(('phase1-only','distill','fft_convolution',prog,'n/a','n/a','n/a',st))
prog,st=parse_distill('atlas_fftconv_phase2_only_v3')
rows.append(('phase2-only','distill','fft_convolution',prog,'n/a','n/a','n/a',st))

print('model | method | problem | progress | correct | pass@k | best/mean_speedup | status')
print('-'*120)
for r in rows:
    print(f'{r[0]:<16} {r[1]:<8} {r[2]:<22} {r[3]:<10} {r[4]:<7} {r[5]:<7} {r[6]:<18} {r[7]}')
PY
)"
  clear || true
  echo "Hardcoded Monitor profile=$MODAL_PROFILE poll=${POLL_SECS}s time=$(date '+%Y-%m-%d %H:%M:%S')"
  echo
  echo "$SNAPSHOT"
  if [[ "$ONCE" == "--once" ]]; then
    break
  fi
  sleep "$POLL_SECS"
done
