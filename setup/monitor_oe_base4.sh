#!/usr/bin/env bash
set -euo pipefail

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
MODAL_PROFILE="${MODAL_PROFILE:-rishipython}"
POLL_SECS="${POLL_SECS:-20}"
WEB=0
WEB_PATH="${WEB_PATH:-/tmp/atlas_oe_base4.html}"
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --web) WEB=1; shift ;;
    --once) ONCE=1; shift ;;
    --poll-secs) POLL_SECS="$2"; shift 2 ;;
    --web-path) WEB_PATH="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

while true; do
SNAPSHOT=$(MODAL_BIN="$MODAL_BIN" MODAL_PROFILE="$MODAL_PROFILE" WEB="$WEB" WEB_PATH="$WEB_PATH" /Users/rishi/miniconda3/envs/atlas/bin/python - <<'PY'
import json, os, tempfile, pathlib, subprocess
from statistics import mean
from html import escape
from datetime import datetime

modal_bin=os.environ['MODAL_BIN']
profile=os.environ['MODAL_PROFILE']
web=os.environ.get('WEB','0')=='1'
web_path=os.environ.get('WEB_PATH','/tmp/atlas_oe_base4.html')
repo_root=pathlib.Path('/Users/rishi/cs288/atlas')

PERM = [x.strip() for x in pathlib.Path('/Users/rishi/cs288/atlas/setup/stream_permutation_v1.txt').read_text().splitlines() if x.strip()]
BASE_RUNS={
  'fft_convolution':'restart_easyv1_base_fft_convolution_oe',
  'affine_transform_2d':'stream_easyv6_base_affine_transform_2d_oe',
  'base64_encoding':'restart_easyv1_base_base64_encoding_oe',
  'sha256_hashing':'restart_easyv2_base_sha256_hashing_oe',
  'rotate_2d':'restart_easyv1_base_rotate_2d_oe',
}

def sh(args):
    env=os.environ.copy(); env['MODAL_PROFILE']=profile
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

def get_txt(path):
    with tempfile.TemporaryDirectory(prefix='oe4_') as td:
        local=pathlib.Path(td)/'x'
        p=sh([modal_bin,'volume','get','atlas-openevolve-outputs',path,str(local)])
        if p.returncode!=0 or not local.exists():
            return None
        return local.read_text()

def vol_exists(volume, path):
    p=sh([modal_bin,'volume','ls',volume,path])
    return p.returncode==0

def trace_stats(run_name):
    txt=get_txt(f'/{run_name}/oe/evolution_trace.jsonl')
    if not txt:
        return None
    recs=[]
    for ln in txt.splitlines():
        ln=ln.strip()
        if not ln: continue
        try: recs.append(json.loads(ln))
        except: pass
    n=len(recs)
    corr=0
    best=0.0
    speeds=[]
    by_iter={}
    for r in recs:
        m=r.get('child_metrics') or {}
        c=float(m.get('correctness',m.get('correctness_score',0.0)) or 0.0)
        s=float(m.get('speedup',m.get('speedup_score',0.0)) or 0.0)
        if c>=0.99:
            corr += 1
            best=max(best,s)
            speeds.append(s)
        it=int(r.get('iteration',-1))
        ts=float(r.get('timestamp',0.0) or 0.0)
        if it>=0 and ts>0:
            by_iter[it]=max(by_iter.get(it,0.0),ts)
    avg_iter='n/a'; last_iter='n/a'
    if len(by_iter)>=2:
        items=sorted(by_iter.items())
        deltas=[items[i][1]-items[i-1][1] for i in range(1,len(items)) if items[i][1]>items[i-1][1]]
        if deltas:
            avg_iter=f'{mean(deltas):.1f}'
            last_iter=f'{deltas[-1]:.1f}'
    mean_s=mean(speeds) if speeds else 0.0
    return n,corr,best,mean_s,avg_iter,last_iter

rows=[]
for prob in PERM:
    run=BASE_RUNS.get(prob,f'restart_easyv1_base_{prob}_oe')
    model,method='base','OE'
    stats=trace_stats(run)
    if not stats:
        rows.append((model,method,prob,run,'n/a','n/a','n/a','n/a','n/a','n/a','n/a','waiting'))
        continue
    n,c,best,mean_s,avg_iter,last_iter=stats
    passk=(c/n) if n else 0.0
    status='done' if n>=40 else 'running'
    rows.append((model,method,prob,run,f'{n}/40',str(c),f'{passk:.3f}',f'{best:.4f}',f'{mean_s:.4f}',avg_iter,last_iter,status))

print('model | method | problem | run_name | progress | correct | pass@k | best_speedup | mean_speedup | avg_iter_s | last_iter_s | status')
print('-'*180)
for r in rows:
    print(f'{r[0]:<16} {r[1]:<10} {r[2]:<20} {r[3]:<48} {r[4]:<9} {r[5]:<7} {r[6]:<7} {r[7]:<12} {r[8]:<12} {r[9]:<10} {r[10]:<11} {r[11]}')

# Distillation status rows (hardcoded atlas methods from fft_conv)
def get_model_file(path):
    with tempfile.TemporaryDirectory(prefix='oe4m_') as td:
        local=pathlib.Path(td)/'x'
        p=sh([modal_bin,'volume','get','atlas-models',path,str(local)])
        return p.returncode==0 and local.exists()

def distill_status(run_name):
    if get_model_file(f'/{run_name}/training_summary.json'):
        return ('done','done')
    # Try latest checkpoint marker
    p=sh([modal_bin,'volume','ls','atlas-models',f'/{run_name}'])
    if p.returncode!=0:
        return ('n/a','waiting')
    steps=[]
    for ln in p.stdout.splitlines():
        ln=ln.strip()
        if '/checkpoint-' in ln:
            try: steps.append(int(ln.rsplit('checkpoint-',1)[1]))
            except: pass
    if not steps:
        return ('n/a','waiting')
    step=max(steps)
    # if trainer_state exists use max_steps for richer progress
    with tempfile.TemporaryDirectory(prefix='oe4m_') as td:
        local=pathlib.Path(td)/'trainer_state.json'
        g=sh([modal_bin,'volume','get','atlas-models',f'/{run_name}/checkpoint-{step}/trainer_state.json',str(local)])
        if g.returncode==0 and local.exists():
            try:
                d=json.loads(local.read_text())
                gs=int(d.get('global_step',step) or step)
                mx=d.get('max_steps')
                if mx is not None:
                    return (f'{gs}/{int(mx)}','training')
                return (str(gs),'training')
            except: pass
    return (str(step),'training')

def chain_stage(method, target_prob, perm):
    """Infer distill stage between previous OE and target OE."""
    try:
        i = perm.index(target_prob)
    except ValueError:
        return "waiting"
    if i <= 1:
        return "waiting"
    prev_prob = perm[i - 1]
    base = repo_root / "runs" / "stream_perm_v1_auto_chain" / method / prev_prob
    if not base.exists():
        # If previous OE is already done for this method, this leg is queued.
        prev_run = f"stream_perm_v1_{method}_{prev_prob}_oe"
        prev_stats = trace_stats(prev_run)
        if prev_stats and prev_stats[0] >= 40:
            return "queued_for_synth"
        return "waiting"
    synth = base / "synth" / "all_samples.json"
    p1 = base / "p1.jsonl"
    p2 = base / "p2.jsonl"
    best = base / "best.jsonl"
    if not synth.exists():
        return "making_synth"
    if not (p1.exists() and p2.exists() and best.exists()):
        return "building_dataset"
    if method == "sft-best-traj":
        return "training" if not vol_exists("atlas-models", f"/stream_perm_v1_bestspeed_after_{prev_prob}/training_summary.json") else "ready_for_next_oe"
    if method == "dpo-weighted-sft":
        return "training" if not vol_exists("atlas-models", f"/stream_perm_v1_twostage_p2_after_{prev_prob}/training_summary.json") else "ready_for_next_oe"
    if method == "rlm":
        return "building_rlm" if not (base / f"scratchpad_after_{prev_prob}.txt").exists() else "ready_for_next_oe"
    return "waiting"

def synth_progress(method, target_prob, perm):
    try:
        i = perm.index(target_prob)
    except ValueError:
        return None
    if i <= 1:
        return None
    prev_prob = perm[i - 1]
    sdir = repo_root / "runs" / "stream_perm_v1_auto_chain" / method / prev_prob / "synth"
    if not sdir.exists():
        return None
    n = len(list(sdir.glob("iter_*_content.txt")))
    if n <= 0:
        return None
    return f"{n}/40"

def remote_synth_progress(method, target_prob, perm):
    try:
        i = perm.index(target_prob)
    except ValueError:
        return None
    if i <= 1:
        return None
    prev_prob = perm[i - 1]
    synth_name = f"stream_perm_v1_{method}_{prev_prob}_synth"
    p = sh([modal_bin, "volume", "ls", "atlas-openevolve-outputs", f"/synth/{synth_name}"])
    if p.returncode != 0:
        return None
    n = 0
    for ln in p.stdout.splitlines():
        if "iter_" in ln and "_content.txt" in ln:
            n += 1
    if n > 0:
        return f"{n}/40"
    return None

def remote_synth_shard_progress(method, target_prob, perm):
    try:
        i = perm.index(target_prob)
    except ValueError:
        return None
    if i <= 1:
        return None
    prev_prob = perm[i - 1]
    synth_name = f"stream_perm_v1_{method}_{prev_prob}_synth"
    shard_n = {"sft-best-traj": 4, "dpo-weighted-sft": 3, "rlm": 3}.get(method, 3)
    total = 0
    for si in range(shard_n):
        p = sh([modal_bin, "volume", "ls", "atlas-openevolve-outputs", f"/synth/{synth_name}_shard{si}"])
        if p.returncode != 0:
            continue
        for ln in p.stdout.splitlines():
            if "iter_" in ln and "_content.txt" in ln:
                total += 1
    if total > 0:
        return f"{total}/40"
    return None

def app_history():
    p = sh([modal_bin, "app", "list", "--json"])
    if p.returncode != 0:
        return []
    try:
        return json.loads(p.stdout)
    except Exception:
        return []

_APPS = app_history()

def latest_synth_status():
    synth = [a for a in _APPS if str(a.get("Description","")) == "atlas-synth-reasoning"]
    if not synth:
        return None
    # newest first by Created at string (already sortable enough for same format)
    synth.sort(key=lambda a: str(a.get("Created at","")), reverse=True)
    s = synth[0]
    state = str(s.get("State",""))
    created = str(s.get("Created at",""))
    stopped = str(s.get("Stopped at",""))
    return {"state": state, "created": created, "stopped": stopped}

distills=[
    # method, model_run_name, synth_artifact_dir, oe_run_name(optional), oe_problem(optional)
    ('sft-best-traj','atlas_fftconv_bestspeed_only_v4','/synth/synth_fft_nodiff_all_v2','atlas_fftconv_bestspeed_only_v4'),
    ('dpo-weighted-sft','atlas_fftconv_dpo_p2bridge_v4','/synth/synth_fft_nodiff_all_v2','atlas_fftconv_dpo_p2bridge_v4'),
    ('rlm',None,'/rlm_memory_fftconv_synth_v1.txt',None),
]
for m,run,synth_dir,adapter in distills:
    # distill stage
    if m == 'rlm':
        ready = vol_exists('atlas-openevolve-outputs', synth_dir)
        st='done' if ready else 'making_synth'
        prog='ready' if ready else 'n/a'
        print(f'{m:<16} {"distill":<10} {"fft_convolution":<20} {"rlm_memory_fftconv_synth_v1.txt":<48} {prog:<9} {"n/a":<7} {"n/a":<7} {"n/a":<12} {"n/a":<12} {"n/a":<10} {"n/a":<11} {"distill_"+m+"_"+st}')
    else:
        prog,st_train=distill_status(run)
        if st_train in ('training','done'):
            st=st_train
        else:
            st='done' if vol_exists('atlas-openevolve-outputs', synth_dir) else 'making_synth'
            prog='ready' if st=='done' else 'n/a'
        print(f'{m:<16} {"distill":<10} {"fft_convolution":<20} {run:<48} {prog:<9} {"n/a":<7} {"n/a":<7} {"n/a":<12} {"n/a":<12} {"n/a":<10} {"n/a":<11} {"distill_"+m+"_"+st}')

    # adaptive OE stage across remaining problems
    for oe_prob in PERM[1:]:
        oe_run=f'stream_perm_v1_{m}_{oe_prob}_oe'
        oe=trace_stats(oe_run)
        if oe:
            n,c,best,mean_s,avg_iter,last_iter=oe
            st='done' if n>=40 else 'doing_oe'
            passk=(c/n) if n else 0.0
            print(f'{m:<16} {"OE":<10} {oe_prob:<20} {oe_run:<48} {f"{n}/40":<9} {str(c):<7} {f"{passk:.3f}":<7} {f"{best:.4f}":<12} {f"{mean_s:.4f}":<12} {avg_iter:<10} {last_iter:<11} {"distill_"+m+"_"+st}')
        else:
            progress_col = "n/a"
            if vol_exists('atlas-openevolve-outputs', f'/{oe_run}'):
                st = "doing_oe"
            else:
                st = chain_stage(m, oe_prob, PERM)
                if st == "making_synth":
                    syn = latest_synth_status()
                    if syn:
                        if "ephemeral" in syn["state"]:
                            st = "making_synth_running"
                        elif "stopped" in syn["state"]:
                            st = "making_synth_done_waiting_artifacts"
                    # Prefer shard progress first; merged/local synth dirs can lag and look stale.
                    prog = remote_synth_shard_progress(m, oe_prob, PERM)
                    if not prog:
                        prog = remote_synth_progress(m, oe_prob, PERM)
                    if not prog:
                        prog = synth_progress(m, oe_prob, PERM)
                    if prog:
                        progress_col = prog
                        if prog == "40/40":
                            st = "building_dataset"
                        st = f"{st}({prog})"
            print(f'{m:<16} {"OE":<10} {oe_prob:<20} {oe_run:<48} {progress_col:<9} {"n/a":<7} {"n/a":<7} {"n/a":<12} {"n/a":<12} {"n/a":<10} {"n/a":<11} {"distill_"+m+"_"+st}')

if web:
    css='body{font-family:ui-sans-serif;background:#0d1224;color:#eaf0ff;margin:20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #2a355f;padding:8px}th{background:#192449}.st-running{background:#6b4f00;color:#ffd67a}.st-doing_oe{background:#3f2e00;color:#ffd067}.st-training{background:#5a3d00;color:#ffd067}.st-done{background:#0f4a2a;color:#a7f3c3}.st-ready{background:#174a5f;color:#b6ecff}.st-waiting{background:#5a1f1f;color:#ffc6c6}.st-making_synth{background:#5a1f1f;color:#ffc6c6}'
    def status_class(status):
        if 'doing_oe' in status: return 'st-doing_oe'
        if 'training' in status: return 'st-training'
        if 'making_synth' in status: return 'st-making_synth'
        if 'ready' in status: return 'st-ready'
        if status == 'running': return 'st-running'
        if 'done' in status or status == 'done': return 'st-done'
        if 'waiting' in status: return 'st-waiting'
        return ''
    parts=['<html><head><meta http-equiv="refresh" content="20"><style>'+css+'</style></head><body>']
    parts.append(f'<h2>OE Base Monitor (4+affine carryover)</h2><div>Updated: {escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>')
    parts.append('<table><tr><th>model</th><th>method</th><th>problem</th><th>run_name</th><th>progress</th><th>correct</th><th>pass@k</th><th>best_speedup</th><th>mean_speedup</th><th>avg_iter_s</th><th>last_iter_s</th><th>status</th></tr>')
    for r in rows:
        status=r[11]
        cls=status_class(status)
        parts.append(f"<tr><td>{escape(r[0])}</td><td>{escape(r[1])}</td><td>{escape(r[2])}</td><td>{escape(r[3])}</td><td>{escape(r[4])}</td><td>{escape(r[5])}</td><td>{escape(r[6])}</td><td>{escape(r[7])}</td><td>{escape(r[8])}</td><td>{escape(r[9])}</td><td>{escape(r[10])}</td><td class='{cls}'>{escape(status)}</td></tr>")
    # Render atlas rows by mirroring terminal logic above
    atlas_rows=[]
    for m,run,synth_dir,adapter in distills:
        if m == 'rlm':
            ready = vol_exists('atlas-openevolve-outputs', synth_dir)
            st='done' if ready else 'making_synth'
            prog='ready' if ready else 'n/a'
            atlas_rows.append((m,'distill','fft_convolution','rlm_memory_fftconv_synth_v1.txt',prog,'n/a','n/a','n/a','n/a','n/a','n/a','distill_'+m+'_'+st))
        else:
            prog,st_train=distill_status(run)
            if st_train in ('training','done'):
                st=st_train
            else:
                st='done' if vol_exists('atlas-openevolve-outputs', synth_dir) else 'making_synth'
                prog='ready' if st=='done' else 'n/a'
            atlas_rows.append((m,'distill','fft_convolution',run,prog,'n/a','n/a','n/a','n/a','n/a','n/a','distill_'+m+'_'+st))
        for oe_prob in PERM[1:]:
            oe_run=f'stream_perm_v1_{m}_{oe_prob}_oe'
            oe=trace_stats(oe_run)
            if oe:
                n,c,best,mean_s,avg_iter,last_iter=oe
                st='done' if n>=40 else 'doing_oe'
                passk=(c/n) if n else 0.0
                atlas_rows.append((m,'OE',oe_prob,oe_run,f'{n}/40',str(c),f'{passk:.3f}',f'{best:.4f}',f'{mean_s:.4f}',avg_iter,last_iter,'distill_'+m+'_'+st))
            else:
                progress_col = "n/a"
                if vol_exists('atlas-openevolve-outputs', f'/{oe_run}'):
                    st = 'doing_oe'
                else:
                    st = chain_stage(m, oe_prob, PERM)
                    if st == "making_synth":
                        syn = latest_synth_status()
                        if syn:
                            if "ephemeral" in syn["state"]:
                                st = "making_synth_running"
                            elif "stopped" in syn["state"]:
                                st = "making_synth_done_waiting_artifacts"
                        # Prefer shard progress first; merged/local synth dirs can lag and look stale.
                        prog = remote_synth_shard_progress(m, oe_prob, PERM)
                        if not prog:
                            prog = remote_synth_progress(m, oe_prob, PERM)
                        if not prog:
                            prog = synth_progress(m, oe_prob, PERM)
                        if prog:
                            progress_col = prog
                            if prog == "40/40":
                                st = "building_dataset"
                            st = f"{st}({prog})"
                atlas_rows.append((m,'OE',oe_prob,oe_run,progress_col,'n/a','n/a','n/a','n/a','n/a','n/a','distill_'+m+'_'+st))
    for rr in atlas_rows:
        cls=status_class(rr[11])
        parts.append(f"<tr><td>{escape(rr[0])}</td><td>{escape(rr[1])}</td><td>{escape(rr[2])}</td><td>{escape(rr[3])}</td><td>{escape(rr[4])}</td><td>{escape(rr[5])}</td><td>{escape(rr[6])}</td><td>{escape(rr[7])}</td><td>{escape(rr[8])}</td><td>{escape(rr[9])}</td><td>{escape(rr[10])}</td><td class='{cls}'>{escape(rr[11])}</td></tr>")
    parts.append('</table></body></html>')
    pathlib.Path(web_path).write_text(''.join(parts))
PY
)
clear || true
echo "OE Base 4 Monitor  profile=$MODAL_PROFILE  poll=${POLL_SECS}s  time=$(date '+%Y-%m-%d %H:%M:%S')"
echo
printf '%s\n' "$SNAPSHOT"
if [[ "$WEB" -eq 1 ]]; then
  echo
  echo "Web: $WEB_PATH"
fi
if [[ "$ONCE" -eq 1 ]]; then break; fi
sleep "$POLL_SECS"
done
