#!/usr/bin/env bash
set -euo pipefail

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
MODAL_PROFILE="${MODAL_PROFILE:-rishipython}"
POLL_SECS="${POLL_SECS:-20}"
WEB=0
WEB_PATH="${WEB_PATH:-/tmp/atlas_probe_easyv1.html}"
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
web_path=os.environ.get('WEB_PATH','/tmp/atlas_probe_easyv1.html')

RUNS=[
 ('correlate_1d','probe_easyv2_base_correlate_1d_oe'),
 ('dct_type_I_scipy_fftpack','probe_easyv1_base_dct_type_I_scipy_fftpack_oe'),
 ('dst_type_II_scipy_fftpack','probe_easyv1_base_dst_type_II_scipy_fftpack_oe'),
 ('fft_real_scipy_fftpack','probe_easyv1_base_fft_real_scipy_fftpack_oe'),
 ('shift_2d','probe_easyv1_base_shift_2d_oe'),
 ('convolve_1d','probe_easyv1_base_convolve_1d_oe'),
 ('matrix_multiplication','probe_easyv1_base_matrix_multiplication_oe'),
]

def sh(args):
    env=os.environ.copy(); env['MODAL_PROFILE']=profile
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

def get_txt(path):
    with tempfile.TemporaryDirectory(prefix='probe_') as td:
        local=pathlib.Path(td)/'x'
        p=sh([modal_bin,'volume','get','atlas-openevolve-outputs',path,str(local)])
        if p.returncode!=0 or not local.exists():
            return None
        return local.read_text()

def vol_exists(path):
    p=sh([modal_bin,'volume','ls','atlas-openevolve-outputs',path])
    return p.returncode==0

rows=[]
for prob,run in RUNS:
    txt=get_txt(f'/{run}/oe/evolution_trace.jsonl')
    if not txt:
        st='doing_oe' if vol_exists(f'/{run}') else 'waiting'
        rows.append((prob,run,'n/a','n/a','n/a','n/a','n/a','n/a','n/a',st))
        continue
    recs=[]
    for ln in txt.splitlines():
        ln=ln.strip()
        if not ln: continue
        try: recs.append(json.loads(ln))
        except: pass
    n=len(recs)
    corr=[]
    by_iter={}
    for r in recs:
        m=r.get('child_metrics') or {}
        c=float(m.get('correctness',m.get('correctness_score',0.0)) or 0.0)
        s=float(m.get('speedup',m.get('speedup_score',0.0)) or 0.0)
        if c>=0.99: corr.append(s)
        it=int(r.get('iteration',-1))
        ts=float(r.get('timestamp',0.0) or 0.0)
        if it>=0 and ts>0:
            by_iter[it]=max(by_iter.get(it,0.0),ts)
    passk=(len(corr)/n) if n else 0.0
    best=max(corr) if corr else 0.0
    mean_s=mean(corr) if corr else 0.0
    avg_iter='n/a'; last_iter='n/a'
    if len(by_iter)>=2:
        items=sorted(by_iter.items())
        deltas=[items[i][1]-items[i-1][1] for i in range(1,len(items)) if items[i][1]>items[i-1][1]]
        if deltas:
            avg_iter=f'{mean(deltas):.1f}'
            last_iter=f'{deltas[-1]:.1f}'
    status='done' if n>=40 else 'doing_oe'
    rows.append((prob,run,f'{n}/40',str(len(corr)),f'{passk:.3f}',f'{best:.4f}',f'{mean_s:.4f}',avg_iter,last_iter,status))

print('problem | run_name | progress | correct | pass@k | best_speedup | mean_speedup | avg_iter_s | last_iter_s | status')
print('-'*170)
for r in rows:
    print(f'{r[0]:<26} {r[1]:<52} {r[2]:<9} {r[3]:<7} {r[4]:<7} {r[5]:<12} {r[6]:<12} {r[7]:<10} {r[8]:<11} {r[9]}')

if web:
    css='body{font-family:ui-sans-serif;background:#0d1224;color:#eaf0ff;margin:20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #2a355f;padding:8px}th{background:#192449}.st-doing_oe{background:#5a3d00;color:#ffd067}.st-done{background:#0f4a2a;color:#a7f3c3}.st-waiting{background:#5a1f1f;color:#ffc6c6}'
    parts=['<html><head><meta http-equiv="refresh" content="20"><style>'+css+'</style></head><body>']
    parts.append(f'<h2>Probe Easy v1 Monitor (7 OE base runs)</h2><div>Updated: {escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>')
    parts.append('<table><tr><th>problem</th><th>run_name</th><th>progress</th><th>correct</th><th>pass@k</th><th>best_speedup</th><th>mean_speedup</th><th>avg_iter_s</th><th>last_iter_s</th><th>status</th></tr>')
    for r in rows:
        cls='st-'+r[9]
        parts.append(f"<tr><td>{escape(r[0])}</td><td>{escape(r[1])}</td><td>{escape(r[2])}</td><td>{escape(r[3])}</td><td>{escape(r[4])}</td><td>{escape(r[5])}</td><td>{escape(r[6])}</td><td>{escape(r[7])}</td><td>{escape(r[8])}</td><td class='{cls}'>{escape(r[9])}</td></tr>")
    parts.append('</table></body></html>')
    pathlib.Path(web_path).write_text(''.join(parts))
PY
)
clear || true
echo "Probe Easy v1  profile=$MODAL_PROFILE  poll=${POLL_SECS}s  time=$(date '+%Y-%m-%d %H:%M:%S')"
echo
printf '%s\n' "$SNAPSHOT"
if [[ "$WEB" -eq 1 ]]; then
  echo
  echo "Web: $WEB_PATH"
fi
if [[ "$ONCE" -eq 1 ]]; then break; fi
sleep "$POLL_SECS"
done
