#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path('/Users/rishi/cs288/atlas')
PY = '/Users/rishi/miniconda3/envs/atlas/bin/python'


def is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-prefix', default='stream_perm_v1')
    ap.add_argument('--modal-profile', default=os.environ.get('MODAL_PROFILE', 'rishipython'))
    ap.add_argument('--poll-secs', type=int, default=30)
    ap.add_argument('--check-secs', type=int, default=20)
    args = ap.parse_args()

    state_dir = REPO / 'runs' / f'{args.run_prefix}_auto_chain'
    state_dir.mkdir(parents=True, exist_ok=True)
    sup_pid = state_dir / 'py_supervisor.pid'
    run_pid = state_dir / 'runner.pid'
    hb = state_dir / 'supervisor_heartbeat.json'
    slog = state_dir / 'py_supervisor.log'
    rlog = state_dir / 'runner.log'

    stop = False

    def handle_term(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    sup_pid.write_text(str(os.getpid()))
    with slog.open('a') as f:
        f.write(f"[{time.strftime('%F %T')}] python supervisor started pid={os.getpid()}\n")

    child: subprocess.Popen[str] | None = None

    while not stop:
        child_pid = None
        if child is not None and child.poll() is None:
            child_pid = child.pid
        if not is_alive(child_pid):
            env = os.environ.copy()
            env['MODAL_PROFILE'] = args.modal_profile
            cmd = [
                PY,
                '-u',
                'setup/auto_stream_chain_v1.py',
                '--run-prefix',
                args.run_prefix,
                '--modal-profile',
                args.modal_profile,
                '--poll-secs',
                str(args.poll_secs),
            ]
            with rlog.open('a') as rf:
                child = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=rf, stderr=rf, text=True)
            child_pid = child.pid
            run_pid.write_text(str(child_pid))
            with slog.open('a') as f:
                f.write(f"[{time.strftime('%F %T')}] runner (re)started pid={child_pid}\n")

        hb.write_text(json.dumps({'ts': time.time(), 'supervisor_pid': os.getpid(), 'runner_pid': child_pid or 0}, indent=2))
        time.sleep(args.check_secs)

    if child is not None and child.poll() is None:
        try:
            child.terminate()
            child.wait(timeout=20)
        except Exception:
            child.kill()
    with slog.open('a') as f:
        f.write(f"[{time.strftime('%F %T')}] python supervisor stopping\n")


if __name__ == '__main__':
    main()
