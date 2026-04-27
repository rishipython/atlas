from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


VLLM_PORT = 8000


def wait_for_vllm(port: int, timeout: int = 900) -> None:
    import urllib.error
    import urllib.request

    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError(f"vLLM did not come up within {timeout}s")


def start_vllm(
    base_model: str,
    *,
    adapter_path: str | None = None,
    port: int = VLLM_PORT,
    gpu_memory_utilization: float = 0.75,
    max_model_len: int = 32768,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base_model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--reasoning-parser",
        "openai_gptoss",
    ]
    if adapter_path:
        cmd.extend(["--enable-lora", "--lora-modules", f"atlas={adapter_path}"])
    print(f"[vllm] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    try:
        wait_for_vllm(port)
        print(f"[vllm] ready at http://localhost:{port}/v1", flush=True)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    return proc


def resolve_adapter_path(adapter: str | None) -> str | None:
    if not adapter:
        return None
    p = Path(adapter)
    if p.exists():
        return str(p.resolve())
    for root in (Path("atlas_models"), Path("outputs"), Path("adapters")):
        candidate = root / adapter
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Could not resolve adapter path from {adapter!r}")
