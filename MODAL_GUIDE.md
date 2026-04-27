# Modal GPU Training — Detailed Guide

If you are using a single Colab A100 instead of Modal, see
`COLAB_GUIDE.md` for the local workflow added to this repo.

A practical reference for training neural networks on [Modal](https://modal.com):
setup, running jobs, GPU types & prices, multi-GPU training, TPU status, cold
starts, and cost-control tips.

All prices below are from Modal's [public pricing page](https://modal.com/pricing)
and are current as of April 2026. Always double-check there before committing to
a long run — pricing is listed **per second** on Modal and the hourly numbers
here are just `/sec × 3600` for readability.

---

## 1. Prerequisites & setup

### One-time

```bash
# Create / activate a local env that has the modal client
conda activate modal
pip install --upgrade modal   # current version is 1.4.x

# Authenticate (opens a browser)
modal token new
```

Your token lives in `~/.modal.toml`. You can have multiple profiles (workspaces);
switch with `modal profile activate <name>`.

### Every session

```bash
conda activate modal
cd /Users/rishi/csc280/droptok/modal_examples
```

That's it — Modal does **not** need any GPU drivers, CUDA, Docker, or PyTorch on
your laptop. Everything GPU-related runs inside Modal's containers.

---

## 2. Anatomy of a Modal training script

Every Modal GPU training script has three pieces:

```python
import modal

# (1) Define the container image remotely — installed once, cached forever
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0")
)

# (2) Create an App object — a namespace for your functions
app = modal.App("my-training-job", image=image)

# (3) Decorate the function that needs a GPU
@app.function(gpu="T4", timeout=600)
def train():
    import torch
    assert torch.cuda.is_available()
    ...

# (4) A local_entrypoint runs on YOUR laptop and calls `.remote()` to kick off
#     the GPU function in Modal's cloud.
@app.local_entrypoint()
def main():
    train.remote()
```

Key distinction:

| Call style | Where it runs |
|---|---|
| `train()` | Locally — plain Python, no GPU |
| `train.remote()` | In Modal's cloud, on the GPU you specified |
| `train.spawn()` | Same as `.remote()` but returns immediately (async) |
| `list(train.map(iter))` | Fan-out: one container per input |

---

## 3. Running jobs

From `modal_examples/`:

```bash
# Foreground — streams logs, Ctrl-C kills the job
modal run 01_mnist_mlp.py

# Pass function args
modal run 01_mnist_mlp.py --epochs 5

# Detached — keeps running after you close the terminal
modal run --detach 02_cifar10_cnn.py

# Deploy as a long-lived app (for cron / web endpoints / reused warm containers)
modal deploy 02_cifar10_cnn.py
```

Useful inspection commands:

```bash
modal app list                    # what's running
modal app logs <app-name>         # tail logs
modal app stop <app-name>         # kill a deployed app
modal volume list                 # persistent storage
modal volume ls  <vol-name>       # files inside a volume
modal volume get <vol-name> <remote_path> <local_path>
```

---

## 4. GPU types and prices

All GPUs below are specified with the `gpu=` argument on `@app.function`:

```python
@app.function(gpu="A100-80GB")
```

Prices (per-second × 3600, April 2026):

| `gpu=` string | VRAM | Price / sec | Price / hr | Good for |
|---|---|---|---|---|
| `"T4"` | 16 GB | $0.000164 | **$0.59** | Tiny models, CS homework, CI, smoke tests |
| `"L4"` | 24 GB | $0.000222 | **$0.80** | Inference for 7B-class models, light training |
| `"A10"` | 24 GB | $0.000306 | **$1.10** | Mid-range training; Stable Diffusion 1.5 |
| `"L40S"` | 48 GB | $0.000542 | **$1.95** | Recommended default for inference |
| `"A100"` / `"A100-40GB"` | 40 GB | $0.000583 | **$2.10** | Real training jobs, fine-tuning |
| `"A100-80GB"` | 80 GB | $0.000694 | **$2.50** | Larger models / longer context |
| `"RTX-PRO-6000"` | 96 GB | $0.000842 | **$3.03** | Huge VRAM at Ada-generation cost |
| `"H100"` / `"H100!"` | 80 GB | $0.001097 | **$3.95** | FP8 training, LLM fine-tuning |
| `"H200"` | 141 GB | $0.001261 | **$4.54** | Like H100, bigger HBM (141 GB @ 4.8 TB/s) |
| `"B200"` / `"B200+"` | 180 GB | $0.001736 | **$6.25** | Blackwell flagship; fastest available |

Automatic upgrades you should know about:

- `gpu="A100"` may silently run on an 80 GB A100 (at the 40 GB price — free win).
- `gpu="H100"` may silently run on an H200 (at the H100 price). Use `"H100!"` to opt out, e.g. for benchmarking.
- `gpu="B200+"` opts into B200 or B300 (billed as B200). Requires CUDA 13.0+.

Fallbacks — try preferred GPU first, fall back if unavailable:

```python
@app.function(gpu=["H100", "A100-80GB", "A100-40GB:2"])
def train(): ...
```

There's also a small CPU + memory charge while the container is running (CPU:
~$0.047/core/hr, memory: ~$0.008/GiB/hr), but for GPU jobs this is rounding
error next to the GPU bill.

---

## 5. Multiple GPUs on one container

Append `:N` to the GPU string to attach multiple GPUs to the **same container**
(same physical machine, NVLink where available):

```python
@app.function(gpu="A100-80GB:4", timeout=3600)
def train_70b():
    ...
```

Supported counts (from Modal docs):

| GPU | Max per container |
|---|---|
| T4, L4, L40S, A100, H100, H200, B200 | up to **8** |
| A10 | up to **4** |

> Requesting >2 GPUs per container typically means longer queueing time — the
> scheduler has to find a node with that many free of the same type.

### Running DDP / torchrun inside a multi-GPU container

PyTorch's preferred multi-GPU launcher re-executes the script, which fights
with Modal's entrypoint machinery. The recommended pattern is to spawn
`torchrun` as a subprocess from your Modal function:

```python
@app.function(gpu="A100:4", timeout=3600)
def launch_ddp():
    import subprocess, sys
    subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=4", "train.py"],
        stdout=sys.stdout, stderr=sys.stderr, check=True,
    )
```

Where `train.py` is shipped into the image (via `image.add_local_file(...)` or
`image.add_local_dir(...)`) and uses standard `torch.distributed` /
`DistributedDataParallel` / Lightning's `ddp` strategy inside.

For PyTorch Lightning specifically, if you call `Trainer.fit()` directly from
the Modal function (instead of subprocess), use
`strategy="ddp_spawn"` or `"ddp_notebook"` — **not** `"ddp"` — because Lightning
re-execs `sys.argv[0]`, which Modal doesn't like.

### Multi-node (beyond one container)

Single-container multi-GPU is GA. **Multi-node** training (spanning several
containers / physical machines with a collective backend like NCCL over
network) is in private beta at the time of writing. If you need it, email
`support@modal.com` to request access.

---

## 6. TPUs — not available on Modal

**Modal does not offer TPUs.** It is a pure NVIDIA GPU platform. If your code
needs Google TPUs (v4, v5e, v5p, Trillium, etc.), use [Google Cloud
TPU](https://cloud.google.com/tpu) or [Kaggle](https://www.kaggle.com/) /
[Colab](https://colab.research.google.com/) directly — Modal can't help there.

Practical translations if you were considering a TPU:

- **JAX workloads** → run on H100 / H200 / B200 with `jax[cuda12]` installed; JAX has excellent CUDA support.
- **Flax / TPU-only research code** → swap `jax.devices("tpu")` for `jax.devices("gpu")`; most high-level APIs are identical.
- **`torch_xla`** → same idea: use vanilla CUDA PyTorch on Modal instead.

If you want a TPU-like *parallelism story* (model-parallel, pipeline-parallel)
on Modal, reach for multi-GPU H100/B200 containers (section 5) and FSDP /
DeepSpeed ZeRO-3 / tensor parallel frameworks.

---

## 7. Cold start & startup time

Modal bills per-second only while the container is actively running a request,
but wall-clock time has a few components you should understand:

| Phase | Typical duration | Billed? |
|---|---|---|
| **Image build** (first deploy or when image changes) | 30 s – 5 min for typical DL images | **No** |
| **Image pull** to a fresh node | 1 – 20 s (depends on size & layer cache) | **No** |
| **Container boot** (Modal's optimized stack) | **~1 s** | No (until your code runs) |
| **Global scope + `@modal.enter`** (e.g. loading model weights) | seconds to minutes | **Yes** |
| **Your function body** | your call | **Yes** |
| **Scaledown idle window** (default 60 s of keep-warm) | up to `scaledown_window` | **Yes** |

A "cold start" combines the pull + boot + model-load phases and commonly lands
around **5–30 s** for a typical PyTorch image. The very first run after you
change the image is slower because the image has to be built.

Things that help:

- Pre-download weights into the image with a build step (`image.run_function(...)`) instead of downloading at runtime.
- Keep warm containers with `min_containers=1` on the function decorator if you need sub-second responses.
- Use `modal.Volume` for datasets you'd otherwise re-download every run (see `02_cifar10_cnn.py`).
- Experimental: `enable_memory_snapshot=True` + `enable_gpu_snapshot=True` on a *deployed* function can cut cold starts to sub-second by checkpointing the container post-model-load. See Modal's ["GPU memory snapshots"](https://modal.com/docs/examples/gpu_snapshot) docs.

---

## 8. Putting it all together — pattern cheatsheet

Cheapest possible smoke test (the `01_mnist_mlp.py` pattern):

```python
@app.function(gpu="T4", timeout=600)
def train(): ...
```

Solo fine-tuning of a small/medium LLM (~7B):

```python
@app.function(gpu="A100-80GB", timeout=3600)
def finetune_7b(): ...
```

Multi-GPU FSDP training of a medium LLM:

```python
@app.function(gpu="H100:4", timeout=4 * 3600, volumes={"/ckpt": ckpt_vol})
def fsdp_train():
    import subprocess, sys
    subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=4", "train.py"],
        check=True, stdout=sys.stdout, stderr=sys.stderr,
    )
```

Fallback chain so cheaper capacity is used when available:

```python
@app.function(gpu=["A100-80GB", "A100-40GB:2", "H100"])
def train(): ...
```

---

## 9. Cost-control checklist

- Default to **T4** while debugging; only move to A100/H100 once the code clearly runs end-to-end.
- Always set an explicit `timeout=` on GPU functions. Nothing burns credit faster than an infinite loop on an H100.
- Use `modal.Volume` for datasets and checkpoints — re-downloading a 170 MB dataset on every run is wasted GPU-seconds.
- Run a single `--epochs 1` smoke pass before launching a 10-hour job.
- When iterating locally on non-GPU code paths, call the function directly (no `.remote()`) so you don't spin up a GPU at all.
- Use `modal app list` and `modal app stop` to clean up detached/deployed jobs you forgot about.
- Academic credits: if you're on a `.edu`, Modal offers up to **$10k in free credits** for students/researchers — worth applying for before your first real training run.

---

## 10. Where to go next

- Modal GPU docs: <https://modal.com/docs/guide/gpu>
- Modal pricing: <https://modal.com/pricing>
- Cold start guide: <https://modal.com/docs/guide/cold-start>
- Example: LLM fine-tuning on Modal: <https://modal.com/docs/examples/llm-finetuning>
- Example: multi-GPU LLM inference: <https://modal.com/docs/examples/llm_inference>
- GPU glossary (very good reference on what an SM / HBM / etc. actually is): <https://modal.com/gpu-glossary/readme>
