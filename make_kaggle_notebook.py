"""Generate a self-contained Kaggle notebook that runs the CAD-SEL grid.

The notebook embeds the `cadsel` package (base64 so nothing is mangled by
notebook quoting), pulls the dataset from the Hugging Face mirror, and drives
two training streams — one per T4 — with live per-epoch output.

    python make_kaggle_notebook.py --models fasterrcnn_resnet --out kaggle_resnet.ipynb
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_FILES = [
    "cadsel/__init__.py",
    "cadsel/config.py",
    "cadsel/data.py",
    "cadsel/models.py",
    "cadsel/metrics.py",
    "cadsel/engine.py",
    "cadsel/train.py",
    "cadsel/run_all.py",
    "cadsel/report.py",
    "cadsel/describe.py",
    "cadsel/visualize.py",
]

HF_REPO = "RohanRamesh/CAD-SEL"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(
    models: list[str], epochs: int, patience: int, workers: int, batch: int
) -> dict:
    payload = {}
    for rel in PACKAGE_FILES:
        raw = (PROJECT_ROOT / rel).read_bytes()
        payload[rel] = base64.b64encode(raw).decode("ascii")
    encoded = json.dumps(payload, indent=0)

    models_literal = json.dumps(models)

    cells = [
        md(
            f"""# CAD-SEL baselines on Kaggle

Reproduces the technical-validation grid from *CAD-SEL: A dual-modal colonoscopy
dataset of subepithelial lesion* (Sci Data 2026) for **{', '.join(models)}**.

**Before running — set these in the notebook sidebar:**

1. **Accelerator → GPU T4 x2** (the notebook drives both GPUs)
2. **Internet → On** (needed for the dataset and the COCO-pretrained weights)

Runs WLE on `cuda:0` and EUS on `cuda:1`, both label schemes on each, so four
runs finish in roughly the time of two. Per-epoch lines stream live below.
"""
        ),
        code(
            f'''# ---- configuration -------------------------------------------------------
MODELS   = {models_literal}   # detectors to train in this notebook
EPOCHS   = {epochs}
PATIENCE = {patience}          # early stop after N epochs with no val AP@0.5 gain
WORKERS  = {workers}           # dataloader workers per stream (Kaggle gives 4 vCPU)
BATCH    = {batch}             # T4 has 16 GB; 4 is the 8 GB default, 16 matches the reference
HF_REPO  = "{HF_REPO}"

DATA_DIR = "/kaggle/temp/cadsel-data"
WORK     = "/kaggle/working"

import os, subprocess, sys, time
os.makedirs(DATA_DIR, exist_ok=True)

import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu)
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{{i}}  {{p.name}}  {{p.total_memory/2**30:.1f}} GiB")
if n_gpu == 0:
    raise SystemExit("No GPU. Set Accelerator to 'GPU T4 x2' in the sidebar.")
if n_gpu == 1:
    print("\\nOnly one GPU visible - both streams will share it (slower but works).")
'''
        ),
        code(
            '''# ---- optional HF token ---------------------------------------------------
# The repo is public, so a token is not required - but an authenticated clone
# gets far higher rate limits. To use one: Add-ons -> Secrets -> add HF_TOKEN.
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN found in Kaggle secrets")
except Exception:
    print("no HF_TOKEN secret - proceeding anonymously")
'''
        ),
        code(
            '''# ---- dataset (~11 GB) ----------------------------------------------------
# The mirror holds 9,824 files. Fetching them one-by-one trips Hugging Face's
# per-file rate limit (HTTP 429 with multi-minute backoff), so clone instead:
# git pulls every label in a single pack and git-lfs batches the images.
import shutil

def sh(cmd, check=False, quiet=False):
    if not quiet:
        print("+", cmd.replace(HF_TOKEN, "***") if HF_TOKEN else cmd)
    r = subprocess.run(cmd, shell=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed: {r.returncode}")
    return r.returncode

def count_data(root):
    """Count real images, not LFS pointers.

    A pointer is a ~130-byte text file still named `.tiff`, so counting by
    extension alone reports success while every image is unusable.
    """
    tif = lbl = pointers = 0
    tif_bytes = 0
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in files:
            if f.endswith((".tiff", ".tif")):
                size = os.path.getsize(os.path.join(dirpath, f))
                tif_bytes += size
                if size < 1024:
                    pointers += 1
                else:
                    tif += 1
            elif f.endswith(".txt"):
                lbl += 1
    return tif, lbl, pointers, tif_bytes

if subprocess.run("git lfs version", shell=True, capture_output=True).returncode != 0:
    sh("apt-get -qq update && apt-get -qq install -y git-lfs", quiet=True)
sh("git lfs install --skip-repo", quiet=True)

url = f"https://huggingface.co/datasets/{HF_REPO}"
if HF_TOKEN:
    url = f"https://user:{HF_TOKEN}@huggingface.co/datasets/{HF_REPO}"

t0 = time.time()
if not os.path.isdir(f"{DATA_DIR}/.git"):
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    sh(f"GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 {url} {DATA_DIR}", check=True)

tif, lbl, ptr, nbytes = count_data(DATA_DIR)
print(f"after clone: {tif} real images, {ptr} LFS pointers, {lbl} labels")

# Always pull at least once: the clone deliberately skipped the payloads.
for attempt in range(1, 7):
    if tif >= 4912 and ptr == 0:
        break
    print(f"\\n--- git lfs pull, attempt {attempt} "
          f"({(time.time()-t0)/60:.1f} min elapsed) ---")
    sh(f"cd {DATA_DIR} && git lfs pull")
    tif, lbl, ptr, nbytes = count_data(DATA_DIR)
    print(f"  now {tif} real images / {ptr} pointers / {nbytes/2**30:.1f} GiB")
else:
    print("WARNING: pointers remain after 6 attempts")

print(f"\\nfinal: {tif} images, {lbl} labels, {nbytes/2**30:.1f} GiB "
      f"in {(time.time()-t0)/60:.1f} min (expect 4912 / 4912 / ~11 GiB)")
'''
        ),
        code(
            '''# ---- fallback + hard verification ---------------------------------------
tif, lbl, ptr, nbytes = count_data(DATA_DIR)
if tif < 4912 or lbl < 4912:
    print(f"incomplete ({tif} images, {ptr} pointers) - falling back to snapshot_download")
    !pip install -q "huggingface_hub>=0.34" 2>&1 | tail -1
    from huggingface_hub import snapshot_download
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=HF_REPO, repo_type="dataset", local_dir=DATA_DIR,
                max_workers=4,            # low concurrency avoids the 429s
                token=HF_TOKEN,
            )
            break
        except Exception as e:
            print(f"attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(30)
    tif, lbl, ptr, nbytes = count_data(DATA_DIR)
    print(f"after fallback: {tif} images / {ptr} pointers / {lbl} labels")

# Locate the Images/Labels roots regardless of how the mirror nests them.
for root, dirs, _ in os.walk(DATA_DIR):
    if "Images" in dirs and "Labels" in dirs:
        DATA_ROOT = root
        break
else:
    raise SystemExit("Could not find Images/ and Labels/ in the download")
print("DATA_ROOT =", DATA_ROOT)

# Actually decode a sample. Counting files is not enough - LFS pointers are
# still named .tiff, which is exactly how the first run got four failed jobs.
import random
from PIL import Image

all_tiffs = []
for dirpath, dirnames, files in os.walk(f"{DATA_ROOT}/Images"):
    dirnames[:] = [d for d in dirnames if d != ".git"]
    all_tiffs += [os.path.join(dirpath, f) for f in files if f.endswith((".tiff", ".tif"))]

sample = random.Random(0).sample(all_tiffs, min(40, len(all_tiffs)))
bad = []
for p in sample:
    try:
        with Image.open(p) as im:
            im.convert("RGB").load()
    except Exception as e:
        bad.append((p, type(e).__name__))

print(f"\\ndecode check: {len(sample)-len(bad)}/{len(sample)} sample images OK")
if bad:
    for p, e in bad[:3]:
        print("  FAILED", e, p, os.path.getsize(p), "bytes")
    raise SystemExit(
        "Images are not real TIFFs (likely unresolved LFS pointers). "
        "Re-run the download cell before training."
    )
print(f"dataset verified: {tif} images, {lbl} labels, {nbytes/2**30:.1f} GiB")
'''
        ),
        code(
            f'''# ---- write the cadsel package -------------------------------------------
import base64, json, pathlib

PACKAGE_B64 = json.loads(r"""{encoded}""")

for rel, b64 in PACKAGE_B64.items():
    dest = pathlib.Path(WORK) / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64))
print(f"wrote {{len(PACKAGE_B64)}} files under {{WORK}}/cadsel")
'''
        ),
        code(
            '''# ---- sanity check: does the data match Table 2 of the paper? -------------
os.chdir(WORK)
r = subprocess.run(
    [sys.executable, "-m", "cadsel.describe", "--data-root", DATA_ROOT],
    capture_output=True, text=True,
)
print(r.stdout[:4000] or r.stderr[-4000:])
'''
        ),
        code(
            '''# ---- launch both streams, one per GPU, and tail them live ---------------
os.chdir(WORK)
os.makedirs(f"{WORK}/logs", exist_ok=True)

def launch(modality, device, tag):
    cmd = [
        sys.executable, "-u", "-m", "cadsel.run_all",
        "--models", *MODELS,
        "--modalities", modality,
        "--epochs", str(EPOCHS),
        "--early-stop-patience", str(PATIENCE),
        "--num-workers", str(WORKERS),
        "--batch-size", str(BATCH),
        "--data-root", DATA_ROOT,
        "--device", device,
        "--output-dir", f"{WORK}/runs",
        "--results-dir", f"{WORK}/results",
        "--no-progress",
    ]
    log = open(f"{WORK}/logs/{tag}.log", "w", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[{tag}] pid {proc.pid} on {device}: {' '.join(MODELS)} / {modality}")
    return proc, f"{WORK}/logs/{tag}.log"

gpu_b = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
jobs = [launch("WLE", "cuda:0", "wle"), launch("EUS", gpu_b, "eus")]

# Stream both logs until each finishes.
offsets = {path: 0 for _, path in jobs}
start = time.time()
while any(p.poll() is None for p, _ in jobs):
    for _, path in jobs:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            fh.seek(offsets[path])
            chunk = fh.read()
            offsets[path] = fh.tell()
        for line in chunk.splitlines():
            if line.strip():
                tag = "WLE" if path.endswith("wle.log") else "EUS"
                print(f"[{tag}] {line}", flush=True)
    time.sleep(20)

for _, path in jobs:
    with open(path) as fh:
        fh.seek(offsets[path])
        rest = fh.read()
    for line in rest.splitlines():
        if line.strip():
            tag = "WLE" if path.endswith("wle.log") else "EUS"
            print(f"[{tag}] {line}", flush=True)

print(f"\\nboth streams done in {(time.time()-start)/60:.1f} min")
print("exit codes:", [p.returncode for p, _ in jobs])
'''
        ),
        code(
            '''# ---- tables + package the outputs ---------------------------------------
r = subprocess.run(
    [sys.executable, "-m", "cadsel.report",
     "--runs-dir", f"{WORK}/runs", "--results-dir", f"{WORK}/results"],
    capture_output=True, text=True,
)
print(r.stdout or r.stderr[-3000:])

# Checkpoints are large; keep them out of the zip unless you want them.
!cd {WORK} && zip -qr cadsel_results.zip results runs -x "*.pth" && ls -lh cadsel_results.zip
print("\\nDownload cadsel_results.zip from the notebook Output tab.")
print("Merge with the other account's runs/ folder, then rerun cadsel.report.")
'''
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the Kaggle notebook")
    p.add_argument("--models", nargs="+", default=["fasterrcnn_resnet"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "kaggle" / "cadsel_kaggle.ipynb")
    args = p.parse_args()

    nb = build_notebook(args.models, args.epochs, args.patience, args.workers, args.batch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    size = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size:.0f} KB) for models={args.models} epochs={args.epochs}")


if __name__ == "__main__":
    main()
