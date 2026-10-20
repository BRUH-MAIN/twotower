"""Experiment 45 -- DINOv3 tower parity across machines / library versions.

The cluster runs transformers 4.x on torch 2.3, the laptop transformers 5.x on
torch 2.11, and the DINOv3 model code moved between them. Before trusting a
cluster-trained DINOv3 two-tower next to the laptop ablation, this checks the
two installs compute the same function: one dataset frame (identical bytes on
both machines) through each *pretrained* DINOv3 tower, CPU float32, and the
CLS embedding compared against a committed reference.

    python -m experiments.src.45_two_tower_dinov3_parity --write   # laptop: write the reference
    python -m experiments.src.45_two_tower_dinov3_parity           # elsewhere: compare against it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.data import build_joint_manifest
from cadsel.joint_fusion_model import build_tower, tower_normalizers

REFERENCE = PROJECT_ROOT / "experiments" / "results" / "45_two_tower_dinov3_parity.json"
FRAME = ("EUS", "Leiomyoma/Leiomyoma-0", "1.tiff")
BACKBONES = ("dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16")
TOLERANCE = 1e-3            # max |difference| of any CLS component, float32 CPU


def frame_tensor() -> tuple[torch.Tensor, int]:
    from PIL import Image

    modality, key, name = FRAME
    sample = next(s for s in build_joint_manifest()
                  if s.modality == modality and s.patient_key == key and Path(s.image).name == name)
    with Image.open(sample.image) as im:
        arr = np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].float(), int(arr.astype(np.int64).sum())


@torch.no_grad()
def embeddings(backbones) -> dict:
    import transformers

    x, pixel_sum = frame_tensor()
    out = {"frame": "/".join(FRAME), "pixel_sum": pixel_sum, "torch": torch.__version__,
           "transformers": transformers.__version__, "cls": {}}
    for bb in backbones:
        tower = build_tower(bb, "partial", 4).eval()
        out["cls"][bb] = tower(tower_normalizers(bb)["EUS"](x))[0].tolist()
        print(f"  {bb}: |cls| = {np.linalg.norm(out['cls'][bb]):.5f}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="write the reference instead of comparing")
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    args = p.parse_args()
    torch.set_num_threads(max(torch.get_num_threads(), 1))

    got = embeddings(args.backbones)
    if args.write:
        REFERENCE.write_text(json.dumps(got, indent=1))
        print(f"wrote {REFERENCE}")
        return
    ref = json.loads(REFERENCE.read_text())
    assert got["pixel_sum"] == ref["pixel_sum"], "the frame's pixels differ -- not the same image"
    print(f"reference: torch {ref['torch']}, transformers {ref['transformers']}; "
          f"here: torch {got['torch']}, transformers {got['transformers']}")
    ok = True
    for bb, cls in got["cls"].items():
        diff = float(np.abs(np.array(cls) - np.array(ref["cls"][bb])).max())
        ok &= diff <= TOLERANCE
        print(f"  {bb}: max |cls - reference| = {diff:.2e}  {'ok' if diff <= TOLERANCE else 'MISMATCH'}")
    raise SystemExit(0 if ok else "DINOv3 towers differ between installs")


if __name__ == "__main__":
    main()
