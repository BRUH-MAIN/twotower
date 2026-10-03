"""Convert Lab-Rasool/RadImageNet's ResNet50.pt to torchvision's key convention.

That checkpoint (https://huggingface.co/Lab-Rasool/RadImageNet, MIT-licensed,
verified in this session to load safely with ``weights_only=True`` -- a plain
tensor state_dict, not a pickled object) wraps a standard torchvision
``resnet50()`` as ``nn.Sequential(*list(resnet.children())[:-2])`` -- so its
keys are ``backbone.<index>...`` (0=conv1, 1=bn1, 2=relu, 3=maxpool,
4=layer1, 5=layer2, 6=layer3, 7=layer4) instead of the named-submodule
convention (``conv1.weight``, ``layer1.0.conv1.weight``, ...) that
``cadsel.models.load_backbone_weights`` expects (it loads into
``model.backbone.body``, an ``IntermediateLayerGetter`` that preserves the
original ResNet attribute names). This remaps the keys once and saves a
plain state_dict directly usable via ``--backbone-weights``.

    python -m cadsel.convert_radimagenet_weights --output runs/radimagenet_resnet50.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import PROJECT_ROOT

_INDEX_TO_NAME = {0: "conv1", 1: "bn1", 4: "layer1", 5: "layer2", 6: "layer3", 7: "layer4"}


def convert(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    skipped = []
    for key, value in state_dict.items():
        if not key.startswith("backbone."):
            skipped.append(key)
            continue
        rest = key[len("backbone.") :]
        idx_str, _, tail = rest.partition(".")
        try:
            idx = int(idx_str)
        except ValueError:
            skipped.append(key)
            continue
        if idx not in _INDEX_TO_NAME:
            # relu (2) / maxpool (3) have no parameters; anything else is unexpected.
            skipped.append(key)
            continue
        name = _INDEX_TO_NAME[idx]
        out[f"{name}.{tail}" if tail else name] = value
    if skipped:
        print(f"  skipped {len(skipped)} key(s) with no torchvision-backbone equivalent: "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "radimagenet_resnet50.pth")
    p.add_argument("--repo-id", default="Lab-Rasool/RadImageNet")
    p.add_argument("--filename", default="ResNet50.pt")
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    print(f"downloading {args.repo_id}/{args.filename} ...")
    path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)

    # weights_only=True: this file was verified in this session to be a plain
    # tensor state_dict; keep the safe loader rather than assuming that stays
    # true if the upstream file ever changes.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    converted = convert(raw)
    print(f"  converted {len(converted)}/{len(raw)} tensors to torchvision ResNet50 key names")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(f"wrote {args.output}")

    # Sanity check: load it into a real torchvision resnet50 backbone body.
    from torchvision.models import resnet50

    probe = resnet50(weights=None)
    body_keys = {
        k for k in probe.state_dict()
        if k.split(".")[0] in _INDEX_TO_NAME.values()
    }
    matched = body_keys & set(converted)
    print(f"  sanity check: {len(matched)}/{len(body_keys)} torchvision ResNet50 "
          f"conv/bn keys matched")
    if len(matched) != len(body_keys):
        print("  WARNING: incomplete match -- inspect before trusting this checkpoint")


if __name__ == "__main__":
    main()
