"""A single dataclass that fully specifies a run, for both the classifier audit
(``experiments/src/12_*.py`` onward) and the detector pipeline (``cadsel/train.py``).

Deliberately plain dataclasses + JSON, no new dependency: matches the project's
existing anti-dependency stance (see ``experiments/probes.py``'s docstring) and
the registry style already used by ``MODEL_REGISTRY`` in ``config.py``. This is
not a framework — no plugin discovery, no dependency injection — just a
structured, serializable stand-in for the growing pile of CLI flags and ad hoc
notebook constants.

``cadsel/train.py`` and ``cadsel/run_all.py`` stay argparse-first; a ``RunConfig``
is an optional, additive way to specify a run (``--config``/``--config-grid``),
never a replacement for the existing flags.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """Which augmentation ops run, plus parameters for the newer ones.

    ``ops`` reuses ``cadsel.data.AUGMENT_OPS`` vocabulary once ``bbox_blur`` is
    added there (see the plan's Phase 2); until then it is meaningful only to
    the classifier experiments' own ``compose_fn`` hooks, which read the
    ``bbox_blur_*`` fields directly.
    """

    ops: tuple[str, ...] = ("flip", "color", "rotate")
    hard_negatives: float = 0.0

    # bbox_blur-only knobs (experiments/bbox_blur.py); ignored unless
    # "bbox_blur" is requested by whatever consumes this config.
    bbox_blur_p: float = 0.5
    bbox_blur_sigma_range: tuple[float, float] = (4.0, 14.0)
    bbox_blur_pad_frac_range: tuple[float, float] = (0.15, 0.5)
    bbox_blur_jitter_frac_range: tuple[float, float] = (0.1, 0.3)
    bbox_blur_feather_sigma_range: tuple[float, float] = (15.0, 30.0)

    def __post_init__(self) -> None:
        # JSON has no tuple type, so a round-trip through to_json/from_json
        # hands these back as lists; coerce so equality and typing stay honest
        # regardless of whether the caller passed tuples or lists.
        self.ops = tuple(self.ops)
        self.bbox_blur_sigma_range = tuple(self.bbox_blur_sigma_range)
        self.bbox_blur_pad_frac_range = tuple(self.bbox_blur_pad_frac_range)
        self.bbox_blur_jitter_frac_range = tuple(self.bbox_blur_jitter_frac_range)
        self.bbox_blur_feather_sigma_range = tuple(self.bbox_blur_feather_sigma_range)


# ---------------------------------------------------------------------------
# Shortcut-mitigation strategy
# ---------------------------------------------------------------------------
@dataclass
class MitigationConfig:
    """Which shortcut-mitigation method is active, and its knobs.

    ``strategy`` selects the primary mechanism; ``compose`` names additional
    strategies layered on top (currently only ``channel_attention`` composed
    with ``adversarial_discriminator`` is exercised, per Experiment 14 — see
    the plan's note that plain channel attention alone has no mechanism to
    suppress shortcut channels, only an accuracy objective).
    """

    strategy: Literal[
        "none", "adversarial_discriminator", "channel_attention", "bbox_blur"
    ] = "none"
    compose: tuple[str, ...] = ()

    # adversarial_discriminator (experiments/adversarial.py)
    discriminator_targets: tuple[str, ...] = ("bg_patch",)
    grl_lambda_max: float = 1.0
    grl_warmup_epochs: int = 2
    discriminator_hidden: int = 128
    discriminator_weight: float = 1.0

    # channel_attention (experiments/backbones.py)
    se_stages: Literal["layer4", "all"] = "all"
    se_reduction: int = 16

    def __post_init__(self) -> None:
        self.compose = tuple(self.compose)
        self.discriminator_targets = tuple(self.discriminator_targets)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """Fully specifies one run.

    ``task`` deliberately keeps two different vocabularies rather than
    unifying them: the detector uses ``cadsel.config.TASKS`` ("multiclass" /
    "binary", a box-level 4-class label space), the classifier harness uses
    ``experiments.harness.classes_for`` ("binary" / "subtype", an image-level
    5-class label space). Forcing one vocabulary would risk silently breaking
    comparability with the already-published Exp 3/7/11 tables, so callers are
    expected to know which world their config targets.

    ``backbone`` is either a ``cadsel.config.MODEL_REGISTRY`` key (detector) or
    a classifier backbone name understood by ``experiments/backbones.py``
    (currently just ``"resnet18"``).
    """

    name: str = "run"
    seed: int = 42
    modality: str = "WLE"
    task: str = "multiclass"
    split_level: Literal["patient", "image"] = "patient"
    val_fraction: float = 0.2

    backbone: str = "fasterrcnn_mobilenet_320"
    img_size: int | None = None
    pretrained: bool = True

    # Tier-1 performance levers from the literature review (see FINDINGS.md /
    # experiments/README.md) -- all additive/opt-in, default off.
    auto_anchors: bool = False
    rpn_fg_iou_thresh: float | None = None
    rpn_bg_iou_thresh: float | None = None
    box_fg_iou_thresh: float | None = None
    box_bg_iou_thresh: float | None = None
    multiscale_sizes: tuple[int, ...] = ()
    ema: bool = False
    ema_decay: float = 0.999
    rfs_threshold: float = 0.0
    size_rfs_threshold: float = 0.0
    center_rfs_threshold: float = 0.0
    loss_fn: Literal["weighted_ce", "seesaw", "group_dro"] = "weighted_ce"
    seesaw_p: float = 0.8
    groupdro_step_size: float = 0.01
    backbone_weights: str | None = None

    # Leave-one-center-out support (Lead 3b): empty = unfiltered, matching
    # cadsel.data.build_manifest's own `centers=None` default.
    centers: tuple[str, ...] = ()

    # Lesion-GAN copy-paste (Lead 5a): swaps LesionBank's real crops for
    # cadsel.lesion_gan.GeneratedLesionBank's synthetic ones when "copy_paste"
    # is in augment.ops -- see cadsel/train.py's --copy-paste-source.
    copy_paste_source: Literal["real", "generated"] = "real"
    lesion_gan_checkpoint: str | None = None

    augment: AugmentConfig = field(default_factory=AugmentConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)

    epochs: int = 100
    batch_size: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    early_stop_patience: int = 0
    class_weights: bool = True
    input_mode: Literal["reference", "torchvision"] = "reference"

    num_workers: int = 4
    device: str | None = None
    output_dir: str = "runs"

    def __post_init__(self) -> None:
        self.multiscale_sizes = tuple(self.multiscale_sizes)
        self.centers = tuple(self.centers)

    # -- (de)serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        data = dict(data)
        aug = data.pop("augment", None)
        mit = data.pop("mitigation", None)
        cfg = cls(**data)
        if aug is not None:
            cfg.augment = AugmentConfig(**aug)
        if mit is not None:
            cfg.mitigation = MitigationConfig(**mit)
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def override(self, **kw: Any) -> "RunConfig":
        return replace(self, **kw)


# A small registry, not a framework -- mirrors cadsel.config.MODEL_REGISTRY's
# style. Extend inline as new presets prove useful; do not build a discovery
# mechanism for a handful of named configs.
PRESETS: dict[str, RunConfig] = {
    "detector_baseline": RunConfig(),
    "detector_hardneg30": RunConfig(
        name="detector_hardneg30", augment=AugmentConfig(hard_negatives=0.3)
    ),
}


# ---------------------------------------------------------------------------
# argparse interop (cadsel/train.py --config, cadsel/run_all.py --config-grid)
# ---------------------------------------------------------------------------
# Maps RunConfig field name -> argparse dest, only where the two names differ
# in more than hyphens-vs-underscores.
_ARGS_ALIASES = {"backbone": "model", "name": "run_name"}

# RunConfig fields whose argparse counterpart is a same-meaning-but-inverted
# `--no-*` store_true flag (default False = "the config's True is active").
_INVERTED_BOOL_ALIASES = {"pretrained": "no_pretrained", "class_weights": "no_class_weights"}


def apply_to_namespace(config: RunConfig, args: Any, explicit: set[str]) -> Any:
    """Fill in ``args`` fields the caller didn't pass on the CLI.

    ``explicit`` is the set of dest names the user actually typed a flag for
    (see ``cadsel/train.py``'s ``--config`` handling); those always win over
    the config, so ``--config foo.json --epochs 5`` behaves the way a reader
    expects. Everything else is overwritten from ``config`` when a same-named
    (or aliased) attribute exists on ``args``.
    """
    flat = config.to_dict()
    aug = flat.pop("augment")
    mit = flat.pop("mitigation")
    flat["augment"] = ",".join(aug["ops"])
    flat["hard_negatives"] = aug["hard_negatives"]
    flat["multiscale_sizes"] = ",".join(str(s) for s in flat["multiscale_sizes"]) or None
    flat["centers"] = ",".join(flat["centers"]) or None

    for src, dest in _INVERTED_BOOL_ALIASES.items():
        value = flat.pop(src)
        if dest not in explicit and hasattr(args, dest):
            setattr(args, dest, not value)

    for key, value in flat.items():
        dest = _ARGS_ALIASES.get(key, key)
        if dest in explicit:
            continue
        if hasattr(args, dest):
            setattr(args, dest, value)

    # Mitigation config isn't consumed by cadsel/train.py yet (Phase 2); stash
    # it so a future consumer can find it without re-parsing the file.
    args._mitigation = mit
    return args
