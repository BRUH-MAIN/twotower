"""Dataset constants and the detector registry used by the CAD-SEL baselines."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(os.environ.get("CADSEL_DATA_ROOT", PROJECT_ROOT / "CAD-SEL-Dataset"))


def resolve_roots(root: Path | str | None = None) -> tuple[Path, Path]:
    """Locate the Images/ and Labels/ trees under ``root``.

    Handles the figshare layout (``<root>/Images``) and the Hugging Face mirror
    (``<root>/data/Images``), and tolerates being handed either the dataset root
    or the directory that contains it.
    """
    root = Path(root) if root is not None else DATASET_ROOT
    candidates = [root, root / "data", root / "CAD-SEL-Dataset", root / "CAD-SEL-Dataset" / "data"]
    for base in candidates:
        images, labels = base / "Images", base / "Labels"
        if images.is_dir() and labels.is_dir():
            return images, labels
    raise FileNotFoundError(
        f"could not find Images/ and Labels/ under {root} "
        f"(looked in: {', '.join(str(c) for c in candidates)})"
    )


IMAGES_ROOT = DATASET_ROOT / "Images"
LABELS_ROOT = DATASET_ROOT / "Labels"
METADATA_XLSX = DATASET_ROOT / "metadata.xlsx"

CENTERS = [f"Center-{i}" for i in range(1, 7)]
MODALITIES = ["WLE", "EUS"]
MODALITY_DIR = {"WLE": "WLE-Set", "EUS": "EUS-Set"}

# The figshare release nests by centre (`Center-1/WLE-Set/...`); the Hugging Face
# mirror at RohanRamesh/CAD-SEL collapses centres into `Full/NET-WL/...`. Both
# carry the same 4,912 images, so accept either spelling of a modality folder.
MODALITY_ALIASES = {
    "WLE": {"wle-set", "wle", "net-wl", "wl"},
    "EUS": {"eus-set", "eus", "net-eus"},
}

# Histological subtypes as they appear in the directory tree. NET_G1 and NET_G2
# are separate folders but the released bounding-box annotations collapse both
# into a single "NET" class id (0), so the detection label space has 4 entries.
SUBTYPE_DIRS = ["NET_G1", "NET_G2", "Leiomyoma", "Lipoma", "NonNeoplasm"]

# YOLO class id -> name, as encoded in Labels/**/*.txt
ANNOTATION_CLASSES = ["NET", "Leiomyoma", "Lipoma", "NonNeoplasm"]

# ---------------------------------------------------------------------------
# Label schemes (Technical Validation, two labelling schemes)
# ---------------------------------------------------------------------------
# (i)  multiclass: distinguish between all SEL categories
# (ii) binary:     NET versus non-NET (potentially malignant versus benign)
TASKS = ["multiclass", "binary"]

TASK_CLASSES = {
    "multiclass": ANNOTATION_CLASSES,
    "binary": ["NET", "non-NET"],
}

# annotation class id -> task class id
TASK_ID_MAP = {
    "multiclass": {0: 0, 1: 1, 2: 2, 3: 3},
    "binary": {0: 0, 1: 1, 2: 1, 3: 1},
}


def num_foreground_classes(task: str) -> int:
    return len(TASK_CLASSES[task])


def num_model_classes(task: str) -> int:
    """Foreground classes plus the Faster R-CNN background class."""
    return len(TASK_CLASSES[task]) + 1


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
# Dataset-specific statistics used by the reference implementation
# (github.com/xulouzhe/CAD-SEL). Kept for `--input-mode reference`.
CADSEL_MEAN = [0.5491333788465744, 0.3259111685958548, 0.2525661486929927]
CADSEL_STD = [0.24582985533315432, 0.19453490875333898, 0.15729866757757052]

# Per-modality statistics, measured over every image of each modality by
# `python -m cadsel.measure_stats` (2,817 WLE + 2,095 EUS).
#
# The two domains barely overlap. WLE spans 0.264 between its brightest and
# darkest channel mean; EUS spans 0.0004 (it is grayscale) and is ~2.9x darker
# overall. The CADSEL_* pair above is dominated by WLE — the brighter, more
# numerous half — so applying it to EUS lands its channels between -0.9 and -1.8
# (mean ~ -1.2) instead of zero, and squeezes its dynamic range by roughly half.
# That is harmless while a run only ever sees one modality, but not when a single
# backbone sees both, hence `make_normalizer(..., per_modality=True)` for joint
# runs. The unimodal baselines keep using CADSEL_* so their numbers stay
# comparable to the already-completed grid.
MODALITY_STATS = {
    "WLE": (
        [0.4886928080, 0.2899083121, 0.2243203920],
        [0.2953299486, 0.2162775730, 0.1761681077],
    ),
    "EUS": (
        [0.1160057515, 0.1160402813, 0.1163710243],
        [0.1360029622, 0.1360255508, 0.1366623216],
    ),
}

# Stable integer encoding, so a sample's modality can ride through the DataLoader
# collate/IPC boundary as a tensor.
MODALITY_INDEX = {name: i for i, name in enumerate(MODALITIES)}
MODALITY_NAMES = list(MODALITIES)

# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------
# Names follow Table 4/5 of the paper.
#
# The paper never states a training input resolution -- it reports only the
# native image sizes (362x290 to 1920x1080, median 1160x1008). The reference
# repository's `--img_size` defaults to 640 for every model, but it then feeds
# those tensors to the torchvision detector whose own GeneralizedRCNNTransform
# resizes again (min 800 for the ResNet variants, min 320 / max 640 for the
# "_320" model). `default_img_size` below is the resolution the network actually
# sees, so 320 for the "_320" variant matches its effective reference input and
# the design intent of that architecture; override with `--img-size`.
# `batch_size` is tuned for an 8 GB GPU.
MODEL_REGISTRY = {
    "fasterrcnn_resnet": {
        "paper_name": "Fasterrcnn_Resnet",
        "builder": "fasterrcnn_resnet50_fpn",
        "weights": "FasterRCNN_ResNet50_FPN_Weights",
        "default_img_size": 640,
        "batch_size": 4,
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "paper_name": "Fasterrcnn_Resnet50_FPN_V2",
        "builder": "fasterrcnn_resnet50_fpn_v2",
        "weights": "FasterRCNN_ResNet50_FPN_V2_Weights",
        "default_img_size": 640,
        "batch_size": 4,
    },
    "fasterrcnn_mobilenet": {
        "paper_name": "Fasterrcnn_Mobilenet",
        "builder": "fasterrcnn_mobilenet_v3_large_fpn",
        "weights": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
        "default_img_size": 640,
        "batch_size": 8,
    },
    "fasterrcnn_mobilenet_320": {
        "paper_name": "Fasterrcnn_Mobilenet_320",
        "builder": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "weights": "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
        "default_img_size": 320,
        "batch_size": 16,
    },
}

MODEL_NAMES = list(MODEL_REGISTRY)

# ---------------------------------------------------------------------------
# FCOS (anchor-free) -- the architecture-swap item from the Tier-3 literature
# review (FINDINGS.md §5c). Kept in a separate registry rather than folded
# into MODEL_REGISTRY: FCOS genuinely differs from the Faster R-CNN family --
# no RPN/ROI-head IoU matching (it uses center-sampling instead, so
# --rpn-fg-iou-thresh/--box-fg-iou-thresh/--auto-anchors don't apply), and no
# call through roi_heads.fastrcnn_loss (so --loss-fn/class-weighting don't
# apply either). Uses torchvision's own tested implementation rather than a
# from-scratch port of a specific paper variant (e.g. YOLO-OB's custom
# BiSPFPN head has no public reference implementation to build against) --
# the literature review's own fallback recommendation was "prioritize an
# anchor-free CNN detector (FCOS-style) over full DETR" if going this route.
FCOS_MODEL_REGISTRY = {
    "fcos_resnet50": {
        "paper_name": "FCOS_ResNet50_FPN",
        "default_img_size": 640,
        "batch_size": 4,
    },
}

# ---------------------------------------------------------------------------
# DINOv3 backbone (Lead 2, this session's counterfactual-protocol follow-up)
# -- Exp 17 (FINDINGS.md §5e) found DINOv3's embedding advantage over
# ResNet-50 partly survives lesion removal on WLE (not on EUS, where it's
# pure shortcut-reading), suggesting the pretrained representation captures
# some genuine lesion signal on WLE specifically. That was only ever tested
# as a frozen classifier probe; this wires the same frozen representation in
# as an actual detector backbone. Kept in its own registry rather than
# MODEL_REGISTRY: DINOv3 is a ViT (patch tokens, not a conv hierarchy), so it
# can't use MODEL_REGISTRY's torchvision-prebuilt-constructor pattern (no
# "fasterrcnn_dinov3_fpn" exists upstream) -- it needs a hand-built FasterRCNN
# from a custom single-scale backbone, the officially documented torchvision
# pattern for a non-FPN backbone (AnchorGenerator + MultiScaleRoIAlign over
# one feature map). Only "torchvision" --input-mode is supported: DINOv3 was
# pretrained with ImageNet normalisation (mean/std torchvision's own
# GeneralizedRCNNTransform already defaults to), and "reference" mode would
# feed it CADSEL_MEAN/STD-normalised pixels instead, silently degrading the
# pretrained representation.
DINOV3_MODEL_REGISTRY = {
    "fasterrcnn_dinov3": {
        "paper_name": "Fasterrcnn_DINOv3",
        "hf_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "default_img_size": 320,  # divisible by patch_size 16
        "batch_size": 8,
    },
}

# ---------------------------------------------------------------------------
# EfficientNet-B0 backbone (per-modality detector, following up on the joint
# two-tower classifier fusion work, FINDINGS.md §15) -- unlike DINOv3,
# EfficientNet-B0 IS a conv hierarchy (its `features` Sequential downsamples
# through several resolutions, verified empirically: stages 0/2/3/4/6 halve
# spatial resolution), so it gets a real multi-level FPN here rather than
# DINOv3's single-scale fallback -- a fairer, more comparable architecture to
# the existing ResNet50-FPN/MobileNetV3-FPN backbones than a single feature
# map would be. Still needs its own registry rather than MODEL_REGISTRY's
# stock-constructor pattern: torchvision ships no "fasterrcnn_efficientnet_b0
# _fpn" builder, so the backbone+FPN is hand-assembled (see
# cadsel/models.py::_efficientnet_b0_fpn_backbone) the same way
# torchvision's own `_mobilenet_extractor` assembles one for MobileNet --
# EfficientNet's blocks don't carry MobileNet's `_is_cn` stage-boundary
# marker, so the stage indices/channel counts below were determined by a
# direct forward-shape probe, not reused from that helper. ImageNet-pretrained,
# same "torchvision" --input-mode requirement as DINOv3 for the same reason
# (its own normalisation statistics, not CADSEL_MEAN/STD).
EFFICIENTNET_MODEL_REGISTRY = {
    "fasterrcnn_efficientnet_b0": {
        "paper_name": "Fasterrcnn_EfficientNetB0",
        "default_img_size": 640,
        "batch_size": 4,  # see FINDINGS.md: batch_size=6 OOM'd (fragmentation, not a
                           # one-shot too-big allocation) partway through a real 100-epoch
                           # run on an 8GB card -- lowered and re-validated past that point.
    },
}

ALL_MODEL_NAMES = (
    MODEL_NAMES + list(FCOS_MODEL_REGISTRY) + list(DINOV3_MODEL_REGISTRY)
    + list(EFFICIENTNET_MODEL_REGISTRY)
)

# ---------------------------------------------------------------------------
# Evaluation defaults (reference repository)
# ---------------------------------------------------------------------------
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5
