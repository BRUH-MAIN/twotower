"""The four Faster R-CNN detectors of Table 4/5, with class-weighted losses.

The paper states: "All models were optimized with classification loss and
bounding-box regression loss. Class weight balancing was further implemented by
assigning weights inversely proportional to the frequency of each category to
address dataset imbalance."

torchvision computes the ROI-head losses in the module-level function
``torchvision.models.detection.roi_heads.fastrcnn_loss``, which ``RoIHeads.forward``
resolves at call time. Installing a weight-aware replacement there is therefore
enough to weight the classification term, and keeps the box-regression term
byte-for-byte identical to upstream. The active weight vector is process-global,
which is fine because a run trains exactly one model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import roi_heads as _roi_heads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead

from .config import FCOS_MODEL_REGISTRY, MODALITY_INDEX, MODEL_REGISTRY, num_model_classes

_ORIGINAL_FASTRCNN_LOSS = _roi_heads.fastrcnn_loss
_ACTIVE_CLASS_WEIGHTS: dict[str, torch.Tensor | None] = {"weights": None}


def _weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    weights = _ACTIVE_CLASS_WEIGHTS["weights"]
    if weights is None:
        return _ORIGINAL_FASTRCNN_LOSS(class_logits, box_regression, labels, regression_targets)

    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(
        class_logits, labels_cat, weight=weights.to(class_logits.device, class_logits.dtype)
    )

    # Identical to torchvision's implementation.
    sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
    labels_pos = labels_cat[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets_cat[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels_cat.numel()

    return classification_loss, box_loss


def install_weighted_loss() -> None:
    """Patch torchvision's ROI-head loss so class weights take effect."""
    _roi_heads.fastrcnn_loss = _weighted_fastrcnn_loss


def set_class_weights(weights: torch.Tensor | None) -> None:
    """Set the foreground+background weight vector (length ``num_classes``)."""
    _ACTIVE_CLASS_WEIGHTS["weights"] = weights


def build_class_weight_tensor(foreground_weights, background_weight: float = 1.0) -> torch.Tensor:
    """Prepend the background weight to the per-category foreground weights."""
    values = [background_weight] + [float(w) for w in foreground_weights]
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Seesaw Loss (Wang et al., CVPR 2021) -- an alternative to the static
# class-weighted CE above, from the literature review backing the Tier-1
# performance levers (FINDINGS.md §5c).
#
# `compute_class_weights`' inverse-frequency weights are a one-time snapshot:
# they penalise confusing two *rare* classes (e.g. Leiomyoma vs NonNeoplasm,
# both scarce) exactly as harshly as a rare-vs-frequent confusion (Leiomyoma
# vs NET), which is the wrong trade-off -- the paper's own motivating
# observation. Seesaw Loss instead tracks *cumulative* per-class sample
# counts as training progresses and, for each ground-truth class i, damps how
# much a more-frequent class j's logit gets to suppress i's probability,
# scaled by their *current* count ratio -- so the correction adapts as the
# effective class balance seen so far shifts (e.g. under --rfs-threshold
# oversampling), rather than staying fixed at a pre-training snapshot.
# ---------------------------------------------------------------------------
class SeesawLossState:
    """Cumulative per-class sample counts and the resulting mitigation matrix.

    Background (class 0) is excluded from the seesaw mechanism entirely (never
    suppressed, never suppresses another class) -- it isn't part of the
    long-tail imbalance this is meant to address, matching the paper's own
    treatment of a "not-an-instance" background class.
    """

    def __init__(self, n_classes: int, p: float = 0.8, device: torch.device | None = None):
        self.n_classes = n_classes
        self.p = p
        # Start at 1 (not 0) so the very first batch's ratio is well-defined.
        self.counts = torch.ones(n_classes, dtype=torch.float64, device=device)

    def update(self, labels: torch.Tensor) -> None:
        counts = torch.bincount(labels, minlength=self.n_classes).to(self.counts.device)
        self.counts += counts.double()

    def mitigation_matrix(self) -> torch.Tensor:
        """``(n_classes, n_classes)`` matrix ``M[i, j]``: the multiplicative
        damping applied to class ``j``'s logit when the ground truth is ``i``."""
        n_j = self.counts.view(1, -1)
        n_i = self.counts.view(-1, 1)
        ratio = n_j / n_i
        m = torch.where(ratio > 1, ratio.pow(-self.p), torch.ones_like(ratio))
        m.fill_diagonal_(1.0)
        m[0, :] = 1.0
        m[:, 0] = 1.0
        return m


_ACTIVE_SEESAW_STATE: dict[str, SeesawLossState | None] = {"state": None}


def set_seesaw_state(state: SeesawLossState | None) -> None:
    _ACTIVE_SEESAW_STATE["state"] = state


def _seesaw_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    state = _ACTIVE_SEESAW_STATE["state"]
    if state is None:
        return _ORIGINAL_FASTRCNN_LOSS(class_logits, box_regression, labels, regression_targets)

    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)

    state.update(labels_cat.detach())
    m = state.mitigation_matrix().to(class_logits.device, class_logits.dtype)
    # Row per sample, selected by its own ground-truth class: log(M[gt, :])
    # added to the logits shifts the softmax normaliser exactly like a
    # per-sample class-dependent bias, damping classes M says are unfairly
    # suppressive for this sample's true class.
    class_logits = class_logits + torch.log(m[labels_cat].clamp_min(1e-12))
    classification_loss = F.cross_entropy(class_logits, labels_cat)

    # Box loss identical to torchvision's implementation -- Seesaw Loss only
    # concerns the classification term.
    sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
    labels_pos = labels_cat[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets_cat[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels_cat.numel()

    return classification_loss, box_loss


_ACTIVE_GROUP: dict[str, str | None] = {"group": None}


def set_active_group(group: str | None) -> None:
    """Declare which group (acquisition center) the next forward pass belongs
    to -- read by ``GroupDROLossState`` via ``_groupdro_fastrcnn_loss``.
    Requires a group-homogeneous batch (``data.CenterBatchSampler``), same
    contract as ``set_active_modality``."""
    _ACTIVE_GROUP["group"] = group


def get_active_group() -> str | None:
    return _ACTIVE_GROUP["group"]


class GroupDROLossState:
    """Group DRO (Sagawa et al., ICLR 2020): minimise *worst-group* risk
    instead of average risk, via an online exponentiated-gradient-ascent
    reweighting over groups (here: acquisition center).

    Where Seesaw (above) reweights *classes* within a single loss
    computation, Group DRO reweights the *whole loss* per batch by how
    adversarially "hard" its group currently looks -- the two are
    orthogonal mechanisms and this project's own cheaper group-aware probes
    (``CenterRepeatFactorSampler``, DFR's ``build_balanced_subset_by_center``)
    approximate what this does more directly, by resampling which examples
    the optimiser sees rather than reweighting the loss itself.

    Maintains an adversarial weight ``q_g`` per group, normalised to sum 1.
    After each group-homogeneous batch, ``update`` multiplies that group's
    weight by ``exp(step_size * batch_loss)`` (an under-performing group's
    loss stays high longer, so its weight grows) and renormalises, then
    returns a scale factor (``q_g * n_groups``, so the *average* weight
    across groups stays ~1 and the learning rate doesn't need retuning) to
    multiply the batch's loss by before backprop.
    """

    def __init__(self, groups: Sequence[str], step_size: float = 0.01, device: torch.device | None = None):
        self.groups = list(groups)
        self.index = {g: i for i, g in enumerate(self.groups)}
        self.step_size = step_size
        self.q = torch.ones(len(self.groups), dtype=torch.float64, device=device) / len(self.groups)

    def update(self, group: str, loss: torch.Tensor) -> torch.Tensor:
        i = self.index[group]
        with torch.no_grad():
            self.q[i] = self.q[i] * torch.exp(self.step_size * loss.detach().double())
            self.q = self.q / self.q.sum()
        return (self.q[i] * len(self.groups)).to(loss.dtype)


_ACTIVE_GROUPDRO_STATE: dict[str, GroupDROLossState | None] = {"state": None}


def set_groupdro_state(state: GroupDROLossState | None) -> None:
    _ACTIVE_GROUPDRO_STATE["state"] = state


def _groupdro_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    state = _ACTIVE_GROUPDRO_STATE["state"]
    if state is None:
        return _weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

    group = get_active_group()
    if group is None:
        raise RuntimeError(
            "GroupDROLossState is active but no group is set -- call "
            "set_active_group(...) before the forward pass (train_one_epoch's "
            "group_dro_aware=True does this automatically for a batch built "
            "with data.CenterBatchSampler)"
        )
    # Reuses the existing class-weighted CE as the per-example loss (Group
    # DRO's group reweighting and weighted_ce's class reweighting are
    # orthogonal and layer cleanly, same as Seesaw could in principle).
    classification_loss, box_loss = _weighted_fastrcnn_loss(
        class_logits, box_regression, labels, regression_targets
    )
    weight = state.update(group, (classification_loss + box_loss).detach())
    return classification_loss * weight, box_loss * weight


LOSS_FUNCTIONS = ("weighted_ce", "seesaw", "group_dro")


def install_loss_fn(name: str) -> None:
    """Select which ROI-head classification loss is active.

    ``"weighted_ce"`` (default) is ``install_weighted_loss``'s existing static
    class-weighted cross-entropy (a no-op falling back to plain CE when
    ``set_class_weights(None)``). ``"seesaw"`` installs the cumulative,
    training-progress-aware alternative above. ``"group_dro"`` installs the
    per-group adversarial reweighting above (requires ``set_groupdro_state``
    and a ``data.CenterBatchSampler``-built loader). The three are mutually
    exclusive (selecting one does not layer onto the others).
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"unknown loss_fn {name!r}; expected one of {LOSS_FUNCTIONS}")
    _roi_heads.fastrcnn_loss = {
        "seesaw": _seesaw_fastrcnn_loss,
        "group_dro": _groupdro_fastrcnn_loss,
    }.get(name, _weighted_fastrcnn_loss)


def _resample(values: list[float], n: int) -> list[float]:
    """Evenly resample sorted ``values`` to exactly ``n`` entries by index
    interpolation. Used when a clustered anchor-size count doesn't match what
    a given backbone's anchor generator structurally wants per FPN level."""
    values = sorted(values)
    if len(values) == n:
        return values
    idx = np.linspace(0, len(values) - 1, n)
    return [values[int(round(i))] for i in idx]


def _reshape_anchor_config(
    default_ag: AnchorGenerator, sizes: Sequence[float], aspect_ratios: Sequence[float] | None
) -> AnchorGenerator:
    """Data-driven anchors reshaped to match ``default_ag``'s existing structure
    (e.g. ResNet-FPN: 5 levels x 1 size each; MobileNetV3-FPN: 3 levels x 5 sizes
    each, all levels sharing the same size list) -- so the RPNHead the caller
    rebuilds around this ends up with the right number of anchors per location.
    ``sizes``/``aspect_ratios`` normally come from ``cadsel.anchors.compute_anchor_config``.
    """
    n_levels = len(default_ag.sizes)
    sizes_per_level = len(default_ag.sizes[0])

    if sizes_per_level == 1:
        chosen = _resample(list(sizes), n_levels)
        ag_sizes = tuple((int(s),) for s in chosen)
    else:
        chosen = _resample(list(sizes), sizes_per_level)
        ag_sizes = (tuple(int(s) for s in chosen),) * n_levels

    ratios = tuple(aspect_ratios) if aspect_ratios else default_ag.aspect_ratios[0]
    ag_ratios = (ratios,) * n_levels
    return AnchorGenerator(sizes=ag_sizes, aspect_ratios=ag_ratios)


def _install_anchor_generator(model, sizes: Sequence[float], aspect_ratios: Sequence[float] | None) -> None:
    """Replace ``model``'s RPN anchor generator in place, rebuilding the RPN head
    to match the new anchor-per-location count.

    Done post-construction rather than passed into the builder because two of
    the four registry builders (the MobileNetV3-FPN family) hardcode their own
    ``rpn_anchor_generator=`` inside the torchvision helper, so it can't be
    threaded through as a kwarg the way ``fasterrcnn_resnet50_fpn`` allows --
    this approach works identically for all four.
    """
    new_ag = _reshape_anchor_config(model.rpn.anchor_generator, sizes, aspect_ratios)
    model.rpn.anchor_generator = new_ag
    model.rpn.head = RPNHead(model.backbone.out_channels, new_ag.num_anchors_per_location()[0])


def build_detection_model(
    model_type: str,
    task: str,
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    anchor_sizes: Sequence[float] | None = None,
    anchor_aspect_ratios: Sequence[float] | None = None,
    rpn_fg_iou_thresh: float | None = None,
    rpn_bg_iou_thresh: float | None = None,
    box_fg_iou_thresh: float | None = None,
    box_bg_iou_thresh: float | None = None,
):
    """Instantiate one of the four Table 4 detectors with a fresh box predictor.

    ``anchor_sizes``/``anchor_aspect_ratios`` (typically from
    ``cadsel.anchors.compute_anchor_config``) replace torchvision's COCO-tuned
    RPN anchors with ones matched to CAD-SEL's own box-size distribution --
    IoU-based anchor matching mechanically starves ground-truth boxes with no
    close anchor size, which is a plausible contributor to the confirmed
    small-lesion AP gap. ``rpn_*_iou_thresh``/``box_*_iou_thresh`` lower the
    default 0.7/0.5 fg thresholds, for the same reason: small boxes rarely
    reach high IoU with any anchor purely from discretisation.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"unknown model_type {model_type!r}; expected one of {list(MODEL_REGISTRY)}")

    spec = MODEL_REGISTRY[model_type]
    builder = getattr(torchvision.models.detection, spec["builder"])
    weights_enum = getattr(torchvision.models.detection, spec["weights"])

    kwargs = {"weights": weights_enum.DEFAULT if pretrained else None}
    if trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers
    if rpn_fg_iou_thresh is not None:
        kwargs["rpn_fg_iou_thresh"] = rpn_fg_iou_thresh
    if rpn_bg_iou_thresh is not None:
        kwargs["rpn_bg_iou_thresh"] = rpn_bg_iou_thresh
    if box_fg_iou_thresh is not None:
        kwargs["box_fg_iou_thresh"] = box_fg_iou_thresh
    if box_bg_iou_thresh is not None:
        kwargs["box_bg_iou_thresh"] = box_bg_iou_thresh

    model = builder(**kwargs)
    if anchor_sizes is not None:
        _install_anchor_generator(model, anchor_sizes, anchor_aspect_ratios)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_model_classes(task))
    return model


def build_fcos_model(
    model_type: str, task: str, pretrained: bool = True, trainable_backbone_layers: int | None = None,
):
    """FCOS (Tian et al., 2019) -- an anchor-free one-stage detector.

    Unlike ``build_detection_model``'s family, ``num_classes`` is set directly
    at construction (torchvision's ``fcos_resnet50_fpn`` builder takes it as a
    kwarg) rather than via a post-hoc box-predictor swap, since FCOS's
    classification head has no COCO-pretrained variant compatible with a
    different class count to swap out of. ``weights_backbone`` still gives the
    ImageNet-pretrained ResNet50 backbone under ``pretrained=True`` -- the
    part of "pretrained" that matters most on a dataset this size.
    """
    if model_type not in FCOS_MODEL_REGISTRY:
        raise ValueError(f"unknown FCOS model_type {model_type!r}; expected one of {list(FCOS_MODEL_REGISTRY)}")

    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import fcos_resnet50_fpn

    kwargs = {}
    if trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers
    return fcos_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT if pretrained else None,
        num_classes=num_model_classes(task),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# DINOv3 backbone (Lead 2 follow-up, FINDINGS.md §5e/§5o) -- see
# cadsel/config.py::DINOV3_MODEL_REGISTRY's docstring for why this needs its
# own builder rather than MODEL_REGISTRY's torchvision-prebuilt pattern.
# ---------------------------------------------------------------------------
def _load_hf_token() -> str | None:
    """``HF_TOKEN`` from the environment, falling back to a ``.env`` file --
    duplicated from ``experiments/embeddings.py`` rather than imported, to
    keep the production pipeline decoupled from the audit tree (same
    convention as ``box_area_frac`` and other small helpers in this repo)."""
    import os

    from .config import PROJECT_ROOT

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip('"').strip("'")
    return None


class Dinov3Backbone(nn.Module):
    """Wraps a frozen (by default) DINOv3 ViT as a single-scale torchvision
    detection backbone: ``forward`` returns one ``(B, out_channels, H', W')``
    feature map (the patch-token grid), the interface
    ``torchvision.models.detection.FasterRCNN`` expects from a non-FPN
    backbone (its own documented custom-backbone pattern).

    DINOv3's ``last_hidden_state`` prepends 1 CLS token + ``num_register_tokens``
    register tokens before the patch tokens (verified: 405 = 1 + 4 + 400 for a
    320x320/patch16 input) -- the patch tokens are always the trailing
    ``H'*W'`` entries regardless of how many special tokens precede them.
    """

    def __init__(self, model_name: str, trainable: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, token=_load_hf_token())
        self.patch_size = self.model.config.patch_size
        self.num_special_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)
        self.out_channels = self.model.config.hidden_size
        self.trainable = trainable
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> "Dinov3Backbone":
        super().train(mode)
        if not self.trainable:
            self.model.eval()  # frozen backbone stays in eval mode (no dropout) regardless
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = x.shape
        tokens = self.model(pixel_values=x).last_hidden_state
        patch_tokens = tokens[:, self.num_special_tokens:, :]
        h_p, w_p = H // self.patch_size, W // self.patch_size
        return patch_tokens.transpose(1, 2).reshape(B, self.out_channels, h_p, w_p)


def build_dinov3_detection_model(model_type: str, task: str, trainable_backbone: bool = False):
    """A ``FasterRCNN`` over a frozen (by default) DINOv3 backbone, single-scale
    (no FPN -- DINOv3's patch grid is one resolution, not a conv hierarchy).
    ``num_classes`` is set directly at construction, same as ``build_fcos_model``,
    since there's no COCO-pretrained head to swap out of for a from-scratch RPN.
    """
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    from .config import DINOV3_MODEL_REGISTRY

    if model_type not in DINOV3_MODEL_REGISTRY:
        raise ValueError(f"unknown DINOv3 model_type {model_type!r}; expected one of {list(DINOV3_MODEL_REGISTRY)}")
    hf_model_name = DINOV3_MODEL_REGISTRY[model_type]["hf_model_name"]

    backbone = Dinov3Backbone(hf_model_name, trainable=trainable_backbone)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        backbone, num_classes=num_model_classes(task),
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
    )


# ---------------------------------------------------------------------------
# EfficientNet-B0 backbone (per-modality detector follow-up to the joint
# two-tower classifier fusion work, FINDINGS.md §15) -- see
# cadsel/config.py::EFFICIENTNET_MODEL_REGISTRY's docstring for why this
# needs its own hand-assembled backbone+FPN rather than a stock torchvision
# constructor.
# ---------------------------------------------------------------------------
def _efficientnet_b0_fpn_backbone(pretrained: bool = True, trainable_layers: int = 6) -> nn.Module:
    """``BackboneWithFPN`` over EfficientNet-B0's ``features``, mirroring
    ``torchvision.models.detection.backbone_utils._mobilenet_extractor``'s
    approach for a non-ResNet conv backbone -- but built by hand rather than
    calling that helper, since EfficientNet's blocks don't carry MobileNet's
    ``_is_cn`` stage-boundary marker (verified: ``hasattr(block, "_is_cn")``
    is ``False`` for every stage). Stage boundaries and channel counts below
    come from a direct forward-shape probe of ``efficientnet_b0().features``
    at a 224x224 input:

        stage  0  1  2  3  4  5  6  7  8
        stride /2 /2 /4 /8 /16 /16 /32 /32 /32
        chans  32 16 24 40 80 112 192 320 1280

    Four levels are picked, one per resolution (the deepest/most-refined
    stage at each), the same "skip the stem, one representative per scale"
    convention ``resnet_fpn_backbone`` uses for C2-C5: stage 2 (/4, 24ch),
    stage 3 (/8, 40ch), stage 5 (/16, 112ch -- deeper than stage 4's 80ch at
    the same resolution), stage 8 (/32, 1280ch -- the final head conv,
    deeper than stages 6/7 at the same resolution). ``LastLevelMaxPool``
    adds a 5th (P6) level for the RPN, matching every other FPN backbone in
    this project.

    ``trainable_layers`` follows ``_mobilenet_extractor``'s own convention:
    0 freezes the whole backbone, N unfreezes the last N of the 6
    stage-boundary segments ``[0, 2, 3, 4, 6, 8]`` (stem + the 4 chosen
    stages + the final block) -- default 6 leaves everything trainable,
    matching how this project's other detector backbones are fully
    fine-tuned rather than partially frozen.

    No ``FrozenBatchNorm2d`` conversion (unlike torchvision's own COCO-
    pretrained ResNet/MobileNet backbones) -- EfficientNet-B0's BatchNorm2d
    layers are left as ordinary trainable BN, a deliberate simplification
    given this project's already-small detection batch sizes rather than an
    oversight.
    """
    import torchvision.models as tvm
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

    backbone = tvm.efficientnet_b0(
        weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    ).features

    stage_indices = [0, 2, 3, 4, 6, 8]
    num_stages = len(stage_indices)
    if not 0 <= trainable_layers <= num_stages:
        raise ValueError(f"trainable_layers must be in [0,{num_stages}], got {trainable_layers}")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]
    for block in backbone[:freeze_before]:
        for p in block.parameters():
            p.requires_grad_(False)

    chosen = [2, 3, 5, 8]
    return_layers = {str(i): str(v) for v, i in enumerate(chosen)}
    # Read the real output channel count off a live forward pass rather than
    # introspecting submodule attributes: `features[8]` (Conv2dNormActivation)
    # is *also* an `nn.Sequential` like the MBConv stages, but its last child
    # is a bare `SiLU` activation with no `out_channels` -- a structural
    # attribute lookup gives the wrong answer there silently, a probe doesn't.
    with torch.no_grad():
        probe = torch.zeros(1, 3, 224, 224)
        in_channels_list = []
        for i, block in enumerate(backbone):
            probe = block(probe)
            if i in chosen:
                in_channels_list.append(probe.shape[1])
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )


def build_efficientnet_detection_model(model_type: str, task: str, pretrained: bool = True,
                                        trainable_layers: int = 6):
    """A ``FasterRCNN`` over an EfficientNet-B0+FPN backbone. ``num_classes``
    is set directly at construction (same as ``build_dinov3_detection_model``/
    ``build_fcos_model``) since there's no COCO-pretrained detection head to
    swap out of for a from-scratch backbone+FPN+RPN.
    """
    from torchvision.models.detection import FasterRCNN
    from torchvision.ops import MultiScaleRoIAlign

    from .config import EFFICIENTNET_MODEL_REGISTRY

    if model_type not in EFFICIENTNET_MODEL_REGISTRY:
        raise ValueError(
            f"unknown EfficientNet model_type {model_type!r}; "
            f"expected one of {list(EFFICIENTNET_MODEL_REGISTRY)}"
        )

    backbone = _efficientnet_b0_fpn_backbone(pretrained=pretrained, trainable_layers=trainable_layers)
    # One anchor size per FPN level (4 chosen scales + LastLevelMaxPool's P6),
    # the same per-level convention torchvision's own `_default_anchorgen`
    # uses for resnet_fpn_backbone.
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        backbone, num_classes=num_model_classes(task),
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
    )


# ---------------------------------------------------------------------------
# Modality conditioning (joint WLE+EUS runs)
# ---------------------------------------------------------------------------
# The active modality is process-global for the same reason the class weights
# above are: torchvision's detector `forward(images, targets)` has nowhere to put
# a per-batch side input, and a run trains exactly one model. `ModalityBatchSampler`
# guarantees batches are modality-homogeneous, so "the active modality" is always
# well defined during a forward pass.
_ACTIVE_MODALITY: dict[str, str | None] = {"modality": None}

CONDITIONING_MODES = ("none", "film", "film+bn")


def set_active_modality(modality: str | None) -> None:
    """Declare which modality the next forward pass will see."""
    if modality is not None and modality not in MODALITY_INDEX:
        raise ValueError(f"unknown modality {modality!r}")
    _ACTIVE_MODALITY["modality"] = modality


def get_active_modality() -> str | None:
    return _ACTIVE_MODALITY["modality"]


class ModalityFiLM(nn.Module):
    """Per-modality affine modulation of one feature map.

    A learned scale and shift per channel, selected by modality. Initialised to
    the identity (scale 1, shift 0) so an untrained conditioner leaves the
    pretrained backbone's features untouched — the run starts from the modality-
    blind model and has to earn any deviation.

    FiLM rather than separate backbones because there are only 421 EUS patients:
    the whole point of joint training is to share representation capacity, and two
    towers would share none of it.
    """

    def __init__(self, channels: int, n_modalities: int = len(MODALITY_INDEX)) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_modalities, channels))
        self.beta = nn.Parameter(torch.zeros(n_modalities, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        modality = get_active_modality()
        if modality is None:
            return x
        i = MODALITY_INDEX[modality]
        gamma = self.gamma[i].view(1, -1, 1, 1)
        beta = self.beta[i].view(1, -1, 1, 1)
        return x * gamma + beta


class ModalityBatchNorm2d(nn.Module):
    """Wraps a frozen//trainable norm layer with per-modality running statistics.

    Domain-specific normalisation: the layer's learnable affine parameters stay
    shared (so the representation stays joint) while each modality keeps its own
    running mean/var. With WLE at a 0.27 channel spread and EUS at 2e-4, one set
    of running statistics has to straddle two distributions that barely overlap.
    """

    def __init__(self, source: nn.Module, n_modalities: int = len(MODALITY_INDEX)) -> None:
        super().__init__()
        num_features = source.num_features
        self.weight = source.weight
        self.bias = source.bias
        self.eps = source.eps
        self.momentum = source.momentum if source.momentum is not None else 0.1
        self.branches = nn.ModuleList()
        for _ in range(n_modalities):
            bn = nn.BatchNorm2d(num_features, eps=self.eps, momentum=self.momentum,
                                affine=False)
            # Seed every branch from the pretrained statistics, so each starts
            # where the ImageNet-trained layer was and diverges from there.
            if getattr(source, "running_mean", None) is not None:
                bn.running_mean.copy_(source.running_mean)
                bn.running_var.copy_(source.running_var)
            self.branches.append(bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        modality = get_active_modality()
        i = MODALITY_INDEX[modality] if modality is not None else 0
        x = self.branches[i](x)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _install_film_on_fpn(model, probe_size: int = 256) -> int:
    """Attach a FiLM layer to each FPN output level.

    Conditioning at the FPN outputs rather than the image or stem keeps one
    modulation point per scale and leaves the pretrained stem untouched, which
    matters because the RPN and ROI heads both read these exact maps.

    The level names and channel counts come from one dry forward pass. They have
    to be discovered *eagerly*: `train_joint` builds the optimizer straight after
    the model, so a FiLM layer created lazily on the first real batch would never
    receive gradients and would be missing from the checkpoint.
    """
    backbone = model.backbone
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        probe = backbone(torch.zeros(1, 3, probe_size, probe_size))
    if was_training:
        backbone.train()

    films = nn.ModuleDict(
        {name.replace(".", "_"): ModalityFiLM(feat.shape[1])
         for name, feat in probe.items()}
    )
    inner = backbone.forward

    def forward(x):
        features = inner(x)
        for name in list(features):
            features[name] = films[name.replace(".", "_")](features[name])
        return features

    backbone.forward = forward
    # Registered as a submodule so the parameters reach the optimizer, the
    # `.to(device)` walk and the checkpoint; the closure only reads it.
    backbone.modality_films = films
    return len(films)


def _install_modality_bn(module: nn.Module) -> int:
    """Recursively replace BatchNorm2d with per-modality variants.

    torchvision's detection backbones use FrozenBatchNorm2d by default, which has
    no running-statistics update to specialise; those are left alone and reported
    so the caller knows conditioning was a no-op there.
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, ModalityBatchNorm2d(child))
            replaced += 1
        else:
            replaced += _install_modality_bn(child)
    return replaced


def build_joint_detection_model(
    model_type: str,
    task: str,
    conditioning: str = "film",
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    verbose: bool = True,
):
    """A single detector trained over the WLE+EUS union.

    ``conditioning``
        ``"none"``     modality-blind. The required control: without it a gain
                       cannot be attributed to conditioning rather than to simply
                       having 1.75x the images.
        ``"film"``     per-modality affine modulation of the FPN outputs.
        ``"film+bn"``  additionally gives each modality its own BatchNorm running
                       statistics.
    """
    if conditioning not in CONDITIONING_MODES:
        raise ValueError(
            f"unknown conditioning {conditioning!r}; expected one of {CONDITIONING_MODES}"
        )

    model = build_detection_model(
        model_type, task, pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    n_film = n_bn = 0
    if conditioning in ("film", "film+bn"):
        n_film = _install_film_on_fpn(model)
    if conditioning == "film+bn":
        n_bn = _install_modality_bn(model.backbone)

    if verbose:
        print(f"  conditioning={conditioning}: {n_film} FiLM level(s), "
              f"{n_bn} per-modality BatchNorm layer(s)")
        if conditioning == "film+bn" and n_bn == 0:
            print("    note: backbone uses FrozenBatchNorm2d (no running stats to "
                  "specialise) — 'film+bn' is equivalent to 'film' here")
    return model


def load_backbone_weights(model, path: str, strict: bool = False) -> dict[str, list[str]]:
    """Load a raw ``state_dict`` (a plain ``.pth``/``.pt`` file, e.g. from an
    externally-pretrained backbone such as RadImageNet or an in-domain
    endoscopy self-supervised run -- see FINDINGS.md §5c) into
    ``model.backbone.body``, the raw ResNet/MobileNetV3 feature extractor
    every registry model wraps in a ``BackboneWithFPN`` (verified identical
    structure across both backbone families).

    This is deliberately just the *mechanism*: no specific third-party weight
    file is bundled or fetched here, since key-naming compatibility can only
    be confirmed against the actual file. ``strict=False`` by default, and the
    matched/missing/unexpected key counts are returned so a mismatch is
    obvious immediately rather than silently loading nothing.
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:  # common wrapper convention (e.g. Lightning checkpoints)
        state_dict = state_dict["state_dict"]
    own_keys = set(model.backbone.body.state_dict())
    incoming_keys = set(state_dict)
    result = model.backbone.body.load_state_dict(state_dict, strict=strict)
    return {
        "matched": sorted(own_keys & incoming_keys),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }


def configure_transform_for_input_mode(model, input_mode: str, img_size: int | Sequence[int]) -> None:
    """Avoid normalising twice when the dataset already normalised.

    In ``reference`` mode the dataset hands over images that are already resized
    and standardised, so the detector's internal transform is neutralised except
    for its (identity) resize. In ``torchvision`` mode the built-in transform is
    left untouched apart from honouring ``--img-size``.

    ``img_size`` may be a single int (unchanged behaviour) or a sequence of ints
    for multi-scale training: torchvision's ``GeneralizedRCNNTransform.resize``
    randomly samples one value from ``min_size`` per image while the model is in
    training mode, and deterministically uses the *last* value in eval mode --
    so pass sizes in ascending order for a sensible (largest, most stable) eval
    scale. This only has an effect in ``"torchvision"`` input_mode: in
    ``"reference"`` mode the dataset (``CadSelDetectionDataset``) has already
    resized every image to one fixed size before the transform ever runs, so
    multiple values here would have nothing to vary against.
    """
    transform = model.transform
    sizes = (img_size,) if isinstance(img_size, int) else tuple(img_size)
    if input_mode == "reference":
        if len(sizes) > 1:
            raise ValueError(
                "multi-scale img_size only has an effect in input_mode='torchvision' -- "
                "in 'reference' mode CadSelDetectionDataset has already resized every "
                "image to one fixed size before the transform runs"
            )
        transform.image_mean = [0.0, 0.0, 0.0]
        transform.image_std = [1.0, 1.0, 1.0]
        transform.min_size = sizes
        transform.max_size = sizes[0]
    else:
        transform.min_size = sizes
        transform.max_size = max(max(sizes), transform.max_size)
