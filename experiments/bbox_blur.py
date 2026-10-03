"""Blur non-lesion pixels during training -- Experiment 13.

Experiment 3's own ``boxmap`` result (48-189% retention) proves box geometry
*alone* -- the sharp shape/position of the annotated region -- is diagnostic.
A hard-edged "blur everything outside the box" mask would recreate exactly the
trap Experiment 3 already found for ``bg_mask`` (flat-fill contaminated by the
rectangle's own size/position): the model could learn to read the blur/sharp
boundary shape as a new shortcut instead of looking at the lesion. Three
mitigations, all randomized independently per sample and per epoch:

1. The sharp/blur transition is a smooth feathered ramp (Gaussian-blur the
   *mask itself*), never a hard step.
2. The sharp region is padded to at least 1.15x the box extent in every
   direction *before* any jitter is added, so the lesion is never adjacent to
   the transition (a boundary hugging the lesion would itself outline it --
   worse than the original problem). Jitter only ever adds to this margin,
   never eats into it, and is applied independently per side, so the boundary
   is not a fixed, deterministic function of the true box shape.
3. Blur strength and whether the augmentation fires at all are also randomized
   per sample.

Evaluation must use the standard *unblurred* ``full``/``bg_patch``/``boxmap``
inputs from ``ablation_inputs`` -- blurring is training-only. The Experiment
13 script explicitly re-checks ``ret_boxmap`` against baseline as a
failure-mode probe: if it *rises*, the model is reading the blur boundary as a
geometry proxy, and the setting should be rejected or re-tuned rather than
adopted.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter


def make_feathered_mask(
    img_size: int,
    box: tuple[int, int, int, int],
    rng: random.Random,
    pad_frac_range: tuple[float, float] = (0.15, 0.5),
    jitter_frac_range: tuple[float, float] = (0.1, 0.3),
    feather_sigma_range: tuple[float, float] = (15.0, 30.0),
) -> np.ndarray:
    """An ``(img_size, img_size)`` float32 mask in [0, 1]; 1 = keep sharp.

    The sharp region's half-extent past the box edge, per side, is
    ``pad_frac + jitter`` (both drawn independently per call and per side),
    so the safe minimum padding is never reduced by jitter -- only added to.
    """
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)

    def side_pad(extent: float) -> float:
        return extent * (rng.uniform(*pad_frac_range) + rng.uniform(0.0, jitter_frac_range[1]))

    sx1 = x1 - side_pad(bw)
    sx2 = x2 + side_pad(bw)
    sy1 = y1 - side_pad(bh)
    sy2 = y2 + side_pad(bh)

    hard = np.zeros((img_size, img_size), dtype=np.uint8)
    x1c, y1c = int(np.clip(sx1, 0, img_size)), int(np.clip(sy1, 0, img_size))
    x2c, y2c = int(np.clip(sx2, 0, img_size)), int(np.clip(sy2, 0, img_size))
    if x2c > x1c and y2c > y1c:
        hard[y1c:y2c, x1c:x2c] = 255

    sigma = rng.uniform(*feather_sigma_range)
    soft = Image.fromarray(hard).filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(soft, dtype=np.float32) / 255.0


def apply_bbox_blur(
    image_u8: np.ndarray,
    box: tuple[int, int, int, int],
    rng: random.Random,
    p_apply: float = 0.5,
    sigma_range: tuple[float, float] = (4.0, 14.0),
    **mask_kwargs,
) -> np.ndarray:
    """Blurs everything outside a feathered, padded, jittered region around
    ``box``. No-ops with probability ``1 - p_apply``."""
    if rng.random() > p_apply:
        return image_u8

    blur_sigma = rng.uniform(*sigma_range)
    blurred = np.asarray(Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(blur_sigma)))
    mask = make_feathered_mask(image_u8.shape[0], box, rng, **mask_kwargs)[..., None]
    out = mask * image_u8.astype(np.float32) + (1.0 - mask) * blurred.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_blur_compose_fn(
    p_apply: float = 0.5,
    sigma_range: tuple[float, float] = (4.0, 14.0),
    pad_frac_range: tuple[float, float] = (0.15, 0.5),
    jitter_frac_range: tuple[float, float] = (0.1, 0.3),
    feather_sigma_range: tuple[float, float] = (15.0, 30.0),
):
    """A ``compose_fn`` for ``experiments.adversarial.train_eval_adversarial``:
    ``compose_fn(D_tr, idx, rng) -> np.ndarray`` replacing ``D_tr["full"][idx]``
    with per-sample bbox-blurred images, using each sample's own
    ``D_tr["box"][idx]``."""

    def compose_fn(D: dict, idx: np.ndarray, rng: random.Random) -> np.ndarray:
        images = D["full"][idx].copy()
        boxes = D["box"][idx]
        for k in range(len(idx)):
            box = tuple(int(v) for v in boxes[k])
            images[k] = apply_bbox_blur(
                images[k],
                box,
                rng,
                p_apply=p_apply,
                sigma_range=sigma_range,
                pad_frac_range=pad_frac_range,
                jitter_frac_range=jitter_frac_range,
                feather_sigma_range=feather_sigma_range,
            )
        return images

    return compose_fn
