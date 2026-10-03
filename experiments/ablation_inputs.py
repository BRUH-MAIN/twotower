"""Shared box/variant-building helpers for Experiments 12-14.

Extracted from ``experiments/src/07_mitigation.py``'s ``build()`` (itself a
near-duplicate of the same helpers in ``03_lesion_vs_background.py``), so the
three new experiments build byte-identical ``full`` / ``bg_patch`` / ``tile`` /
``box`` / ``boxmap`` arrays instead of three independently-drifting copies.
That matters here specifically because 12/13/14's retention numbers need to be
comparable to each other and to ``experiments/results/07_mitigation_verdict.json``.

Deliberate exception to this project's usual per-file-duplication style (each
numbered experiment is normally a frozen, standalone artifact -- see
``experiments/README.md``): 03/07/11 are left untouched and do NOT import this
module, both to avoid re-deriving their already-published numbers and to avoid
the "editing a live shared module mid-run" trap ``FINDINGS.md`` (S6) already
documents costing real time.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from cadsel.data import read_yolo_label

IMG = 224
LESION_TILE = 112


def boxes_for(sample, W: int, H: int) -> list[tuple[int, int, int, int]]:
    """Pixel-space xyxy boxes for one sample, dropping degenerate ones."""
    out = []
    for _c, cx, cy, bw, bh in read_yolo_label(sample.label):
        x1 = int(max(0, (cx - bw / 2) * W))
        x2 = int(min(W, (cx + bw / 2) * W))
        y1 = int(max(0, (cy - bh / 2) * H))
        y2 = int(min(H, (cy + bh / 2) * H))
        if x2 - x1 > 4 and y2 - y1 > 4:
            out.append((x1, y1, x2, y2))
    return out


def largest_free_rect(
    boxes: list[tuple[int, int, int, int]], W: int, H: int
) -> tuple[int, int, int, int] | None:
    """Biggest strip above/below/left/right of the box union -- no lesion, no marker."""
    if not boxes:
        return (0, 0, W, H)
    x1 = min(b[0] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y1 = min(b[1] for b in boxes)
    y2 = max(b[3] for b in boxes)
    cands = [(0, 0, W, y1), (0, y2, W, H), (0, 0, x1, H), (x2, 0, W, H)]
    cands = [c for c in cands if c[2] - c[0] >= 24 and c[3] - c[1] >= 24]
    return max(cands, key=lambda c: (c[2] - c[0]) * (c[3] - c[1])) if cands else None


def resize_u8(a: np.ndarray | None, size: int) -> np.ndarray:
    if a is None or a.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return np.asarray(
        Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.uint8
    )


def build_full_bgpatch_tile_box(
    samples, img_size: int = IMG, lesion_tile: int = LESION_TILE, progress_every: int = 750
) -> dict[str, np.ndarray]:
    """``full`` (resized frame), ``bg_patch`` (lesion-free rect), ``tile`` (largest
    lesion crop) and ``box`` (that lesion's box in ``img_size``-space), all uint8.

    Identical in behavior to ``07_mitigation.py``'s ``build()`` at ``img_size=224,
    lesion_tile=112`` -- verify any change here against that function's output.
    """
    n = len(samples)
    full = np.zeros((n, img_size, img_size, 3), dtype=np.uint8)
    bgp = np.zeros((n, img_size, img_size, 3), dtype=np.uint8)
    tile = np.zeros((n, lesion_tile, lesion_tile, 3), dtype=np.uint8)
    box = np.zeros((n, 4), dtype=np.int32)

    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB")
            W, H = im.size
            arr = np.asarray(im, dtype=np.uint8)
        bs = boxes_for(s, W, H)
        full[i] = resize_u8(arr, img_size)

        if bs:
            x1, y1, x2, y2 = max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            tile[i] = resize_u8(arr[y1:y2, x1:x2], lesion_tile)
            box[i] = [
                int(x1 * img_size / W),
                int(y1 * img_size / H),
                int(x2 * img_size / W),
                int(y2 * img_size / H),
            ]
            free = largest_free_rect(bs, W, H)
            bgp[i] = (
                resize_u8(arr[free[1] : free[3], free[0] : free[2]], img_size)
                if free is not None
                else np.zeros((img_size, img_size, 3), np.uint8)
            )
        else:
            tile[i] = resize_u8(arr, lesion_tile)
            box[i] = [0, 0, img_size, img_size]
            bgp[i] = full[i]

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    built {i+1}/{n}")

    return {"full": full, "bg_patch": bgp, "tile": tile, "box": box}


def boxmap_from(box: np.ndarray, img_size: int = IMG) -> np.ndarray:
    """Black canvas, the sample's lesion box drawn white -- geometry only, no tissue."""
    n = len(box)
    out = np.zeros((n, img_size, img_size, 3), dtype=np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(box):
        out[i, y1:y2, x1:x2] = 255
    return out


FIVE_VARIANTS = ("full", "lesion", "bg_mask", "bg_patch", "boxmap")


def build_five_variants(
    samples, img_size: int = IMG, progress_every: int = 750
) -> dict[str, np.ndarray]:
    """``full``, ``lesion`` (box + 10% context padding), ``bg_mask`` (box filled
    with mean colour), ``bg_patch`` (largest lesion-free rectangle) and
    ``boxmap`` (black canvas, box drawn white), all uint8 ``(N, img_size,
    img_size, 3)``.

    This is a line-for-line port of ``03_lesion_vs_background.py``'s own
    in-file ``build_variants`` -- same crop/pad/fill math, same order of
    operations -- reusing this module's ``boxes_for``/``largest_free_rect``
    where 03's local copies overlap with them. It exists so Experiment 17 (SVM
    on frozen embeddings, run through the same counterfactual protocol) builds
    numerically identical variant pixels to Experiment 3's published table,
    rather than a second implementation that might silently drift. Per this
    module's docstring, 03 itself is deliberately left untouched rather than
    importing this -- verify any future change here against 03's local
    ``build_variants`` output (``np.array_equal``) before trusting new numbers
    against the published Experiment 3 figures.
    """
    n = len(samples)
    store = {v: np.zeros((n, img_size, img_size, 3), dtype=np.uint8) for v in FIVE_VARIANTS}
    no_free = 0

    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB")
            W, H = im.size
            arr = np.asarray(im, dtype=np.uint8)
        boxes = boxes_for(s, W, H)

        store["full"][i] = resize_u8(arr, img_size)

        if boxes:
            bx1, by1, bx2, by2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            px, py = int(0.1 * (bx2 - bx1)), int(0.1 * (by2 - by1))
            store["lesion"][i] = resize_u8(
                arr[max(0, by1 - py):min(H, by2 + py), max(0, bx1 - px):min(W, bx2 + px)],
                img_size,
            )

            masked = arr.copy()
            fill = arr.reshape(-1, 3).mean(0).astype(np.uint8)
            for (a, b, c, d) in boxes:
                masked[b:d, a:c] = fill
            store["bg_mask"][i] = resize_u8(masked, img_size)

            free = largest_free_rect(boxes, W, H)
            if free is None:
                no_free += 1
                store["bg_patch"][i] = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                fx1, fy1, fx2, fy2 = free
                store["bg_patch"][i] = resize_u8(arr[fy1:fy2, fx1:fx2], img_size)

            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            for (a, b, c, d) in boxes:
                canvas[b:d, a:c] = 255
            store["boxmap"][i] = resize_u8(canvas, img_size)
        else:
            for v in FIVE_VARIANTS:
                store[v][i] = store["full"][i]

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    decoded {i+1}/{n}")

    if no_free:
        print(f"    note: {no_free}/{n} images had no lesion-free strip >=24px")
    return store
