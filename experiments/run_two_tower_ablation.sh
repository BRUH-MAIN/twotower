#!/usr/bin/env bash
# Two-tower backbone ablation queue: 34_joint_fusion_train.py under the
# leak-free nested protocol, one GPU job at a time. Each run resumes per
# epoch / per fold if the queue is interrupted, so re-running is safe.
#
#   bash experiments/run_two_tower_ablation.sh                        # stage 1: seed 42, all backbones
#   SEEDS="123 7" BACKBONES="efficientnet_b0 dinov3_vits16" bash experiments/run_two_tower_ablation.sh
set -u
cd "$(dirname "$0")/.."
export CADSEL_DATA_ROOT="${CADSEL_DATA_ROOT:-/run/media/rohan/New Volume/CAD-SEL-Dataset}"
PY="${PY:-.venv-linux/bin/python}"
SEEDS="${SEEDS:-42}"
TASKS="${TASKS:-binary subtype}"
BACKBONES="${BACKBONES:-efficientnet_b0 dinov3_vits16 deit_s convnext_tiny resnet50 mobilenet_v3_large mobilevit_s}"

# ViTs train only their last 4 blocks (full DINOv3 fine-tuning collapsed in
# the single-modality screen); everything else is fully fine-tuned.
declare -A FREEZE=([deit_s]=partial [dinov3_vits16]=partial)
# Per-backbone overrides from the stage-0 smoke test (memory), e.g.
# [resnet50]="--batch-size 4 --grad-accum 2"
declare -A EXTRA=()

for seed in $SEEDS; do
  for task in $TASKS; do
    for bb in $BACKBONES; do
      log="logs/two_tower_${bb}_${task}_s${seed}_nested.log"
      echo "$(date '+%F %T') start $bb $task s$seed -> $log"
      $PY -m experiments.src.34_joint_fusion_train --task "$task" --backbone "$bb" \
          --freeze "${FREEZE[$bb]:-none}" --lesion-crop --epochs 100 --early-stop-patience 20 \
          --inner-val-fraction 0.2 --seed "$seed" ${EXTRA[$bb]:-} >> "$log" 2>&1
      rc=$?
      echo "$(date '+%F %T') done  $bb $task s$seed exit=$rc"
    done
  done
done
