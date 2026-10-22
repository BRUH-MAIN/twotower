#!/usr/bin/env bash
# Two-tower backbone ablation queue: 34_joint_fusion_train.py under the
# leak-free nested protocol, one GPU job at a time. Each run resumes per
# epoch / per fold if the queue is interrupted, so re-running is safe.
#
#   bash experiments/run_two_tower_ablation.sh                        # stage 1: seed 42, all backbones
#   SEEDS="123 7" BACKBONES="efficientnet_b0 dinov3_vits16" bash experiments/run_two_tower_ablation.sh
#
# Follow-up variants (env, default = the ablation's own config):
#   LESION_CROP=0        full frames instead of the GT-box lesion crop
#   FUSION=symmetric     SymmetricCrossAttentionFusion instead of GatedFusion
#   EXT_CENTRES=Center-2,Center-3,Center-4,Center-5,Center-6   one external test set
#   DATASET=limuc        another dataset from cadsel.external_datasets (TASKS must be its tasks)
#   MODALITIES=WLE       a subset of the dataset's modalities (single-backbone run)
set -u
cd "$(dirname "$0")/.."
export CADSEL_DATA_ROOT="${CADSEL_DATA_ROOT:-/run/media/rohan/New Volume/CAD-SEL-Dataset}"
PY="${PY:-.venv-linux/bin/python}"
SEEDS="${SEEDS:-42}"
TASKS="${TASKS:-binary subtype}"
BACKBONES="${BACKBONES:-efficientnet_b0 dinov3_vits16 deit_s convnext_tiny resnet50 mobilenet_v3_large mobilevit_s}"
LESION_CROP="${LESION_CROP:-1}"
FUSION="${FUSION:-gated}"
EXT_CENTRES="${EXT_CENTRES:-}"
DATASET="${DATASET:-cadsel}"
MODALITIES="${MODALITIES:-}"
VARIANT=""
[ "$LESION_CROP" = 1 ] && CROP_ARG="--lesion-crop" || { CROP_ARG=""; VARIANT+="_fullframe"; }
[ "$FUSION" = gated ] || VARIANT+="_$FUSION"
EXT_ARG=()
[ -n "$EXT_CENTRES" ] && { EXT_ARG=(--external-test-centres "$EXT_CENTRES"); VARIANT+="_ext"; }
DS_ARG=(--dataset "$DATASET")
[ "$DATASET" = cadsel ] || VARIANT+="_$DATASET"
[ -n "$MODALITIES" ] && { DS_ARG+=(--modalities "$MODALITIES"); VARIANT+="_${MODALITIES//,/+}"; }

# ViTs train only their last 4 blocks (full DINOv3 fine-tuning collapsed in
# the single-modality screen); everything else is fully fine-tuned.
declare -A FREEZE=([deit_s]=partial [dinov3_vits16]=partial [dinov3_vitb16]=partial [dinov3_vitl16]=partial)
# Per-backbone overrides from the stage-0 smoke test (memory), e.g.
# [resnet50]="--batch-size 4 --grad-accum 2"
declare -A EXTRA=()

for seed in $SEEDS; do
  for task in $TASKS; do
    for bb in $BACKBONES; do
      log="logs/two_tower_${bb}_${task}_s${seed}_nested${VARIANT}.log"
      echo "$(date '+%F %T') start $bb $task s$seed -> $log"
      $PY -m experiments.src.34_joint_fusion_train --task "$task" --backbone "$bb" \
          --freeze "${FREEZE[$bb]:-none}" $CROP_ARG --fusion "$FUSION" "${EXT_ARG[@]}" "${DS_ARG[@]}" \
          --epochs 100 --early-stop-patience 20 \
          --inner-val-fraction 0.2 --seed "$seed" ${EXTRA[$bb]:-} >> "$log" 2>&1
      rc=$?
      echo "$(date '+%F %T') done  $bb $task s$seed exit=$rc"
    done
  done
done
