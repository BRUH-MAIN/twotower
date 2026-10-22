#!/usr/bin/env bash
# Seeds 123 and 7 for the backbone x task cells the ablation ran with seed 42
# only, so every backbone gets a 3-seed ensemble (42's ensemble grids, 44's
# ensemble CIs). Training resumes per epoch and per fold, so re-running is
# safe. After training, everything downstream is regenerated.
set -u
cd "$(dirname "$0")/.."
export CADSEL_DATA_ROOT="${CADSEL_DATA_ROOT:-/run/media/rohan/New Volume/CAD-SEL-Dataset}"
PY="${PY:-.venv-linux/bin/python}"
CFG="fusion-gated__loss-weighted_ce__moddrop0.3__mf4__ep100__es20__lesioncrop__nested0.2"

# shortest first; ConvNeXt (~2.5x slower per epoch) last
SEEDS="123 7" TASKS=binary  BACKBONES="resnet50"                       bash experiments/run_two_tower_ablation.sh
SEEDS="123 7" TASKS=subtype BACKBONES="mobilenet_v3_large mobilevit_s" bash experiments/run_two_tower_ablation.sh
SEEDS="123 7" TASKS="binary subtype" BACKBONES="convnext_tiny"         bash experiments/run_two_tower_ablation.sh

echo "$(date '+%F %T') scoring (41)"
$PY -m experiments.src.41_two_tower_ablation_bargraphs
for cell in "resnet50 binary" "mobilenet_v3_large subtype" "mobilevit_s subtype" \
            "convnext_tiny binary" "convnext_tiny subtype"; do
  set -- $cell
  echo "$(date '+%F %T') ensemble (35) $1 $2"
  $PY -m experiments.src.35_joint_fusion_multiseed_ensemble --run-names \
      "joint_$1__$2__freeze-none__${CFG}__s42" "joint_$1__$2__freeze-none__${CFG}__s123" \
      "joint_$1__$2__freeze-none__${CFG}__s7"
done
echo "$(date '+%F %T') confusion matrices (42), stats (44)"
$PY -m experiments.src.42_two_tower_confusion_matrices
$PY -m experiments.src.44_two_tower_stats
echo "$(date '+%F %T') all done"
