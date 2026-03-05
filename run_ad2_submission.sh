#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <PATH_TO_MVTEC_AD_2_ROOT> [OUTPUT_DIR]"
  echo "Example: $0 /data/mvtec_ad_2 ./ad2_run"
  exit 1
fi

DATA_ROOT="$1"
OUTPUT_DIR="${2:-./ad2_run}"

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-200}"
CLASSES="${CLASSES:-can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

set -x
"$PYTHON_BIN" ad2_benchmark_pipeline.py \
  --data_root "$DATA_ROOT" \
  --output_root "$OUTPUT_DIR" \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --classes $CLASSES \
  --check_submission \
  $EXTRA_ARGS
set +x

echo "Done. Submission folder: ${OUTPUT_DIR}/submission"
echo "Archive created by checker in current working directory."
