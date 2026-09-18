#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 SUBSAMPLE_DIR [OUTPUT_DIR] [DEVICE_INDEX] [EXP_PREFIX]" >&2
    exit 2
fi

SUBSAMPLE_DIR=$1
OUTPUT_DIR=${2:-./outputs}
DEVICE_INDEX=${3:-0}
EXP_PREFIX=${4:-Data}

shopt -s nullglob
FILES=("$SUBSAMPLE_DIR"/subsample_*.h5ad)
if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No subsample_*.h5ad files found in $SUBSAMPLE_DIR" >&2
    exit 1
fi

# Loop over all subsample files dynamically
for FILE_PATH in "${FILES[@]}"; do
    FILE_NAME=$(basename "$FILE_PATH" .h5ad)
    EXP_NAME="${EXP_PREFIX}_${FILE_NAME}"

    # Run the DeepSAS command for each subsample
    uv run python -u deepsas_v1.py \
        --input_data_count "$FILE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "$EXP_NAME" \
        --device_index "$DEVICE_INDEX" \
        --retrain

    echo "Processing file: $FILE_PATH with experiment name: $EXP_NAME"
done
