

#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 SUBSAMPLE_DIR [OUTPUT_DIR] [EXP_PREFIX]" >&2
    exit 2
fi

SUBSAMPLE_DIR=$1
OUTPUT_DIR=${2:-./outputs}
EXP_PREFIX=${3:-Data}

shopt -s nullglob
FILES=("$SUBSAMPLE_DIR"/subsample_*.h5ad)
if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No subsample_*.h5ad files found in $SUBSAMPLE_DIR" >&2
    exit 1
fi

for FILE_PATH in "${FILES[@]}"; do
    FILE_NAME=$(basename "$FILE_PATH" .h5ad)
    EXP_NAME="${EXP_PREFIX}_${FILE_NAME}"
    uv run python -u generate_3tables.py \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "$EXP_NAME"
    echo "Processing file: $FILE_PATH with experiment name: $EXP_NAME"
done
