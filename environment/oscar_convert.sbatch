#!/usr/bin/env bash
#SBATCH --job-name=ttbar-h5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ttbar-spanet/ttbar-dilep-spanet}"
ENV_NAME="${ENV_NAME:-ttbar-convert}"
INPUT_GLOB="${INPUT_GLOB:?Set INPUT_GLOB to the prod_v2 ROOT file glob}"
OUTPUT_H5="${OUTPUT_H5:-$SCRATCH/ttbar-spanet/hdf5/ttbar_dilep_v1.h5}"
TREE_NAME="${TREE_NAME:-Events}"
CHUNK_EVENTS="${CHUNK_EVENTS:-200000}"

mkdir -p "$REPO_DIR/logs" "$(dirname "$OUTPUT_H5")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

cd "$REPO_DIR"

echo "Repo: $REPO_DIR"
echo "Input: $INPUT_GLOB"
echo "Output: $OUTPUT_H5"
echo "Tree: $TREE_NAME"
echo "Chunk events: $CHUNK_EVENTS"
echo "Visible CPUs: $(nproc)"
python environment/import_test.py

python scripts/root_to_hdf5.py \
  --input "$INPUT_GLOB" \
  --output "$OUTPUT_H5" \
  --tree "$TREE_NAME" \
  --chunk-events "$CHUNK_EVENTS"
