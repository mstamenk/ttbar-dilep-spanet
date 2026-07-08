#!/usr/bin/env bash
#SBATCH --job-name=ttbar-spanet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ttbar-spanet/ttbar-dilep-spanet}"
ENV_NAME="${ENV_NAME:-ttbar-spanet}"
DATA_PATH="${DATA_PATH:-$SCRATCH/ttbar-spanet/hdf5/ttbar_dilep_v1.h5}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/configs/spanet_multitask.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/ttbar-spanet/checkpoints/$SLURM_JOB_ID}"

mkdir -p "$REPO_DIR/logs" "$OUTPUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

cd "$REPO_DIR"

echo "Repo: $REPO_DIR"
echo "Data: $DATA_PATH"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
python environment/import_test_spanet.py

# Replace this with the concrete SPANET training entry point once the model
# package is added.
python - <<'PY'
print("Environment is ready. Add the SPANET training command here.")
PY
