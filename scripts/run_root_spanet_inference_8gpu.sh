#!/usr/bin/env bash
set -euo pipefail

INPUT="/HEP/export/home/mstamenk/jet-charge-calibration/CMSSW_15_1_0_patch4/src/jet-charge-calib-miniaod/run/local_dilep_emu_os/merged_prod_v2/ttbar_dileptonic.root"
LOG_DIR="${HOME}/scratch/ttbar-spanet/logs/ttbar_dilep_1M_8gpu_clean/version_0"
REFERENCE_H5="${HOME}/scratch/ttbar-spanet/hdf5/splits/ttbar_dilep_v1_train.h5"
EVENT_INFO="configs/ttbar_dilep_event.yaml"
OUTPUT="${HOME}/scratch/ttbar-spanet/root/ttbar_dileptonic_spanet_augmented.root"
GPUS=8
BATCH_SIZE=8192
CHUNK_EVENTS=10000
WORKERS_PER_PROCESS=8
ORIGINAL_BRANCHES="all"

OUTPUT_DIR="$(dirname "${OUTPUT}")"
OUTPUT_BASE="$(basename "${OUTPUT}" .root)"
SHARD_DIR="${OUTPUT_DIR}/${OUTPUT_BASE}_shards"
mkdir -p "${SHARD_DIR}"

pids=()
for rank in $(seq 0 "$((GPUS - 1))"); do
  shard="${SHARD_DIR}/${OUTPUT_BASE}_rank${rank}.root"
  log="${SHARD_DIR}/${OUTPUT_BASE}_rank${rank}.log"
  echo "Starting rank ${rank}/${GPUS} on GPU ${rank}: ${shard}"
  CUDA_VISIBLE_DEVICES="${rank}" python scripts/infer_root_spanet.py \
    --input "${INPUT}" \
    --log-dir "${LOG_DIR}" \
    --reference-h5 "${REFERENCE_H5}" \
    --event-info "${EVENT_INFO}" \
    --output "${shard}" \
    --batch-size "${BATCH_SIZE}" \
    --chunk-events "${CHUNK_EVENTS}" \
    --workers "${WORKERS_PER_PROCESS}" \
    --rank "${rank}" \
    --world-size "${GPUS}" \
    --device cuda \
    --original-branches "${ORIGINAL_BRANCHES}" \
    >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" != 0 ]]; then
  echo "At least one shard failed. Check logs in ${SHARD_DIR}."
  exit 1
fi

if command -v hadd >/dev/null 2>&1; then
  echo "Merging shards into ${OUTPUT}"
  hadd -f "${OUTPUT}" "${SHARD_DIR}/${OUTPUT_BASE}"_rank*.root
  echo "Wrote ${OUTPUT}"
else
  echo "hadd was not found. Shards are complete in ${SHARD_DIR}."
  echo "After loading ROOT, merge with:"
  echo "  hadd -f ${OUTPUT} ${SHARD_DIR}/${OUTPUT_BASE}_rank*.root"
fi
