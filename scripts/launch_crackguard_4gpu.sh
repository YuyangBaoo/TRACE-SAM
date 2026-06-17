#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-runs/trace_sam_sr/full_image_aug_v3_unfreeze_0611/configs/trace_sam_runtime.yaml}"
SR_RESULTS_ROOT="${SR_RESULTS_ROOT:-results/crackguard_diffsr/sr_ablation}"
SR_CFG_ROOT="${SR_CFG_ROOT:-configs/crackguard_diffsr/ablations}"
WORK_ROOT="${WORK_ROOT:-results/crackguard_diffsr}"
LOG_ROOT="${LOG_ROOT:-results/crackguard_diffsr/logs/$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-1234}"
SR_EPOCHS="${SR_EPOCHS:-50}"
REC_EPOCHS="${REC_EPOCHS:-16}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_ROOT}"

run_or_print() {
  local name="$1"
  local gpu="$2"
  local cmd="$3"
  local log="${LOG_ROOT}/${name}.log"
  echo
  echo "### ${name} on GPU${gpu}"
  echo "LOG=${log}"
  echo "CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=. ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  nohup bash -lc "CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=. ${cmd}" >"${log}" 2>&1 &
  echo "$!" > "${LOG_ROOT}/${name}.pid"
  echo "PID=$(cat "${LOG_ROOT}/${name}.pid")"
}

phase1_plan() {
  local common_sr
  common_sr="${PYTHON_BIN} -m trace_sam.scripts.run_trace_sam_sr_suite --base_config ${BASE_CONFIG} --cfg_root ${SR_CFG_ROOT} --results_root ${SR_RESULTS_ROOT} --seeds ${SEED} --epochs ${SR_EPOCHS} --eval_degradations clean_x4 --device cuda --skip_fid --skip_lpips"

  run_or_print "gpu0_sr_no_fracture_field" 0 "${common_sr} --variants no_fracture_field"
  run_or_print "gpu1_sr_no_gated_refiner" 1 "${common_sr} --variants no_gated_refiner"
  run_or_print "gpu2_sr_no_structure_losses" 2 "${common_sr} --variants no_structure_losses"

  local queue_script="${LOG_ROOT}/gpu3_recognition_queue.sh"
cat > "${queue_script}" <<QUEUE
set -euo pipefail
cd "${PROJECT_ROOT}"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
"${PYTHON_BIN}" scripts/prepare_bicubic_aug_control.py --config "${BASE_CONFIG}"
"${PYTHON_BIN}" scripts/run_recognition_variant.py --base_config "${BASE_CONFIG}" --variant bicubic_aug --work_root "${WORK_ROOT}" --seed "${SEED}" --epochs "${REC_EPOCHS}" --device cuda
"${PYTHON_BIN}" scripts/run_recognition_variant.py --base_config "${BASE_CONFIG}" --variant no_sr_uncertainty --work_root "${WORK_ROOT}" --seed "${SEED}" --epochs "${REC_EPOCHS}" --device cuda
"${PYTHON_BIN}" scripts/run_recognition_variant.py --base_config "${BASE_CONFIG}" --variant no_thin_line_refiner --work_root "${WORK_ROOT}" --seed "${SEED}" --epochs "${REC_EPOCHS}" --device cuda
QUEUE
  chmod +x "${queue_script}"
  run_or_print "gpu3_recognition_and_bicubic_queue" 3 "bash ${queue_script}"

  cat > "${LOG_ROOT}/launch_manifest.txt" <<EOF
started_at=$(date -Is)
base_config=${BASE_CONFIG}
seed=${SEED}
sr_epochs=${SR_EPOCHS}
recognition_epochs=${REC_EPOCHS}
sr_results_root=${SR_RESULTS_ROOT}
work_root=${WORK_ROOT}
log_root=${LOG_ROOT}
dry_run=${DRY_RUN}
EOF
}

status() {
  echo "### GPU"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo
  echo "### tracked jobs"
  shopt -s nullglob
  for pidfile in "${LOG_ROOT}"/*.pid; do
    local name pid
    name="$(basename "${pidfile}" .pid)"
    pid="$(cat "${pidfile}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "${name}: RUNNING pid=${pid}"
    else
      echo "${name}: EXITED pid=${pid}"
    fi
    local log="${LOG_ROOT}/${name}.log"
    if [[ -f "${log}" ]]; then
      tail -n 5 "${log}" || true
    fi
    echo
  done
}

usage() {
  cat <<EOF
Usage:
  $0 plan      # print the 4-GPU schedule without launching
  $0 start     # launch phase-1: 3 SR ablations + GPU3 recognition/control queue
  $0 status    # show GPU state and tails from the current LOG_ROOT

Environment overrides:
  SEED=1234 SR_EPOCHS=50 REC_EPOCHS=16 DRY_RUN=1 LOG_ROOT=...
EOF
}

cmd="${1:-plan}"
case "${cmd}" in
  plan)
    DRY_RUN=1 phase1_plan
    ;;
  start)
    phase1_plan
    ;;
  status)
    status
    ;;
  *)
    usage
    exit 2
    ;;
esac
