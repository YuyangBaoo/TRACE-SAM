#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PRESET="dry"
DEVICE="cuda"
EVAL_STEPS="0"
CONFIG=""
PYTHON_BIN="${PYTHON:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset|-p)
      PRESET="$2"
      shift 2
      ;;
    --device|-d)
      DEVICE="$2"
      shift 2
      ;;
    --eval-steps|--eval_steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --config|-c)
      CONFIG="$2"
      shift 2
      ;;
    --aug-gpus|--aug_gpus)
      EXTRA_ARGS+=("--aug_gpus" "$2")
      shift 2
      ;;
    --skip-sr-pretrain|--skip_sr_pretrain)
      EXTRA_ARGS+=("--skip_sr_pretrain")
      shift
      ;;
    --skip-sr-topology|--skip_sr_topology)
      EXTRA_ARGS+=("--skip_sr_topology")
      shift
      ;;
    --skip-sr-metric|--skip_sr_metric)
      EXTRA_ARGS+=("--skip_sr_metric")
      shift
      ;;
    --skip-sr-fid|--skip_sr_fid)
      EXTRA_ARGS+=("--skip_sr_fid")
      shift
      ;;
    --skip-joint|--skip_joint)
      EXTRA_ARGS+=("--skip_joint")
      shift
      ;;
    --skip-eval|--skip_eval)
      EXTRA_ARGS+=("--skip_eval")
      shift
      ;;
    --skip-aug|--skip_aug)
      EXTRA_ARGS+=("--skip_aug")
      shift
      ;;
    --skip-aug-recognition|--skip_aug_recognition)
      EXTRA_ARGS+=("--skip_aug_recognition")
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No Python interpreter found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

if [[ -z "$CONFIG" ]]; then
  if [[ "$PRESET" == "dry" ]]; then
    CONFIG="$ROOT/configs/demo_cpu.yaml"
  else
    CONFIG="$ROOT/configs/paper_trace_sam_sr.yaml"
  fi
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$EVAL_STEPS" != "0" ]]; then
  EXTRA_ARGS+=("--eval_steps" "$EVAL_STEPS")
fi

case "$PRESET" in
  dry)
    echo
    echo "================== TRACE-SAM-SR dry workflow =================="
    if [[ ! -f "$ROOT/demo_data/manifest.csv" ]]; then
      "$PYTHON_BIN" "$ROOT/tools/make_demo_dataset.py" --out "$ROOT/demo_data" --image-size 64 --overwrite
    fi
    "$PYTHON_BIN" -m trace_sam.scripts.validate_trace_data --config "$CONFIG"
    "$PYTHON_BIN" -m trace_sam.scripts.generate_trace_aug_patches --config "$CONFIG" --dry_run
    "$PYTHON_BIN" -m trace_sam.scripts.run_full_pipeline --config "$CONFIG" --device "$DEVICE" --dry_run "${EXTRA_ARGS[@]}"
    ;;
  full)
    echo
    echo "================== TRACE-SAM-SR full workflow =================="
    "$PYTHON_BIN" -m trace_sam.scripts.run_full_pipeline --config "$CONFIG" --device "$DEVICE" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Preset must be 'dry' or 'full'." >&2
    exit 2
    ;;
esac
