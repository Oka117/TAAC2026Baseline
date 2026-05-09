#!/usr/bin/env bash
set -euo pipefail

# Full-data 2-epoch wall-clock benchmark for the competition platform.
#
# Usage:
#   TRAIN_DATA_PATH=/path/to/full/data ./benchmark_full_data_2epoch.sh
#   ./benchmark_full_data_2epoch.sh /path/to/full/data
#
# Optional env vars:
#   BENCH_RUNS="baseline,amp_compile"
#   BENCH_EPOCHS=2
#   BENCH_AMP_DTYPE=bf16
#   BENCH_COMPILE_MODE=reduce-overhead
#   BENCH_OUTPUT_ROOT=/path/to/output
#
# Extra train.py args are appended to every run, for example:
#   ./benchmark_full_data_2epoch.sh /data --batch_size 192

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_DIR="${BENCH_DATA_DIR:-${TRAIN_DATA_PATH:-}}"
SCHEMA_PATH=""
EXTRA_ARGS=()

# The competition platform may append --data_dir/--schema_path/--num_epochs
# to the entry command. Consume those known options here instead of forwarding
# them again to run.sh; otherwise the same arguments can be duplicated many
# times in the pod log and eventually break command parsing.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --schema_path)
      SCHEMA_PATH="$2"
      shift 2
      ;;
    --num_epochs)
      BENCH_EPOCHS="$2"
      shift 2
      ;;
    --eval_every_n_steps)
      # Benchmark always disables step-level eval; consume platform default.
      shift 2
      ;;
    --use_amp|--use_compile|--compile_dynamic)
      # These are controlled per benchmark run below.
      shift
      ;;
    --amp_dtype)
      BENCH_AMP_DTYPE="$2"
      shift 2
      ;;
    --compile_mode)
      BENCH_COMPILE_MODE="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    --*)
      EXTRA_ARGS+=("$1")
      shift
      if [[ $# -gt 0 && "$1" != --* ]]; then
        EXTRA_ARGS+=("$1")
        shift
      fi
      ;;
    *)
      if [[ -z "${DATA_DIR}" ]]; then
        DATA_DIR="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "${DATA_DIR}" ]]; then
  echo "ERROR: set TRAIN_DATA_PATH, BENCH_DATA_DIR, or pass data_dir as first arg." >&2
  exit 2
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: data directory does not exist: ${DATA_DIR}" >&2
  exit 2
fi

if [[ -z "${SCHEMA_PATH}" ]]; then
  SCHEMA_PATH="${DATA_DIR}/schema.json"
fi
if [[ ! -f "${SCHEMA_PATH}" ]]; then
  echo "ERROR: schema.json not found under data directory: ${SCHEMA_PATH}" >&2
  exit 2
fi

BENCH_RUNS="${BENCH_RUNS:-baseline,amp_compile}"
BENCH_EPOCHS="${BENCH_EPOCHS:-2}"
BENCH_AMP_DTYPE="${BENCH_AMP_DTYPE:-bf16}"
BENCH_COMPILE_MODE="${BENCH_COMPILE_MODE:-reduce-overhead}"
BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-${REPO_DIR}/outputs/bench_full_data_2epoch_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${BENCH_OUTPUT_ROOT}"
SUMMARY_PATH="${BENCH_OUTPUT_ROOT}/summary.tsv"
HELP_PATH="${BENCH_OUTPUT_ROOT}/train_help.txt"
ENV_PATH="${BENCH_OUTPUT_ROOT}/environment.txt"

python3 "${REPO_DIR}/train.py" --help > "${HELP_PATH}"
if ! grep -q -- "--use_amp" "${HELP_PATH}" || ! grep -q -- "--use_compile" "${HELP_PATH}"; then
  echo "ERROR: train.py does not expose --use_amp / --use_compile. Apply AMP/compile implementation first." >&2
  exit 2
fi

{
  echo "repo_dir=${REPO_DIR}"
  echo "data_dir=${DATA_DIR}"
  echo "schema_path=${SCHEMA_PATH}"
  echo "bench_runs=${BENCH_RUNS}"
  echo "bench_epochs=${BENCH_EPOCHS}"
  echo "bench_amp_dtype=${BENCH_AMP_DTYPE}"
  echo "bench_compile_mode=${BENCH_COMPILE_MODE}"
  echo "extra_args=${EXTRA_ARGS[*]:-<none>}"
  python3 - <<'PY'
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
        print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
except Exception as exc:
    print(f"torch_probe_error={exc}")
PY
} | tee "${ENV_PATH}"

printf "label\tstatus\tstart_epoch\tend_epoch\telapsed_sec\tlog_dir\tckpt_dir\tevents_dir\tflags\n" > "${SUMMARY_PATH}"

run_one() {
  local label="$1"
  shift
  local run_dir="${BENCH_OUTPUT_ROOT}/${label}"
  local log_dir="${run_dir}/log"
  local ckpt_dir="${run_dir}/ckpt"
  local events_dir="${run_dir}/events"
  local console_log="${run_dir}/console.log"
  local flags=("$@")

  mkdir -p "${log_dir}" "${ckpt_dir}" "${events_dir}"

  echo
  echo "================================================================"
  echo "[bench] ${label}"
  echo "flags: ${flags[*]:-<none>}"
  echo "extra_args: ${EXTRA_ARGS[*]:-<none>}"
  echo "output: ${run_dir}"
  echo "================================================================"

  local start_ts
  local end_ts
  local elapsed
  local status
  start_ts="$(date +%s)"

  set +e
  TRAIN_DATA_PATH="${DATA_DIR}" \
  TRAIN_CKPT_PATH="${ckpt_dir}" \
  TRAIN_LOG_PATH="${log_dir}" \
  TRAIN_TF_EVENTS_PATH="${events_dir}" \
  bash "${REPO_DIR}/run.sh" \
    --data_dir "${DATA_DIR}" \
    --schema_path "${SCHEMA_PATH}" \
    --num_epochs "${BENCH_EPOCHS}" \
    --eval_every_n_steps 0 \
    "${flags[@]}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${console_log}"
  status="${PIPESTATUS[0]}"
  set -e

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))

  printf "%s\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s %s\n" \
    "${label}" "${status}" "${BENCH_EPOCHS}" "${elapsed}" \
    "${log_dir}" "${ckpt_dir}" "${events_dir}" \
    "${flags[*]:-}" "${EXTRA_ARGS[*]:-}" >> "${SUMMARY_PATH}"

  echo "[bench] ${label} status=${status} elapsed_sec=${elapsed}"
  return "${status}"
}

declare -a RUN_LABELS
IFS=',' read -r -a RUN_LABELS <<< "${BENCH_RUNS}"

failures=0
for raw_label in "${RUN_LABELS[@]}"; do
  label="${raw_label// /}"
  case "${label}" in
    baseline)
      if ! run_one "baseline"; then failures=$((failures + 1)); fi
      ;;
    amp)
      if ! run_one "amp" --use_amp --amp_dtype "${BENCH_AMP_DTYPE}"; then failures=$((failures + 1)); fi
      ;;
    compile)
      if ! run_one "compile" --use_compile --compile_mode "${BENCH_COMPILE_MODE}" --compile_dynamic; then failures=$((failures + 1)); fi
      ;;
    amp_compile)
      if ! run_one "amp_compile" --use_amp --amp_dtype "${BENCH_AMP_DTYPE}" --use_compile --compile_mode "${BENCH_COMPILE_MODE}" --compile_dynamic; then failures=$((failures + 1)); fi
      ;;
    "")
      ;;
    *)
      echo "ERROR: unknown BENCH_RUNS label: ${label}" >&2
      exit 2
      ;;
  esac
done

echo
echo "================ benchmark summary ================"
cat "${SUMMARY_PATH}"
echo
awk -F '\t' '
  NR == 1 { next }
  $2 == 0 { elapsed[$1] = $5 }
  END {
    base = elapsed["baseline"]
    if (base > 0) {
      for (label in elapsed) {
        if (label != "baseline") {
          printf "%s_vs_baseline_speedup=%.3fx\n", label, base / elapsed[label]
        }
      }
    } else {
      print "baseline did not complete successfully; speedup not computed"
    }
  }
' "${SUMMARY_PATH}"

if [[ "${failures}" -ne 0 ]]; then
  echo "ERROR: ${failures} benchmark run(s) failed. See ${BENCH_OUTPUT_ROOT}." >&2
  exit 1
fi

echo "Benchmark output: ${BENCH_OUTPUT_ROOT}"
