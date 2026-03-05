#!/bin/bash
#SBATCH --job-name=moe-vllm-1model
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm-repro-single-%j.out
#SBATCH --error=slurm-repro-single-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TOKENOMICS_DIR="${REPO_ROOT}"

RESULTS_DIR="${REPO_ROOT}/results"
LOG_DIR="${REPO_ROOT}/repro_logs"
PORT="${PORT:-$((32000 + (SLURM_JOB_ID % 1000)))}"
API_BASE="http://127.0.0.1:${PORT}/v1"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_ALIAS="${MODEL_ALIAS:-$(echo "${MODEL_NAME}" | tr '/:@' '-' | tr -cd '[:alnum:]-_')}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64,128,256,512,1024}"
NUM_RUNS="${NUM_RUNS:-2}"
SERVER_READY_RETRIES="${SERVER_READY_RETRIES:-180}"
SERVER_READY_SLEEP_S="${SERVER_READY_SLEEP_S:-5}"
SCENARIO="${SCENARIO:-N(300,0)/(500,0)}"
DATASET_CONFIG="${DATASET_CONFIG:-${TOKENOMICS_DIR}/examples/dataset_configs/aime_simple.json}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
EXECUTION_MODE="${EXECUTION_MODE:-refill}"
NUM_PROMPTS="${NUM_PROMPTS:-0}"
MAX_CONCURRENCY_OVERRIDE="${MAX_CONCURRENCY_OVERRIDE:-}"
MAX_INPUT_TOKENS_CAP="${MAX_INPUT_TOKENS_CAP:-1024}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-512}"

VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_ASYNC_MAX_CONCURRENT_BATCHES="${VLLM_ASYNC_MAX_CONCURRENT_BATCHES:-}"
VLLM_ASYNC_INPUT_BUFFER_MAX_SETS="${VLLM_ASYNC_INPUT_BUFFER_MAX_SETS:-}"

# Qwen-30B-A3B often needs explicit memory shaping to avoid startup OOM on 1xH100.
if [[ "${MODEL_NAME}" == "Qwen/Qwen3-30B-A3B-Instruct-2507" ]]; then
  QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-16384}"
  QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-1024}"
  QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-32768}"
  if [[ -z "${VLLM_MAX_MODEL_LEN}" ]]; then
    VLLM_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN}"
  fi
  if [[ -z "${VLLM_MAX_NUM_SEQS}" ]]; then
    VLLM_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS}"
  fi
  if [[ -z "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    VLLM_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS}"
  fi
fi

RESULTS_FILE="${RESULTS_DIR}/vllm-${MODEL_ALIAS}_1xH100_bf16_in300_out500_repro_${RUN_TAG}.json"
SERVER_LOG="${LOG_DIR}/server-${MODEL_ALIAS}-${SLURM_JOB_ID:-$$}.log"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${REPO_ROOT}/repro_work"
cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH" >&2
  exit 1
fi

if [[ ! -f "${DATASET_CONFIG}" ]]; then
  echo "Dataset config not found: ${DATASET_CONFIG}" >&2
  exit 1
fi

if [[ -n "$(git -C "${TOKENOMICS_DIR}" status --porcelain)" ]]; then
  echo "Tokenomics checkout is dirty: ${TOKENOMICS_DIR}" >&2
  echo "Please clean the checkout before running reproducibility jobs." >&2
  exit 1
fi

ACTUAL_TOKENOMICS_COMMIT="$(git -C "${TOKENOMICS_DIR}" rev-parse --short=10 HEAD)"

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}
trap cleanup_server EXIT

wait_for_server() {
  for _ in $(seq 1 "${SERVER_READY_RETRIES}"); do
    if curl -sSf "${API_BASE}/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${SERVER_READY_SLEEP_S}"
  done
  return 1
}

echo "=== Single Model Repro Config ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_ALIAS=${MODEL_ALIAS}"
echo "RUN_TAG=${RUN_TAG}"
echo "RESULTS_FILE=${RESULTS_FILE}"
echo "ACTUAL_TOKENOMICS_COMMIT=${ACTUAL_TOKENOMICS_COMMIT}"
echo "API_BASE=${API_BASE}"
echo "SCENARIO=${SCENARIO}"
echo "DATASET_CONFIG=${DATASET_CONFIG}"
echo "BATCH_SIZES=${BATCH_SIZES}"
echo "NUM_RUNS=${NUM_RUNS}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "EXECUTION_MODE=${EXECUTION_MODE}"
echo "NUM_PROMPTS=${NUM_PROMPTS}"
echo "MAX_CONCURRENCY_OVERRIDE=${MAX_CONCURRENCY_OVERRIDE}"
echo "MAX_INPUT_TOKENS_CAP=${MAX_INPUT_TOKENS_CAP}"
echo "MAX_OUTPUT_TOKENS_CAP=${MAX_OUTPUT_TOKENS_CAP}"
echo "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
echo "VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
echo "VLLM_ASYNC_MAX_CONCURRENT_BATCHES=${VLLM_ASYNC_MAX_CONCURRENT_BATCHES}"
echo "VLLM_ASYNC_INPUT_BUFFER_MAX_SETS=${VLLM_ASYNC_INPUT_BUFFER_MAX_SETS}"
echo "================================="

echo "=== Serving ${MODEL_NAME} on ${API_BASE} ==="
serve_cmd=(
  uv run --no-sync --
  vllm serve "${MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --no-enable-prefix-caching
)
if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  serve_cmd+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  serve_cmd+=(--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
  serve_cmd+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
fi
printf 'Serve command: %q ' "${serve_cmd[@]}"
echo

if [[ -n "${VLLM_ASYNC_MAX_CONCURRENT_BATCHES}" ]]; then
  export VLLM_ASYNC_MAX_CONCURRENT_BATCHES
fi
if [[ -n "${VLLM_ASYNC_INPUT_BUFFER_MAX_SETS}" ]]; then
  export VLLM_ASYNC_INPUT_BUFFER_MAX_SETS
fi

"${serve_cmd[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

if ! wait_for_server; then
  echo "Server did not become ready for ${MODEL_NAME}" >&2
  tail -n 200 "${SERVER_LOG}" || true
  exit 1
fi

echo "=== Benchmarking ${MODEL_NAME} ==="
bench_cmd=(
  uv run --no-sync --
  python "${TOKENOMICS_DIR}/completion_advanced_benchmark.py"
  --model "${MODEL_NAME}"
  --scenario "${SCENARIO}"
  --dataset-config "${DATASET_CONFIG}"
  --api-base "${API_BASE}"
  --batch-sizes "${BATCH_SIZES}"
  --num-runs "${NUM_RUNS}"
  --temperature "${TEMPERATURE}"
  --tokenizer "${MODEL_NAME}"
  --description "${MODEL_ALIAS} 1xH100 bf16 in300_out500 repro ${RUN_TAG}"
  --results-file "${RESULTS_FILE}"
  --execution-mode "${EXECUTION_MODE}"
  --num-prompts "${NUM_PROMPTS}"
  --max-input-tokens-cap "${MAX_INPUT_TOKENS_CAP}"
  --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}"
)
if [[ -n "${MAX_CONCURRENCY_OVERRIDE}" ]]; then
  bench_cmd+=(--max-concurrency "${MAX_CONCURRENCY_OVERRIDE}")
fi
printf 'Benchmark command: %q ' "${bench_cmd[@]}"
echo
"${bench_cmd[@]}"

if [[ ! -s "${RESULTS_FILE}" ]]; then
  echo "Expected results file not found or empty: ${RESULTS_FILE}" >&2
  exit 1
fi

echo "RESULTS_FILE=${RESULTS_FILE}"
echo "SERVER_LOG=${SERVER_LOG}"
