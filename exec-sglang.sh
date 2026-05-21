#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/metroai/sglang-runtime"
ENV_FILE="${BASE_DIR}/sglang.env"
LOG_DIR="${BASE_DIR}/logs"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MODEL_PATH="${SGLANG_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${SGLANG_HOST:-127.0.0.1}"
PORT="${SGLANG_PORT:-30000}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-torch_native}"
SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"
QUANTIZATION="${SGLANG_QUANTIZATION:-}"
KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
ENABLE_CACHE_REPORT="${SGLANG_ENABLE_CACHE_REPORT:-true}"
EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
VENV_DIR="${SGLANG_VENV:-${BASE_DIR}/venv-managed}"
VENV_PY="${VENV_DIR}/bin/python"

mkdir -p "${LOG_DIR}" "${HF_HOME:-${BASE_DIR}/hf-cache}"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "SGLang python not found: ${VENV_PY}" >&2
  exit 1
fi

OPTIONAL_ARGS=()
if [[ -n "${QUANTIZATION}" ]]; then
  OPTIONAL_ARGS+=(--quantization "${QUANTIZATION}")
fi
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  OPTIONAL_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
case "${ENABLE_CACHE_REPORT,,}" in
  1|true|yes|on)
    OPTIONAL_ARGS+=(--enable-cache-report)
    ;;
esac

exec "${VENV_PY}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --sampling-backend "${SAMPLING_BACKEND}" \
  "${OPTIONAL_ARGS[@]}" \
  ${EXTRA_ARGS}
