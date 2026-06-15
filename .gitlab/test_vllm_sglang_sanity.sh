#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 1-prefill / 1-decode NIXL KV-transfer sanity for vLLM or SGLang.
#
# Runs INSIDE the per-PR framework image on a GPU node. The image carries the test
# model weights (baked in nightly, exposed via NIXL_SANITY_MODEL_* env) and the freshly
# built NIXL wheels. Brings up a prefill instance and a decode instance wired through the
# framework's NIXL disaggregation path, sends one request, and asserts a non-empty
# completion came back through the transfer.
#
# Usage: test_vllm_sglang_sanity.sh <vllm|sglang>
set -euo pipefail

FRAMEWORK="${1:?usage: test_vllm_sglang_sanity.sh <vllm|sglang>}"
MODEL="${NIXL_SANITY_MODEL_DIR:?NIXL_SANITY_MODEL_DIR not set (baked into the base image)}/${NIXL_SANITY_MODEL_ID:?NIXL_SANITY_MODEL_ID not set}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8000}"
PROMPT="${PROMPT:-San Francisco is a}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-300}"
PROXY_TIMEOUT="${PROXY_TIMEOUT:-120}"

log() { echo "[sanity:${FRAMEWORK}] $*"; }

wait_for() {  # wait_for <url> <timeout_s>
  local url="$1" t="${2:-180}" i=0
  until curl -sf "$url" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge "$t" ]; then
      log "timeout (${t}s) waiting for ${url}"
      return 1
    fi
    sleep 1
  done
  log "ready: ${url}"
}

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT

log "model: ${MODEL}"
nvidia-smi -L
python3 -c "import nixl; from importlib.metadata import version; print('nixl', version('nixl'))"

if [ "$FRAMEWORK" = "vllm" ]; then
  python3 -c "from importlib.metadata import version; print('vllm', version('vllm'))"

  CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --port "$PREFILL_PORT" \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' &
  pids+=($!)

  CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL" --port "$DECODE_PORT" \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' &
  pids+=($!)

  wait_for "http://localhost:${PREFILL_PORT}/health" "$SERVER_TIMEOUT"
  wait_for "http://localhost:${DECODE_PORT}/health" "$SERVER_TIMEOUT"

  # toy_proxy_server.py was baked into the image by contrib/Dockerfile.vllm.
  python3 /toy_proxy_server.py --port "$PROXY_PORT" \
    --prefiller-port "$PREFILL_PORT" --decoder-port "$DECODE_PORT" &
  pids+=($!)
  wait_for "http://localhost:${PROXY_PORT}/health" "$PROXY_TIMEOUT" || true
  ENDPOINT="http://localhost:${PROXY_PORT}/v1/completions"

elif [ "$FRAMEWORK" = "sglang" ]; then
  python3 -c "from importlib.metadata import version; print('sglang', version('sglang'))"

  CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path "$MODEL" \
    --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
    --trust-remote-code --host 0.0.0.0 --port "$PREFILL_PORT" &
  pids+=($!)

  CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path "$MODEL" \
    --disaggregation-mode decode --disaggregation-transfer-backend nixl \
    --trust-remote-code --host 0.0.0.0 --port "$DECODE_PORT" &
  pids+=($!)

  wait_for "http://localhost:${PREFILL_PORT}/health" "$SERVER_TIMEOUT"
  wait_for "http://localhost:${DECODE_PORT}/health" "$SERVER_TIMEOUT"

  # SGLang mini load balancer fronts the prefill/decode pair (sglang.srt.disaggregation).
  python3 -m sglang.srt.disaggregation.mini_lb \
    --prefill "http://localhost:${PREFILL_PORT}" \
    --decode "http://localhost:${DECODE_PORT}" \
    --host 0.0.0.0 --port "$PROXY_PORT" &
  pids+=($!)
  wait_for "http://localhost:${PROXY_PORT}/health" "$PROXY_TIMEOUT" || true
  ENDPOINT="http://localhost:${PROXY_PORT}/v1/completions"

else
  log "unknown framework: ${FRAMEWORK}"
  exit 2
fi

log "sending request through the disaggregation proxy"
RESP="$(curl -sf "$ENDPOINT" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":16,\"temperature\":0}")"
echo "$RESP"

# Assert a non-empty completion came back through the NIXL transfer path.
RESP="$RESP" python3 - <<'PY'
import json
import os

resp = json.loads(os.environ["RESP"])
text = resp["choices"][0]["text"]
assert text and text.strip(), f"empty completion: {resp!r}"
print("[sanity] transfer OK, completion:", repr(text))
PY

log "PASS"
