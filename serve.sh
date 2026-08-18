#!/bin/zsh
# Start mlx_lm.server with the Alchemist. Leave running; probes talk to it.
#
# --decode-concurrency / --prompt-concurrency are what make llm_query_batched
# actually parallel: without them the server would serialise the batch.
set -eu
TARGET="${1:-alchemist}"
case "$TARGET" in
  alchemist) MODEL="${ALCHEMIST_MODEL:-}" ;;
  qwen9b) MODEL="${QWEN9B_MODEL:-}" ;;
  qwen4b-bf16) MODEL="${QWEN4B_BF16_MODEL:-}" ;;
  agents-bf16) MODEL="${AGENTS_BF16_MODEL:-}" ;;
  *) echo "usage: ./serve.sh [alchemist|agents-bf16|qwen4b-bf16|qwen9b]" >&2; exit 2 ;;
esac
[ -n "$MODEL" ] || {
  echo "model path is not configured for $TARGET; set the matching *_MODEL variable" >&2
  exit 2
}
MLX="${RLM_SERVER_PYTHON:-$(command -v python3)}"
cd "$(dirname "$0")"
mkdir -p logs
echo "loading $TARGET from $MODEL"
# serve_patched wraps the qwen3_coder tool parser with a recovery path:
# a <tool_call> with no <function=> becomes a PythonInterpreter call
# instead of a dropped connection. Four episodes died to that defect.
# MLX balances assistant, user and system cache entries separately. Reducing
# its original ten entries changed greedy trajectories even with identical
# prompts. Keep the original conversation policy; the independent byte budget
# trims inactive entries before active prefill to control memory.
exec "$MLX" scripts/serve_patched.py \
  --model "$MODEL" \
  --port 8081 \
  --max-tokens 4096 \
  --decode-concurrency 1 \
  --prompt-concurrency 1 \
  --prompt-cache-size 10 \
  --prompt-cache-bytes 1GB \
  2>&1 | tee logs/mlx_server_${TARGET}.log
