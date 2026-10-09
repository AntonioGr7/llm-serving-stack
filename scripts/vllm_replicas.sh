#!/usr/bin/env bash
# Start N vLLM replicas of the same model on one GPU, on ports BASE_PORT..BASE_PORT+N-1.
# These are the backends the llm-d EPP routes to (Ray Serve LLM launches its own engines).
#
#   scripts/vllm_replicas.sh up [N]     # default N=2
#   scripts/vllm_replicas.sh down
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.env

N=${2:-2}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

up() {
  for i in $(seq 0 $((N - 1))); do
    port=$((BASE_PORT + i))
    echo "starting vllm-$i on :$port"
    # Replicas start one at a time: vLLM profiles free GPU memory at startup,
    # so concurrent starts race for the same budget.
    nohup uv run vllm serve "$MODEL" "${VLLM_ARGS[@]}" --port "$port" \
      > "$LOG_DIR/vllm-$i.log" 2>&1 &
    echo $! > "$LOG_DIR/vllm-$i.pid"
    until curl -sf "$HOST_IP:$port/v1/models" >/dev/null; do
      if ! kill -0 "$(cat "$LOG_DIR/vllm-$i.pid")" 2>/dev/null; then
        echo "vllm-$i died, see $LOG_DIR/vllm-$i.log"; tail -5 "$LOG_DIR/vllm-$i.log"; exit 1
      fi
      sleep 2
    done
    echo "vllm-$i ready"
  done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

down() {
  for f in "$LOG_DIR"/vllm-*.pid; do
    [ -e "$f" ] || continue
    pkill -TERM -P "$(cat "$f")" 2>/dev/null || true
    kill "$(cat "$f")" 2>/dev/null || true
    rm -f "$f"
  done
}

"${1:-up}"
