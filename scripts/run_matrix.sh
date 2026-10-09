#!/usr/bin/env bash
# Run every workload against every router variant, one stack at a time (the GPU only
# fits one stack's two replicas). Each variant starts from fresh engines and a fresh
# router, so no prefix-cache state leaks between variants.
#
#   scripts/run_matrix.sh [results/run-name.jsonl]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.env

OUT=${1:-results/$(date +%Y%m%d-%H%M%S).jsonl}
WORKLOADS=${WORKLOADS:-"unique shared-prefix multiturn"}
BENCH_ARGS=${BENCH_ARGS:-"--requests 120 --concurrency 8"}
LLMD_VARIANTS=${LLMD_VARIANTS-"load-only:configs/llmd/epp-config-load-only.yaml prefix-aware:configs/llmd/epp-config.yaml"}
RAY_VARIANTS=${RAY_VARIANTS-"pow2 prefix"}

bench() { # label url
  for w in $WORKLOADS; do
    echo "--- $1 / $w $(date +%s)"
    uv run bench/bench.py --url "$2" --label "$1" --workload "$w" --out "$OUT" $BENCH_ARGS \
      | grep -E '"(req_per_s|ttft_ms_p50|ttft_ms_p90|prefix_hit_rate|errors)"'
  done
}

for v in $LLMD_VARIANTS; do
  name=${v%%:*}; cfg=${v#*:}
  scripts/vllm_replicas.sh down; sleep 5
  scripts/vllm_replicas.sh up 2 >/dev/null
  scripts/llmd.sh down; EPP_CONFIG=$cfg scripts/llmd.sh up 2 >/dev/null
  sleep 20  # Envoy marks the ext_proc cluster healthy after 2 checks (10s interval)
  bench "llmd-$name" "http://$HOST_IP:8081/v1"
  scripts/llmd.sh down
done
scripts/vllm_replicas.sh down; sleep 5

for r in $RAY_VARIANTS; do
  uv run ray_serve/app.py --router "$r" > "logs/ray-$r.log" 2>&1 &
  pid=$!
  # /v1/models answers before the engines load; wait for a real completion.
  until curl -sf -m20 "$HOST_IP:8100/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"max_tokens\":2}" | grep -q choices; do
    kill -0 $pid 2>/dev/null || { echo "ray ($r) died, see logs/ray-$r.log"; exit 1; }
    sleep 5
  done
  bench "ray-$r" "http://$HOST_IP:8100/v1"
  kill -INT $pid
  timeout 60 tail --pid=$pid -f /dev/null || true
  uv run ray stop --force >/dev/null 2>&1 || true; sleep 5
done

echo "results: $OUT"
