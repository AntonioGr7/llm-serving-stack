#!/usr/bin/env bash
# llm-d routing layer without Kubernetes: EPP (endpoint picker) + Envoy, both in Docker
# on the host network, in front of the vLLM replicas from scripts/vllm_replicas.sh.
# Follows llm-d guides/no-kubernetes-deployment.
#
#   scripts/llmd.sh up [N]  # N vLLM replicas on BASE_PORT.. (default 2); API on :8081
#   scripts/llmd.sh down
#
# EPP_CONFIG selects the scheduling profile (default: optimized-baseline scorers).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.env

N=${2:-2}
EPP_IMAGE=${EPP_IMAGE:-ghcr.io/llm-d/llm-d-router-endpoint-picker:main}
ENVOY_IMAGE=${ENVOY_IMAGE:-docker.io/envoyproxy/envoy:distroless-v1.33.2}
EPP_CONFIG=${EPP_CONFIG:-configs/llmd/epp-config.yaml}
GEN=$(realpath -m logs/llmd)

render() {
  mkdir -p "$GEN"
  {
    echo "endpoints:"
    for i in $(seq 0 $((N - 1))); do
      printf '  - name: vllm-%s\n    address: %s\n    port: "%s"\n    labels:\n      model: %s\n' \
        "$i" "$HOST_IP" $((BASE_PORT + i)) "$MODEL"
    done
  } > "$GEN/endpoints.yaml"
  sed "s/127\.0\.0\.1/$HOST_IP/g" configs/llmd/envoy.yaml > "$GEN/envoy.yaml"
  cp "$EPP_CONFIG" "$GEN/config.yaml"
}

up() {
  render
  docker run -d --name llmd-epp --network host -v "$GEN:/etc/epp:ro" "$EPP_IMAGE" \
    --config-file=/etc/epp/config.yaml \
    --pool-name=file-discovery --pool-namespace=default \
    --grpc-port=9002 --grpc-health-port=9003 --metrics-port=9090 \
    --secure-serving=false --v=2

  docker run -d --name llmd-envoy --network host \
    -v "$GEN/envoy.yaml:/etc/envoy/envoy.yaml:ro" "$ENVOY_IMAGE" \
    --service-node envoy-proxy --log-level warn --concurrency 2 \
    -c /etc/envoy/envoy.yaml

  until curl -sf "$HOST_IP:19000/ready" >/dev/null; do sleep 1; done
  echo "llm-d up: http://$HOST_IP:8081/v1  (EPP $(docker inspect -f '{{.Image}}' llmd-epp | cut -c1-19))"
}

down() { docker rm -f llmd-envoy llmd-epp >/dev/null 2>&1 || true; }

"${1:-up}"
