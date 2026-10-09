# llm-serving-stack: Ray Serve LLM vs llm-d in front of vLLM

Hands-on comparison of two ways to route traffic to vLLM replicas:

- **llm-d**: Envoy + the llm-d EPP (endpoint picker), a router in front of independently-run
  vLLM servers. Run here in llm-d's no-Kubernetes mode (file-based endpoint discovery).
- **Ray Serve LLM**: Ray Serve owns the vLLM engines (one per Serve replica) and routes
  between them in the Serve handle.

Both stacks serve the same model with the same vLLM version and engine args
([scripts/common.env](scripts/common.env)), and are measured with the same client-side
harness ([bench/bench.py](bench/bench.py)). Findings: [REPORT.md](REPORT.md).

## Layout

| Path | What |
|---|---|
| [scripts/common.env](scripts/common.env) | Model, vLLM engine args, host IP — shared by both stacks |
| [scripts/vllm_replicas.sh](scripts/vllm_replicas.sh) | Start/stop N vLLM replicas on ports 8000.. (llm-d backends) |
| [scripts/llmd.sh](scripts/llmd.sh) | Start/stop llm-d EPP + Envoy (Docker); API on `:8081` |
| [configs/llmd/](configs/llmd/) | EPP scheduling profiles (prefix-aware, load-only) and Envoy config |
| [ray_serve/app.py](ray_serve/app.py) | Ray Serve LLM app, `--router pow2\|prefix\|kv`; API on `:8100` |
| [bench/bench.py](bench/bench.py) | Load generator: TTFT, latency, throughput, prefix-cache hit rate |
| [scripts/run_matrix.sh](scripts/run_matrix.sh) | Runs every workload against every router variant |

## Reproduce

Requirements: NVIDIA GPU (tested on a 4 GiB RTX 2050 under WSL2), Docker, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
scripts/run_matrix.sh results/myrun.jsonl     # ~30 min on the RTX 2050
```

Or drive the pieces by hand:

```bash
source scripts/common.env
scripts/vllm_replicas.sh up 2 && scripts/llmd.sh up 2
uv run bench/bench.py --url http://$HOST_IP:8081/v1 --label llmd --workload shared-prefix
scripts/llmd.sh down && scripts/vllm_replicas.sh down

uv run ray_serve/app.py --router prefix &     # Ctrl-C / kill -INT to stop
uv run bench/bench.py --url http://$HOST_IP:8100/v1 --label ray-prefix --workload shared-prefix
```

## Environment notes (WSL2, small GPU)

These cost time to discover; each is also commented where it is handled.

- **Loopback TCP is refused under WSL2 mirrored networking** on this machine, so every
  component talks to `HOST_IP` (the LAN address), never `127.0.0.1`.
- **vLLM needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`** on WSL2, otherwise it fails with
  `UVA is not available`.
- **Two replicas on 4 GiB** need the FP8 checkpoint (`Qwen/Qwen3-0.6B-FP8`) and an explicit
  `--kv-cache-memory-bytes`: with several vLLM processes on one GPU, the
  `--gpu-memory-utilization` budget counts the other replicas and goes negative.
- **Ray under `uv run`** rebuilds the venv per worker in `/tmp/ray` unless
  `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`; fractional GPUs also need `VLLM_RAY_PER_WORKER_GPUS`.
- Ray Serve's `/v1/models` answers before the engines are loaded; wait for a completion.
