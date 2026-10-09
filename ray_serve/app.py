"""Ray Serve LLM deployment of the same model/engine config used behind llm-d.

Unlike llm-d (a router in front of independently-run vLLM servers), Ray Serve LLM
owns the engines: each Serve replica embeds a vLLM AsyncLLM, and routing happens
in the Serve handle between the OpenAI ingress and those replicas.

  uv run ray_serve/app.py --router pow2      # Ray default: power-of-two-choices on queue length
  uv run ray_serve/app.py --router prefix    # PrefixCacheAffinityRouter (approximate prefix tree)
  uv run ray_serve/app.py --router kv        # KVAwareRouter (KV events; needs ai-dynamo)

Serves the OpenAI API on http://HOST_IP:8100/v1 until interrupted.
"""

import argparse
import os
import signal
import subprocess

# Under `uv run`, Ray otherwise rebuilds the whole venv (vLLM, torch, ...) per worker
# in /tmp/ray. Workers should reuse the project's .venv, like the driver.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
# Two engines under Ray take ~8.6 GiB of host RAM (Serve replica + EngineCore + RayWorkerProc
# per engine), right at Ray's 95% kill threshold on an 11 GiB laptop. Let swap absorb idle
# startup memory instead of having Ray kill the engines.
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import ray  # noqa: E402
from ray import serve  # noqa: E402
from ray.serve.llm import LLMConfig, build_openai_app  # noqa: E402

ROUTERS = {
    "pow2": None,  # Serve's default PowerOfTwoChoicesRequestRouter
    "prefix": "ray.serve.llm.request_router.PrefixCacheAffinityRouter",
    "kv": "ray.serve.llm.request_router.KVAwareRouter",
}
PORT = 8100


def common_env() -> dict:
    """Read MODEL / VLLM_ARGS from scripts/common.env so both stacks share one config."""
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "common.env")
    out = subprocess.check_output(
        ["bash", "-c", f'source "{script}"; echo "$MODEL"; echo "$HOST_IP"; printf "%s\\n" "${{VLLM_ARGS[@]}}"'],
        text=True,
    ).splitlines()
    model, host_ip, args = out[0], out[1], out[2:]
    kwargs, i = {}, 0
    while i < len(args):
        key = args[i].removeprefix("--").replace("-", "_")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            value = args[i + 1]
            kwargs[key] = float(value) if "." in value else int(value) if value.isdigit() else value
            i += 2
        else:
            kwargs[key] = True
            i += 1
    return {"model": model, "host_ip": host_ip, "engine_kwargs": kwargs}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router", choices=ROUTERS, default="pow2")
    p.add_argument("--replicas", type=int, default=2)
    args = p.parse_args()

    env = common_env()
    deployment_config = {"num_replicas": args.replicas, "max_ongoing_requests": 64}
    if ROUTERS[args.router]:
        deployment_config["request_router_config"] = {"request_router_class": ROUTERS[args.router]}

    llm_config = LLMConfig(
        model_loading_config={"model_id": env["model"], "model_source": env["model"]},
        engine_kwargs=env["engine_kwargs"],
        # All replicas share the single GPU, like the standalone vLLM replicas do.
        placement_group_config={"bundle_per_worker": {"GPU": 1 / args.replicas, "CPU": 1}},
        deployment_config=deployment_config,
        runtime_env={
            "env_vars": {
                "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
                # vLLM's Ray executor otherwise asks for a whole GPU per worker,
                # which cannot fit in the fractional bundle above.
                "VLLM_RAY_PER_WORKER_GPUS": str(1 / args.replicas),
            }
        },
    )

    # Keep Ray's own footprint small: this host has ~7 GiB of RAM to spare, and Ray's
    # memory monitor kills workers (the vLLM engines) above 95% host memory.
    ray.init(
        runtime_env={"env_vars": {"VLLM_WSL2_ENABLE_PIN_MEMORY": "1"}},
        include_dashboard=False,
        object_store_memory=512 * 1024**2,
    )
    serve.start(http_options={"host": "0.0.0.0", "port": PORT})
    serve.run(build_openai_app({"llm_configs": [llm_config]}), blocking=False)
    print(f"Ray Serve LLM ({args.router}) up: http://{env['host_ip']}:{PORT}/v1", flush=True)
    # Signals must be blocked for sigwait to receive them instead of the default handlers.
    stop = {signal.SIGINT, signal.SIGTERM}
    signal.pthread_sigmask(signal.SIG_BLOCK, stop)
    signal.sigwait(stop)
    serve.shutdown()


if __name__ == "__main__":
    main()
