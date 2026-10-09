"""Router-agnostic load generator for OpenAI-compatible LLM endpoints.

The same workload is replayed against llm-d (Envoy+EPP) and Ray Serve LLM, and
everything is measured client-side so the two stacks are compared identically:

  * TTFT / end-to-end latency from the token stream
  * prefix-cache hit rate from usage.prompt_tokens_details.cached_tokens
    (vLLM must run with --enable-prompt-tokens-details)

Workloads (sized for a ~3.5k-token KV cache per replica, see scripts/common.env):
  unique         no shared prefixes -> routing reduces to load balancing
  shared-prefix  a few long system prompts reused by many requests: they cannot all
                 stay cached on one replica, but fit when split across replicas
  multiturn      growing conversations; each turn re-sends the whole history

  uv run bench/bench.py --url http://HOST:8081/v1 --label llmd --workload shared-prefix
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

WORDS = (
    "system policy customer order invoice refund shipping account billing support "
    "product warranty return address payment card delivery status tracking agent "
    "escalate priority ticket region currency discount coupon loyalty member tier "
    "inventory warehouse supplier contract compliance audit privacy consent data"
).split()


def text(rng: random.Random, n_words: int) -> str:
    # ~1 Qwen token per word for this vocabulary.
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


@dataclass
class Request:
    prompt: str
    session: str  # prefix group / conversation id, for reporting only
    turn: int = 0


def build_workload(name: str, n: int, seed: int) -> list[list[Request]]:
    """Return waves of requests; requests in a wave are independent, waves are ordered."""
    rng = random.Random(seed)
    if name == "unique":
        return [[Request(text(rng, 400), f"u{i}") for i in range(n)]]
    if name == "shared-prefix":
        prefixes = [f"[tenant {k}] " + text(rng, 700) for k in range(6)]
        reqs = []
        for i in range(n):
            k = rng.randrange(len(prefixes))
            reqs.append(Request(prefixes[k] + "\nQuestion: " + text(rng, 30), f"p{k}"))
        return [reqs]
    if name == "multiturn":
        # 8 conversations ending at ~550 tokens (~4.4k total): the live set fits in the
        # KV cache of two replicas only if each conversation sticks to one replica.
        # (n is ignored: too many conversations evict each other under any routing.)
        n_conv, n_turns = 8, 10
        history = [f"[conversation {c}] " + text(rng, 150) for c in range(n_conv)]
        waves = []
        for t in range(n_turns):
            wave = []
            for c in range(n_conv):
                history[c] += f"\nUser: {text(rng, 25)}\nAssistant: {text(rng, 15)}"
                wave.append(Request(history[c], f"c{c}", t))
            waves.append(wave)
        return waves
    raise ValueError(name)


@dataclass
class Result:
    session: str
    turn: int
    ok: bool
    ttft: float = 0.0
    latency: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


async def send(client: httpx.AsyncClient, url: str, model: str, req: Request, max_tokens: int) -> Result:
    body = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": max_tokens,
        "ignore_eos": True,  # fixed output length so runs are comparable
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    res = Result(req.session, req.turn, ok=False)
    try:
        async with client.stream("POST", f"{url}/completions", json=body) as r:
            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if not res.ttft and chunk.get("choices") and chunk["choices"][0].get("text"):
                    res.ttft = time.perf_counter() - t0
                if usage := chunk.get("usage"):
                    res.prompt_tokens = usage["prompt_tokens"]
                    res.output_tokens = usage["completion_tokens"]
                    details = usage.get("prompt_tokens_details") or {}
                    res.cached_tokens = details.get("cached_tokens") or 0
        res.latency = time.perf_counter() - t0
        res.ok = True
    except Exception as e:  # noqa: BLE001 - record and keep the run going
        res.error = repr(e)
    return res


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def summarize(label: str, workload: str, results: list[Result], wall: float) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft * 1000 for r in ok]
    lat = [r.latency * 1000 for r in ok]
    prompt = sum(r.prompt_tokens for r in ok)
    return {
        "label": label,
        "workload": workload,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": round(wall, 2),
        "req_per_s": round(len(ok) / wall, 2),
        "ttft_ms_p50": round(pct(ttft, 50), 1),
        "ttft_ms_p90": round(pct(ttft, 90), 1),
        "ttft_ms_mean": round(statistics.fmean(ttft), 1) if ttft else None,
        "latency_ms_p50": round(pct(lat, 50), 1),
        "latency_ms_p90": round(pct(lat, 90), 1),
        "prefix_hit_rate": round(sum(r.cached_tokens for r in ok) / prompt, 3) if prompt else None,
        "prompt_tokens_mean": round(prompt / len(ok)) if ok else None,
    }


async def run(args) -> dict:
    waves = build_workload(args.workload, args.requests, args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency * 2)
    async with httpx.AsyncClient(timeout=300, limits=limits) as client:

        async def one(req):
            async with sem:
                return await send(client, args.url, args.model, req, args.max_tokens)

        if args.warmup:
            await one(Request("warmup " + "hello " * 20, "warmup"))
        results: list[Result] = []
        t0 = time.perf_counter()
        for wave in waves:
            results += await asyncio.gather(*(one(r) for r in wave))
        wall = time.perf_counter() - t0

    summary = summarize(args.label, args.workload, results, wall)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps({**summary, "args": vars(args)}) + "\n")
        detail = out.with_name(f"{out.stem}.{args.label}.{args.workload}.requests.jsonl")
        detail.write_text("".join(json.dumps(asdict(r)) + "\n" for r in results))
    errors = [r.error for r in results if not r.ok][:3]
    if errors:
        summary["sample_errors"] = errors
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="OpenAI base URL, e.g. http://HOST:8081/v1")
    p.add_argument("--label", required=True, help="name of the stack/router under test")
    p.add_argument("--workload", choices=["unique", "shared-prefix", "multiturn"], default="shared-prefix")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-FP8")
    p.add_argument("--requests", type=int, default=120)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/results.jsonl")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = p.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
