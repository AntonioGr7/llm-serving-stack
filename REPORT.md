# Ray Serve LLM vs llm-d: when to use which

*Draft, 2026-10-07. Ray 2.59.0 · llm-d router (EPP) `main` @ `sha256:a7778665` · vLLM 0.27.0 ·
Qwen3-0.6B-FP8 on one RTX 2050 (4 GiB), two replicas.*

> **Status:** 8 of 12 benchmark cells measured, one run each. Missing:
> - Ray `PrefixCacheAffinityRouter`: it does not fit this laptop's 11 GiB of RAM, failing in two
>   attempts (see "Resource footprint" below).
> - llm-d load-only on the corrected multiturn workload, plus repeat runs. Both were attempted
>   on 2026-10-09 but discarded: the host was swapping, which distorted the numbers.
>
> See [Next steps](#next-steps).

## TL;DR

- **The decision is mostly about the platform around the router, not the routing algorithm.**
  Both stacks now offer cache-aware routing. Both will beat cache-blind routing on workloads
  with shared prefixes, and both are moving toward precise, KV-event-driven routing.
- **Use llm-d** when LLM serving is a platform service, usually on Kubernetes:
  - many models or tenants behind one gateway;
  - priority, fairness and SLO-aware routing;
  - prefill/decode disaggregation or wide expert parallelism at scale;
  - a mix of accelerators (NVIDIA, AMD, TPU) or engines (vLLM, SGLang).

  Its router is a Go service behind Envoy, and it builds on the Kubernetes Gateway API Inference Extension standard.
- **Use Ray Serve LLM** when the LLM is one part of a Python system that already runs on Ray:
  - multi-model pipelines and composition;
  - batch inference with Ray Data;
  - RL post-training (verl, OpenRLHF);
  - non-LLM models alongside the LLM;
  - custom Python routing logic;
  - deployments without Kubernetes.
- **They can be combined.** Keep Ray for orchestration, batch and RL, and put the llm-d EPP in
  front of the vLLM replicas for online traffic. llm-d documents this pattern for verl on Ray.
- **On our bench, llm-d with prefix-aware routing was best on every workload measured.** It beat
  Ray's default router by 1.5–1.7x in throughput, and cut multiturn p50 TTFT by 2.5x. The
  comparison with Ray's own prefix router is still missing.

## What we measured

Same model, same vLLM version, same engine arguments for both stacks
([scripts/common.env](scripts/common.env)). Each replica has about 3,000 tokens of KV cache.
That is deliberately small: cache-aware routing only matters under cache pressure.

The client is identical for every run ([bench/bench.py](bench/bench.py)):
- 120 requests at concurrency 8, 32 output tokens each (`ignore_eos`), streaming;
- TTFT and latency are measured client-side;
- the prefix-cache hit rate comes from vLLM's `usage.prompt_tokens_details.cached_tokens`.

| Workload | Shape | What it tests |
|---|---|---|
| `unique` | ~400-token prompts, nothing shared | Pure load balancing and router overhead |
| `shared-prefix` | 6 "tenant" system prompts of ~700 tokens + a short question | The prefixes don't fit in one replica's cache (~4.2k > 3k) but do fit when split across two |
| `multiturn` | 8 conversations × 10 turns, each turn re-sends the history | Session stickiness |

Router variants. Each stack has a load-only control and a cache-aware variant:

| Stack | Load-only | Cache-aware |
|---|---|---|
| llm-d (Envoy + EPP, no-Kubernetes mode) | `queue` + `kv-cache-utilization` scorers | optimized-baseline: + `prefix-cache` (approximate) + `no-hit-lru` |
| Ray Serve LLM | `pow2` (default) | `PrefixCacheAffinityRouter` |

## Results (single run per cell)

| Variant | Workload | req/s | TTFT p50 (ms) | TTFT p90 (ms) | Latency p50 (ms) | Prefix hit |
|---|---|--:|--:|--:|--:|--:|
| llm-d load-only | unique | 2.16 | 646 | 1724 | 3568 | 0% |
| llm-d prefix-aware | unique | 2.49 | 448 | 1447 | 3017 | 0% |
| Ray pow2 | unique | 1.65 | 795 | 1309 | 4545 | 0% |
| llm-d load-only | shared-prefix | 2.16 | 703 | 1891 | 3422 | 53.8% |
| **llm-d prefix-aware** | shared-prefix | **2.79** | 616 | 1427 | **2653** | **67.9%** |
| Ray pow2 | shared-prefix | 1.82 | 619 | 1268 | 3954 | 62.6% |
| llm-d load-only | multiturn | *pending* | | | | |
| **llm-d prefix-aware** | multiturn | **3.42** | **279** | **748** | **2135** | **71.2%** |
| Ray pow2 | multiturn | 2.04 | 710 | 1513 | 3516 | 47.5% |
| Ray PrefixCacheAffinity | all | *pending* | | | | |

Raw data: [results/run1.jsonl](results/run1.jsonl), with per-request detail next to it.

### Reading the numbers

- **Cache-aware routing pays off where prefixes repeat.** On `multiturn`, llm-d prefix-aware
  served 1.7x Ray pow2's throughput with 2.5x lower p50 TTFT, and the hit rate rose from 47%
  to 71%. On `shared-prefix`, it beat llm-d's own load-only profile by 29% in throughput, with
  hits up from 54% to 68%.
- **Ray was slower even with nothing to cache.** On `unique`, Ray pow2 reached 1.65 req/s
  against 2.2–2.5 for llm-d, on the same engines. Ray's default HTTP path is a Python proxy
  and ingress deployment, plus a Ray executor process per engine. Ray's newer HAProxy and
  "direct streaming" mode targets exactly this overhead, but it is still experimental and we
  did not enable it. Treat this gap as a property of the default configuration, not a hard limit.
- **Noise:** each cell is one run. The two llm-d variants differ by 15% on `unique`, where
  routing policy should not matter, so differences below about 15–20% are not meaningful here.
- **Resource footprint:** Ray Serve LLM needs much more host RAM than standalone vLLM behind
  llm-d.
  - Under Ray, each engine runs as three processes: the Serve replica (~1.2 GiB), the vLLM
    EngineCore (~0.8 GiB) and a `RayWorkerProc` (~1.5 GiB). Two engines plus the ingress,
    driver and controller came to **~8.6 GiB** of host RAM.
  - llm-d's router is two small containers (Envoy + EPP) next to plain vLLM processes. Those ran
    without memory pressure even while other services held ~4–5 GiB.
  - With `PrefixCacheAffinityRouter`, the Ray ingress (where the router runs) grew from ~0.7 to
    2.2 GiB and was restarted. The engines got killed by Ray's memory monitor, or swapped out
    when it was disabled, so we could not complete this variant on an 11 GiB laptop.
  - This is an artifact of a tiny host, but it shows the operational cost of Ray's
    actor-per-component design.

### Limits of this bench

- **Scale:** two replicas of a 0.6B model on a 4 GiB laptop GPU, with one shared GPU. At this
  scale the router overhead is a large share of each request. Published data center results
  show bigger cache-aware gains on long prompts, but they are not head-to-head.
- **Precise prefix routing was not tested.** That is llm-d's KV-event mode and Ray's
  `KVAwareRouter`, which needs NVIDIA's `ai-dynamo`.
- **Not testable on one GPU:** P/D disaggregation, wide expert parallelism, multi-node,
  autoscaling, high availability.

## What we could not test: the evidence from docs and published benchmarks

No independent llm-d vs Ray Serve head-to-head has been published. Vendor numbers:

**llm-d:**
- **Precise vs approximate vs load-only vs random routing** (Qwen3-32B, 8×2 H100, 150 tenants with 6k-token contexts):
  - output tok/s: 8,730 vs 6,944 vs 4,429 vs 4,429;
  - p90 TTFT: 0.54 s vs 31 s vs 95 s vs 93 s.
- **Predicted-latency routing:** −47% p90 TTFT versus a plain Kubernetes Service.

**Ray:**
- **`PrefixCacheAffinityRouter`:** the ">2.5x" claim is *input-token* throughput versus pow2.
  The same benchmark reports −60% TTFT and +40% end-to-end throughput, on Anyscale's
  proprietary runtime.
- **HAProxy + direct streaming:** 4–25x over the old Ray Serve LLM path, now on par with vllm-router.

| Capability | llm-d | Ray Serve LLM |
|---|---|---|
| Where routing runs | Go EPP via Envoy ext_proc; ~0.1–0.2 ms per decision; HA is active-passive when prefix state matters | Python router inside each Serve handle; scales out with ingress replicas; prefix state in one actor |
| Routing policies | prefix (approximate and precise via KV events), queue, KV utilization, token load, predicted latency, LoRA affinity, session affinity, flow control with priority, fairness and SLO tiers | pow2, prefix affinity (stable), KV-aware (alpha, needs ai-dynamo), custom `RequestRouter` in Python |
| P/D disaggregation | Built into the router (NIXL, MoRI-IO, offloading connectors), multimodal E/P/D | `build_pd_openai_app` (stable), NIXL/LMCache connectors |
| Wide-EP (MoE) | Well-lit path (DeepSeek-R1 on 32×H200/B200); needs a custom vLLM image | `build_dp_openai_app` (stable), gang scheduling |
| Tiered KV cache | CPU, filesystem, LMCache, Mooncake offload; P2P prefix sharing between pods | Engine-side via vLLM connectors only |
| Autoscaling | KEDA/HPA on inference metrics (queue, saturation, tokens), scale to zero; needs Prometheus | Built-in, on queue length (`target_ongoing_requests`), custom policies |
| Runtime | Kubernetes-native (Gateway API, `InferencePool`); no-Kubernetes mode exists but has no autoscaling | VMs, bare metal or Kubernetes (KubeRay) |
| Beyond online LLM serving | Online inference only | Batch (Ray Data LLM), RL, pipelines, any Python model |
| Engines and accelerators | vLLM, SGLang (+ TRT-LLM KV events); NVIDIA, AMD, TPU, Intel | vLLM first-class, SGLang early; mostly NVIDIA |
| Governance | CNCF Sandbox (Mar 2026); Red Hat, Google, IBM, NVIDIA, CoreWeave | Anyscale; PyTorch Foundation (Oct 2025) |
| Maturity risks | Fast churn (renames, config breaks), many Alpha plugins, many moving parts | KV-aware router alpha, HAProxy/direct streaming experimental, queue-only autoscaling |

## Decision guide

**Choose llm-d if most of these are true:**
- You run, or will run, inference on Kubernetes with a platform team.
- One endpoint serves many models, LoRAs or tenants, and you need priority or fairness between them.
- Traffic has heavy prefix reuse (chat, agents, RAG with shared system prompts) at many replicas.
- You plan P/D disaggregation, wide-EP MoE, KV offloading, or multi-vendor accelerators.
- You want a standard, vendor-neutral data plane (Gateway API).

**Stay on Ray Serve LLM if most of these are true:**
- Your stack is already Ray, and the LLM sits inside a larger Python application (pipelines,
  pre- and post-processing, other models).
- You also run batch inference or RL on the same cluster and want one runtime.
- You don't run Kubernetes, or the team is mostly ML engineers rather than platform engineers.
- Scale is modest (a few replicas per model), where routing sophistication matters less than simplicity.
- You need custom routing logic written in Python.

**For an existing Ray user, as here:** keep Ray where it adds value (batch, RL, pipelines).
For high-traffic online LLM endpoints, especially multi-tenant ones with shared prefixes, put
llm-d in front of vLLM. If moving is not worth it yet, first enable Ray's
`PrefixCacheAffinityRouter` and HAProxy/direct streaming. Both are low-cost changes that address
the gaps measured here.

## Next steps

1. **Finish the matrix.**
   - **llm-d load-only on `multiturn`** fits this laptop when about 5 GiB of RAM is free.
     Pause the local kind clusters first:
     ```bash
     WORKLOADS=multiturn LLMD_VARIANTS="load-only:configs/llmd/epp-config-load-only.yaml" RAY_VARIANTS="" scripts/run_matrix.sh results/run1.jsonl
     ```
   - **Ray `PrefixCacheAffinityRouter`** needs a host with 16 GiB of RAM or more, for example a
     small cloud GPU VM. Run `RAY_VARIANTS="prefix" LLMD_VARIANTS=""` there.
2. **Repeat each cell 3 times** with different seeds (`BENCH_ARGS="... --seed N"`), then
   combine the runs with `uv run bench/summarize.py results/run*.jsonl`.
3. **Optional:**
   - Ray with HAProxy and direct streaming, to separate proxy overhead from routing policy.
   - llm-d precise prefix routing (KV events).
   - `llm-d-inference-sim` at 8+ simulated replicas, to show routing at scale without GPUs.

## Sources

**Web:**
- llm-d: [releases](https://github.com/llm-d/llm-d/releases) ·
  [running without Kubernetes](https://llm-d.ai/blog/running-llm-d-without-kubernetes) ·
  [KV-cache wins](https://llm-d.ai/blog/kvcache-wins-you-can-see) ·
  [sticky-until-saturated routing](https://llm-d.ai/blog/sticky-until-saturated-token-aware-routing) ·
  [CNCF announcement](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/)
- Ray: [routing policies](https://docs.ray.io/en/master/serve/llm/architecture/routing-policies.html) ·
  [prefix routing blog](https://anyscale.com/blog/ray-serve-faster-first-token-custom-routing) ·
  [KV/token-aware routing](https://anyscale.com/blog/llm-kv-token-aware-routing) ·
  [HAProxy](https://anyscale.com/blog/ray-serve-inference-lower-latency-higher-throughput-haproxy) ·
  [direct streaming](https://docs.ray.io/en/latest/serve/llm/user-guides/direct-streaming.html) ·
  [PyTorch Foundation](https://pytorch.org/blog/pytorch-foundation-welcomes-ray-to-deliver-a-unified-open-source-ai-compute-stack/)

**Code read directly:** llm-d `guides/*` (no-kubernetes-deployment, optimized-baseline, pd-disaggregation,
wide-ep, flow-control, workload-autoscaling, rl), llm-d router `docs/operations.md`, and Ray
`ray/llm/_internal/serve/routing_policies/*`.
