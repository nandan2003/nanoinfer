# nanoinfer: Bare-metal CPU LLM inference with prefix caching & OS thread pinning

Minimal CPU LLM inference server written from scratch in Python and C-bindings.
To test prefix caching, zero-allocation request queues, and hardware thread pinning on bare-metal x86 CPUs.

---

## Architecture

```text
Client (SDK / curl)
  │  HTTP/1.1 POST /v1/chat/completions
  ▼
Raw AsyncIO Socket Server (server/http_server.py)
  │  hand-parsed HTTP/1.1, zero web frameworks
  ▼
Bounded Ring Buffer (server/scheduler.py)
  │  fixed array, mutex + condition variables, HTTP 429 when full
  ▼
Pinned Worker Pool (server/scheduler.py)
  │  workers pinned via os.sched_setaffinity (serialized via engine_lock)
  ├──▶ Prefix Trie Cache (server/cache.py): O(K) lookup, O(1) LRU eviction
  └──▶ llama.cpp Engine (server/inference.py): streaming tokens + KV snapshots
```

---

## Benchmarks (AMD Ryzen 5 4500U, 6 cores)

Hardware: 6 cores @ 2.38 GHz, L1d 32 KiB, L2 512 KiB, L3 8 MiB, DDR4 dual-channel (~15.6 GB/s).  
Model: Qwen 2.5 0.5B Instruct (`q4_k_m`, 390 MB).

### Multi-Core OpenMP AVX2 (6 Cores — Memory-Bandwidth Bound)
Decoding saturates memory bandwidth ($390\text{ MB} / 15.6\text{ GB/s} \approx 25\text{ ms}$):

| Prompt | Input Tokens | Generated | TTFT | ITL Mean | TPS | E2E |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short** | 16 | 17 | 137 ms | 26.3 ms | 38.1 tok/s | 0.59 s |
| **Medium** | 64 | 17 | 158 ms | 26.9 ms | 37.2 tok/s | 0.61 s |
| **Long** | 128 | 17 | 279 ms | 26.3 ms | 38.1 tok/s | 0.74 s |

### Single-Core Isolation (`os.sched_setaffinity` Pinning — Compute/Dequant Bound)
Single APU core sequentially dequantizing 390 MB of Q4 blocks without multi-core SIMD reduction:

| Configuration | Prompt (Tokens) | TTFT | ITL Mean | TPS | E2E (17 tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pinned Core (`n_threads=1`)** | Short (16) | 245 ms | 2060 ms | 0.48 tok/s | 22.7 s |
| **Pinned Core (`n_threads=1`)** | Medium (64) | 410 ms | 2065 ms | 0.48 tok/s | 22.9 s |
| **Pinned Core (`n_threads=1`)** | Long (128) | 780 ms | 2058 ms | 0.49 tok/s | 23.2 s |
| **Pinned Core (Default OpenMP)** | Short (16) | 267 ms | 3050 ms | 0.33 tok/s | 49.1 s |

### Performance Analysis & Hardware Physics:
* **Prefill scales with prompt length:** Prompt ingestion is dense matrix multiplication (GEMM). Arithmetic intensity exceeds the DDR4 ridge point (20 FLOPs/byte), saturating CPU SIMD execution units.
* **Multi-core decode is memory-bandwidth bound (25 ms / token, ~39 tok/s):** Autoregressive token generation is memory-bandwidth bound (GEMV). Every generated token sweeps all model weights (390 MB) through memory. At dual-channel DDR4 bandwidth (~15.6 GB/s), $390\text{ MB} / 15.6\text{ GB/s} \approx 25.0\text{ ms}$ per token (~39-40 tok/s).
* **Single-core pinning bottleneck (2060 ms / token vs. 78 ms napkin math):**
  - *Napkin math check:* At a single-core DDR4 streaming speed of ~5 GB/s, reading 390 MB takes $\approx 78\text{ ms}$. Why does single-core decode take 2060–3050 ms?
  - *Profiled barrier contention:* In default mode, `llama.cpp` spawned 6 OpenMP threads that inherited the worker's single-core affinity mask. The threads oversubscribed the single core, spin-locking on OpenMP reduction barriers (`GOMP_barrier`). Clamping to `n_threads=1` eliminated barrier thrashing and cut latency from 49s to 22s (>2.3× speedup).
  - *Remaining gap:* Bounded by single-core AVX2 unpacking/dequantization of 4-bit nibbles and scales, plus single-core Line Fill Buffer limits (~5 GB/s), shifting the bottleneck from memory bus to single-thread compute.
* **Prefix Trie cache hit speedup:** When a prompt shares a prefix with an earlier request, TTFT drops from ~200-300 ms to < 2 ms (time to restore the KV snapshot).

---

## Design Choices

* **Prefix Trie Cache (`server/cache.py`):** Token IDs form a trie, with nodes linked in an LRU doubly-linked list. Lookups are $O(K)$, evictions are $O(1)$. Leaves store `llama.cpp` KV snapshots, with recursive subtree unlinking on eviction.
* **Ring Buffer Queue (`server/scheduler.py`):** Pre-allocated circular array `[None] * capacity`. Synchronized with `threading.Lock` and two `threading.Condition` variables (`not_empty`, `not_full`). No heap allocations during request handling. Drops requests with 429 when saturated.
* **Multi-Worker Core Pinning (`server/scheduler.py`):** Uses `os.sched_setaffinity` to pin worker threads to dedicated physical cores. Supports both single-engine and multi-engine pools, with per-worker locking that frees token generation from contention.
* **Raw Async Sockets (`server/http_server.py`):** `asyncio.start_server` with manual HTTP/1.1 header parsing and chunked SSE streaming. Supports `/v1/chat/completions` and `/v1/models`. No web framework dependencies.
* **Trailing Telemetry:** Emits an `event: telemetry` SSE frame right before `data: [DONE]`. Gives TTFT, ITL, TPS, and latency without extra polling.

---

## Known Limitations & Trade-offs

1. **Monolithic KV Snapshot Memory Footprint:**
   * *The Math:* Storing full `save_state()` snapshots consumes $\approx 7\text{ MB}$ per trie node. At cache capacity ($M = 500$ entries), RAM usage reaches:
     $$\text{Total Cache Memory} = 500 \times 7\text{ MB} \approx 3.5\text{ GB}$$
   * *Rationale & Solution:* Selected in v1 for exact zero-code serialization of `llama.cpp` state. In production (e.g. Kompact AI), block-level PagedAttention on CPU RAM divides context into non-contiguous 16-token pages, eliminating internal fragmentation and sharing prefix blocks across sessions.

2. **Continuous Batching vs. Worker Dispatch:**
   * *The Math:* In autoregressive decode (GEMV), sweeping 390 MB of weights at batch size $B=1$ costs:
     $$\text{Memory Transfer / Token} = \frac{390\text{ MB}}{1} = 390\text{ MB / token}$$
     With continuous batching at $B=16$, sweeping 390 MB yields 16 tokens ($\approx 24.4\text{ MB / token}$), yielding up to $16\times$ throughput efficiency.
   * *Rationale & Solution:* v1 uses worker-by-worker dispatch for deterministic per-request latency and zero scheduling overhead. Continuous batching is the next architectural milestone for multi-tenant data center servers (EBox/FinBox).

3. **Memory Bus Saturation (Single-Core vs. Multi-Core):**
   * *The Math:* The AMD Ryzen 5 4500U dual-channel DDR4 memory bus delivers $\approx 15.6\text{ GB/s}$. However, a single Zen 2 core has only 12–16 Line Fill Buffers (LFBs) to track outstanding cache misses, capping single-thread sequential bandwidth at $\approx 5\text{ GB/s}$.
   * *Rationale & Solution:* Multi-core execution (6 cores) aggregates LFBs across cores, fully saturating the 15.6 GB/s bus ($390\text{ MB} / 15.6\text{ GB/s} \approx 25\text{ ms} \to \approx 39\text{ tok/s}$). Single-core mode is kept to demonstrate hardware thread isolation and OS scheduler jitter elimination.

---

## Project Layout

```text
nanoinfer/
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI pipeline (Python 3.11)
├── models/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf # Quantized model weights (390 MB)
│
├── server/
│   ├── inference.py     # llama.cpp wrapper (streaming & state snapshots)
│   ├── cache.py         # Prefix Trie + Doubly-Linked List (O(1) LRU)
│   ├── scheduler.py     # RingBufferQueue + CPU core pinning
│   └── http_server.py   # Raw asyncio HTTP/1.1 socket server with SSE
│
├── sdk/
│   ├── __init__.py      # Exports (NanoInferClient, Telemetry)
│   └── client.py        # Python SDK (streaming iterators & telemetry)
│
├── benchmarks/
│   ├── matmul.py        # 32x32 L1 cache-blocked GEMM vs BLAS
│   ├── roofline.py      # Arithmetic intensity & DDR4 ridge point
│   ├── quantize.py      # INT8 symmetric quantization error
│   └── telemetry_suite.py # TTFT/ITL benchmark suite (multicore & pinned)
│
├── tests/               # 17 unit tests (cache, subtree pruning, queue, scheduler, server, sdk)
├── LICENSE              # MIT License
└── requirements.txt     # llama-cpp-python, numpy
```

---

## Quickstart

```bash
# 1. Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install --prefer-binary -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Download model weights (Qwen 2.5 0.5B Instruct Q4_K_M ~390 MB)
# Compatible with standard GGUF architectures (Qwen, Llama 3, Phi-3)
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# 2. Run unit tests (17 tests)
python -m unittest discover -s tests -v

# 3. Run benchmarks
python benchmarks/matmul.py
python benchmarks/roofline.py
python benchmarks/quantize.py
python benchmarks/telemetry_suite.py --mode multicore

# 4. Start server
python -m server.http_server

# 5. Client SDK example
from sdk.client import NanoInferClient

client = NanoInferClient("http://127.0.0.1:8080")
for token in client.chat_stream("In high performance computing, cache lines"):
    print(token, end="", flush=True)

print("\nMetrics:", client.last_telemetry)
```
