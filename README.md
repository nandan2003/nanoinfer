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
  │  workers pinned to physical cores via os.sched_setaffinity
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

| Prompt | Input Tokens | Generated | TTFT | ITL Mean | TPS | E2E |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short** | 16 | 17 | 267 ms | 3050 ms | 0.33 tok/s | 49.1 s |
| **Medium** | 64 | 17 | 455 ms | 3051 ms | 0.33 tok/s | 49.3 s |
| **Long** | 128 | 17 | 835 ms | 3044 ms | 0.33 tok/s | 49.6 s |

### Performance Analysis & Hardware Physics:
* **Prefill scales with prompt length:** Prompt ingestion is dense matrix multiplication (GEMM). Arithmetic intensity exceeds the DDR4 ridge point (20 FLOPs/byte), saturating CPU SIMD execution units.
* **Multi-core decode is memory-bandwidth bound (26 ms / token, ~38 tok/s):** Autoregressive token generation is memory-bandwidth bound (GEMV). Every generated token sweeps all model weights (390 MB) through memory. At dual-channel DDR4 bandwidth (~15.6 GB/s), $390\text{ MB} / 15.6\text{ GB/s} \approx 25\text{ ms}$ per token (~38-40 tok/s).
* **Single-core pinning bottleneck (3050 ms / token, 0.33 tok/s):** Pinning a worker thread via `os.sched_setaffinity` isolates execution to 1 physical core, restricting OpenMP to that single core. A single 15W Zen 2 core cannot dequantize 390 MB of Q4 weights fast enough to saturate memory channels, shifting the bottleneck from memory bandwidth to single-thread compute.
* **Prefix Trie cache hit speedup:** When a prompt shares a prefix with an earlier request, TTFT drops from ~200-300 ms to < 2 ms (time to restore the KV snapshot).

---

## Design Choices

* **Prefix Trie Cache (`server/cache.py`):** Token IDs form a trie, with nodes linked in an LRU doubly-linked list. Lookups are $O(K)$, evictions are $O(1)$. Leaves store `llama.cpp` KV snapshots.
* **Ring Buffer Queue (`server/scheduler.py`):** Pre-allocated circular array `[None] * capacity`. Synchronized with `threading.Lock` and two `threading.Condition` variables (`not_empty`, `not_full`). No heap allocations during request handling. Drops requests with 429 when saturated.
* **CPU Core Pinning (`server/scheduler.py`):** Uses `os.sched_setaffinity` to lock worker threads to dedicated physical cores. Prevents the OS scheduler from bouncing threads between cores and trashing L1/L2 caches.
* **Raw Async Sockets (`server/http_server.py`):** `asyncio.start_server` with manual HTTP/1.1 header parsing and chunked SSE streaming. No web framework dependencies.
* **Trailing Telemetry:** Emits an `event: telemetry` SSE frame right before `data: [DONE]`. Gives TTFT, ITL, TPS, and latency without extra polling.

---

## Known Limitations & Trade-offs

* **Single-context serialization:** Workers share one `llama_context`. Model calls are serialized behind an engine lock. To scale throughput across cores, each worker needs its own context handle.
* **KV snapshot memory:** Storing full `save_state()` snapshots takes 5–10 MB per entry. At 500 entries, that's ~3–5 GB of RAM. A production engine needs block-based paged memory (PagedAttention) instead of monolithic snapshots.
* **Trie orphan leak:** Pruning an intermediate trie node unlinks it from its parent, but descendants stay in the LRU list until evicted. Needs reference-counted subtree eviction.

---

## Project Layout

```text
nanoinfer/
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI matrix (Python 3.10, 3.11)
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
├── tests/               # 14 unit tests (cache, queue, scheduler, server, sdk)
├── LICENSE              # MIT License
└── requirements.txt     # llama-cpp-python, numpy
```

---

## Quickstart

```bash
# 1. Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download model weights (Qwen 2.5 0.5B Instruct Q4_K_M ~390 MB)
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# 2. Run unit tests (14 tests)
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
