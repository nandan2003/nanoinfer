import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import asyncio
import threading
import time
from server.http_server import HTTPServer
from sdk.client import NanoInferClient

BENCHMARK_PORT = 8095

PROMPTS = [
    ("Short (16 tokens)", "Explain what a CPU cache is in one concise sentence:"),
    ("Medium (64 tokens)", "In high performance computing, CPU cache lines and memory bandwidth determine the throughput of autoregressive language model decoding. Explain why:"),
    (
        "Long (128 tokens)",
        (
            "Operating system schedulers routinely migrate threads across different physical CPU cores "
            "to balance workload. However, in low latency artificial intelligence inference engines, "
            "thread migration causes severe performance degradation due to L1 and L2 cache invalidation. "
            "Describe how thread affinity mitigates this problem:"
        ),
    ),
]

def benchmark_server(pin_cores: bool, port: int, max_tokens: int = 16):
    mode_name = "Core-Pinned Single Core (os.sched_setaffinity)" if pin_cores else "Multi-Core OpenMP (6 Cores AVX2)"
    print(f"\n[+] Starting server in mode: {mode_name} on port {port}...")

    server = HTTPServer(host="127.0.0.1", port=port, pin_cores=pin_cores)
    loop = asyncio.new_event_loop()

    def server_thread():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        loop.run_forever()

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()
    time.sleep(1.0)

    client = NanoInferClient(base_url=f"http://127.0.0.1:{port}")

    # Warmup
    print("  [-] Warming up engine & JIT...")
    client.chat("Warmup", max_tokens=2)

    results = []
    for label, prompt in PROMPTS:
        print(f"  [-] Profiling prompt: {label}...")
        client.chat(prompt, max_tokens=max_tokens, temperature=0.1)
        m = client.last_telemetry
        if m:
            results.append({
                "label": label,
                "ttft_ms": m.ttft_ms,
                "itl_ms": m.itl_mean_ms,
                "tps": m.tps,
                "e2e_s": m.e2e_sec,
                "tokens": m.token_count,
            })

    # Cleanup
    loop.call_soon_threadsafe(loop.stop)
    time.sleep(0.5)

    return results

def print_table(title: str, results: list):
    print("\n" + "=" * 80)
    print(f"### {title}")
    print("=" * 80)
    header = f"| {'Prompt Category':<22} | {'Tokens':<6} | {'TTFT (ms)':<10} | {'ITL (ms)':<10} | {'TPS':<8} | {'E2E (s)':<8} |"
    divider = f"|{'-'*24}|{'-'*8}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*10}|"
    print(header)
    print(divider)
    for r in results:
        row = f"| {r['label']:<22} | {r['tokens']:<6} | {r['ttft_ms']:<10.2f} | {r['itl_ms']:<10.2f} | {r['tps']:<8.2f} | {r['e2e_s']:<8.4f} |"
        print(row)
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="nanoinfer Performance Telemetry Suite")
    parser.add_argument("--mode", choices=["multicore", "pinned", "all"], default="all",
                        help="Execution mode: multicore (OpenMP 6 cores), pinned (single-core isolation), or all")
    args = parser.parse_args()

    print("=" * 80)
    print(" nanoinfer: Empirical Telemetry Benchmark Suite (AMD Ryzen 5 4500U)")
    print("=" * 80)

    if args.mode in ("multicore", "all"):
        results_mc = benchmark_server(pin_cores=False, port=BENCHMARK_PORT)
        print_table("Multi-Core OpenMP AVX2 (6 Cores — Memory-Bandwidth Bound)", results_mc)

    if args.mode in ("pinned", "all"):
        tokens = 8 if args.mode == "all" else 16
        results_pin = benchmark_server(pin_cores=True, port=BENCHMARK_PORT + 1, max_tokens=tokens)
        print_table("Core-Pinned Isolation (Single Core — Compute/Dequant Bound)", results_pin)

if __name__ == "__main__":
    main()
