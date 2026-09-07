import numpy as np

def analyze_roofline():
    print("=== Roofline Performance Analysis (AMD Ryzen 5 4500U) ===")

    cores = 6
    base_ghz = 2.38
    boost_ghz = 4.00
    peak_gflops = cores * boost_ghz * 32

    mem_bandwidth_gbs = 38.4

    ridge_point = peak_gflops / mem_bandwidth_gbs

    print(f"Theoretical Peak Compute: {peak_gflops:.1f} GFLOP/s")
    print(f"Peak Memory Bandwidth:    {mem_bandwidth_gbs:.1f} GB/s")
    print(f"Ridge Point:              {ridge_point:.2f} FLOPs/byte\n")

    print(f"{'Matrix Size (N)':<18} | {'Intensity (FLOP/byte)':<22} | {'Regime'}")
    print("-" * 65)

    for n in [32, 64, 128, 256, 512, 1024, 2048]:
        intensity = n / 6.0

        if intensity < ridge_point:
            regime = "MEMORY-BANDWIDTH-BOUND (DRAM Stalled)"
        else:
            regime = "COMPUTE-BOUND (CPU Saturated)"

        print(f"N = {n:<14} | {intensity:<22.2f} | {regime}")

if __name__ == "__main__":
    analyze_roofline()

