import time
import numpy as np

def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Triple-nested loop matrix multiply (O(N^3)). Memory-unfriendly."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            total = 0.0
            for k in range(K):
                total += A[i, k] * B[k, j]
            C[i, j] = total
    return C

def matmul_blocked(A: np.ndarray, B: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Cache-tiled / blocked GEMM.
    Partitions matrices into 32x32 tiles (~4 KB each) that fit into CPU L1 cache (32 KB).
    """
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)

    for i0 in range(0, M, block_size):
        i_max = min(i0 + block_size, M)
        for j0 in range(0, N, block_size):
            j_max = min(j0 + block_size, N)
            for k0 in range(0, K, block_size):
                k_max = min(k0 + block_size, K)

                for i in range(i0, i_max):
                    for k in range(k0, k_max):
                        a_ik = A[i, k]
                        for j in range(j0, j_max):
                            C[i, j] += a_ik * B[k, j]
    return C

def benchmark(N: int = 128):
    print(f"=== Matrix Multiplication Benchmark (Size: {N}x{N}) ===")
    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    t0 = time.perf_counter()
    C_numpy = np.dot(A, B)
    t_numpy = time.perf_counter() - t0
    flops = 2 * (N ** 3)
    gflops_numpy = (flops / t_numpy) / 1e9
    print(f"1. NumPy (BLAS C-kernel): {t_numpy:.4f}s | {gflops_numpy:.2f} GFLOP/s")

    t0 = time.perf_counter()
    C_blocked = matmul_blocked(A, B, block_size=32)
    t_blocked = time.perf_counter() - t0
    gflops_blocked = (flops / t_blocked) / 1e9
    np.testing.assert_allclose(C_blocked, C_numpy, rtol=1e-3, atol=1e-3)
    print(f"2. Blocked (32x32 L1 tiles): {t_blocked:.4f}s | {gflops_blocked:.4f} GFLOP/s")

    t0 = time.perf_counter()
    C_naive = matmul_naive(A, B)
    t_naive = time.perf_counter() - t0
    gflops_naive = (flops / t_naive) / 1e9
    np.testing.assert_allclose(C_naive, C_numpy, rtol=1e-3, atol=1e-3)
    print(f"3. Naive (Triple-loop):   {t_naive:.4f}s | {gflops_naive:.4f} GFLOP/s")

    speedup = t_naive / t_blocked
    print(f"\n[Result] Cache blocking achieved {speedup:.2f}x speedup over naive triple-loop!")

if __name__ == "__main__":
    benchmark(N=128)

