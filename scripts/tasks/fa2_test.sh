#!/bin/bash
set -euo pipefail

# =============================================================================
# Task: Test FA2 NKI Attention Kernel on trn1.2xlarge
# =============================================================================
# This script:
# 1. Sets up the Python environment
# 2. Runs tile size unit tests (no NeuronCore needed)
# 3. Compiles the FA2 kernel via NeuronCC
# 4. Runs correctness tests against PyTorch reference
# 5. Runs performance benchmarks at 1K/2K/4K/8K/16K
# =============================================================================

echo "=========================================="
echo "FA2 NKI Kernel Test — $(date -u)"
echo "=========================================="

# --- Environment ---
cd /home/ubuntu
export NEURON_RT_VISIBLE_CORES=0,1
mkdir -p /tmp/results

echo "[1/6] Checking NeuronCore availability..."
neuron-ls || { echo "FAIL: neuron-ls not found"; exit 1; }

echo "[2/6] Installing dependencies..."
pip install -q torch-neuronx neuronx-cc pytest 2>/dev/null || true

echo "[3/6] Unpacking attention-forge code..."
# The tarball is SCP'd alongside this script
if [ -f /tmp/attention-forge.tar.gz ]; then
    cd /home/ubuntu && tar xzf /tmp/attention-forge.tar.gz
else
    echo "FAIL: /tmp/attention-forge.tar.gz not found"
    exit 1
fi
cd /home/ubuntu/attention-forge

echo "[4/6] Running tile size unit tests..."
python3 -c "
from src.kernels.fa2_fwd import select_tile_sizes
# Test basic tile selection
assert select_tile_sizes(1024, 128) == (16, 16), f'Got {select_tile_sizes(1024, 128)}'
assert select_tile_sizes(4096, 128) == (16, 16), f'Got {select_tile_sizes(4096, 128)}'
assert select_tile_sizes(16384, 128) == (8, 16), f'Got {select_tile_sizes(16384, 128)}'
assert select_tile_sizes(1024, 64) == (32, 32), f'Got {select_tile_sizes(1024, 64)}'
print('PASS: All tile size tests passed')
"

echo "[5/6] Compile smoke test..."
python3 -c "
import torch
import torch_neuronx
import neuronxcc.nki as nki
from src.kernels.fa2_fwd import fa2_attention_fwd

# Small test: batch=1, heads=1, seqlen=128, d=128 (minimal)
B, H, N, D = 1, 1, 128, 128
Q = torch.randn(B, H, N, D, dtype=torch.bfloat16)
K = torch.randn(B, H, N, D, dtype=torch.bfloat16)
V = torch.randn(B, H, N, D, dtype=torch.bfloat16)
O = torch.zeros(B, H, N, D, dtype=torch.bfloat16)
L = torch.zeros(B, H, N, dtype=torch.float32)

print(f'Compiling FA2 kernel for N={N}, D={D}...')
try:
    fa2_attention_fwd(Q, K, V, O, L)
    print(f'PASS: Compilation + execution succeeded')
    print(f'Output shape: {O.shape}, L shape: {L.shape}')
    print(f'O range: [{O.min():.4f}, {O.max():.4f}]')
    print(f'L range: [{L.min():.4f}, {L.max():.4f}]')
except Exception as e:
    print(f'FAIL: {type(e).__name__}: {e}')
    import traceback; traceback.print_exc()
"

echo "[6/6] Correctness + benchmark at multiple sequence lengths..."
python3 << 'PYEOF'
import torch
import torch_neuronx
import neuronxcc.nki as nki
import time
import json
from src.kernels.fa2_fwd import fa2_attention_fwd

results = {}
SEQLENS = [512, 1024, 2048, 4096, 8192, 16384]
B, H, D = 1, 8, 128
WARMUP = 20
TIMED = 100

for N in SEQLENS:
    print(f"\n--- seqlen={N} ---")
    Q = torch.randn(B, H, N, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, N, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, N, D, dtype=torch.bfloat16)
    O = torch.zeros(B, H, N, D, dtype=torch.bfloat16)
    L = torch.zeros(B, H, N, dtype=torch.float32)

    # --- Correctness ---
    try:
        fa2_attention_fwd(Q, K, V, O, L)
        
        # PyTorch reference
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float()
        ).bfloat16()
        
        # Check cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            O.flatten().float(), ref.flatten().float(), dim=0
        ).item()
        
        # Max absolute error
        max_err = (O.float() - ref.float()).abs().max().item()
        
        correct = cos_sim >= 0.99
        print(f"  Correctness: cos_sim={cos_sim:.6f}, max_err={max_err:.6f} -> {'PASS' if correct else 'FAIL'}")
    except Exception as e:
        print(f"  Correctness: FAIL ({type(e).__name__}: {e})")
        correct = False
        cos_sim = 0
        max_err = -1

    # --- Performance ---
    latencies = []
    try:
        # Warmup
        for _ in range(WARMUP):
            O_bench = torch.zeros(B, H, N, D, dtype=torch.bfloat16)
            L_bench = torch.zeros(B, H, N, dtype=torch.float32)
            fa2_attention_fwd(Q, K, V, O_bench, L_bench)
        
        # Timed runs
        for _ in range(TIMED):
            O_bench = torch.zeros(B, H, N, D, dtype=torch.bfloat16)
            L_bench = torch.zeros(B, H, N, dtype=torch.float32)
            start = time.perf_counter()
            fa2_attention_fwd(Q, K, V, O_bench, L_bench)
            end = time.perf_counter()
            latencies.append((end - start) * 1e6)  # microseconds
        
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p90 = latencies[int(len(latencies)*0.9)]
        p99 = latencies[int(len(latencies)*0.99)]
        print(f"  Latency: p50={p50:.0f}µs  p90={p90:.0f}µs  p99={p99:.0f}µs")
    except Exception as e:
        print(f"  Performance: SKIP ({type(e).__name__}: {e})")
        p50 = p90 = p99 = -1

    results[str(N)] = {
        "correct": correct,
        "cos_sim": cos_sim,
        "max_err": max_err,
        "p50_us": p50,
        "p90_us": p90,
        "p99_us": p99,
    }

# Save results
with open("/tmp/results/fa2_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n========== SUMMARY ==========")
print(f"{'SeqLen':>8} {'Correct':>8} {'CosSim':>8} {'p50(µs)':>10} {'p90(µs)':>10}")
for N in SEQLENS:
    r = results[str(N)]
    print(f"{N:>8} {'PASS' if r['correct'] else 'FAIL':>8} {r['cos_sim']:>8.4f} {r['p50_us']:>10.0f} {r['p90_us']:>10.0f}")

# Baselines for comparison
print("\n========== vs BASELINES ==========")
baselines_v7 = {1024: 39, 2048: 112, 4096: 400, 8192: 1526, 16384: 6227}
baselines_v11 = {1024: 37, 2048: 98, 4096: 361, 8192: 1387, 16384: 35078}
for N in [1024, 2048, 4096, 8192, 16384]:
    r = results.get(str(N), {})
    p50 = r.get('p50_us', -1)
    if p50 > 0:
        vs_v7 = baselines_v7[N] / p50
        vs_v11 = baselines_v11[N] / p50
        print(f"  N={N:>5}: FA2={p50:.0f}µs  v7={baselines_v7[N]}µs ({vs_v7:.2f}x)  v11={baselines_v11[N]}µs ({vs_v11:.2f}x)")

print("\nDone. Results saved to /home/ubuntu/task/fa2_benchmark_results.json")
PYEOF

echo "=========================================="
echo "Task complete — $(date -u)"
echo "=========================================="
