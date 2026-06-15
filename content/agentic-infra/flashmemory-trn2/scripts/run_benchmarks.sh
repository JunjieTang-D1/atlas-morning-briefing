#!/bin/bash
# Run full benchmark suite for HASA
# Execute on trn2.48xlarge after Phase 0-2 complete

set -e
RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== HASA Benchmark Suite ==="
echo "Results: $RESULTS_DIR"

# 1. Accuracy benchmarks
echo "[1/5] Running RULER benchmark..."
python -m benchmarks.accuracy \
    --benchmark ruler \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --method hasa \
    --seq-lengths 4096 16384 65536 131072 \
    --output "$RESULTS_DIR/ruler.json"

echo "[2/5] Running Needle-in-Haystack..."
python -m benchmarks.accuracy \
    --benchmark needle \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --method hasa \
    --seq-lengths 4096 16384 65536 131072 \
    --output "$RESULTS_DIR/needle.json"

echo "[3/5] Running LongBench-v2..."
python -m benchmarks.accuracy \
    --benchmark longbench \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --method hasa \
    --output "$RESULTS_DIR/longbench.json"

# 2. System benchmarks
echo "[4/5] Running throughput benchmark..."
python -m benchmarks.system \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --method hasa \
    --batch-sizes 32 64 128 256 \
    --seq-length 131072 \
    --output "$RESULTS_DIR/throughput.json"

# 3. Baselines
echo "[5/5] Running baselines..."
for method in full_attention streaming_llm h2o snapkv flashmemory_flat; do
    python -m benchmarks.accuracy \
        --benchmark ruler \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --method "$method" \
        --seq-lengths 4096 16384 65536 131072 \
        --output "$RESULTS_DIR/baseline_${method}.json"
done

echo "=== All benchmarks complete ==="
echo "Results in: $RESULTS_DIR"
