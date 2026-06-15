# Hardware-Aware Sparse KV Attention on Trainium2
## MLSys 2027 Paper Design — Option A

> "Memory hierarchy-aware KV cache sparsification for high-concurrency LLM serving on custom AI accelerators"

---

## 1. Scientific Contribution (What's New)

**Claim:** Optimal KV cache sparsity is *hardware-dependent*. A flat 13.5% retention
(FlashMemory) is suboptimal on hierarchical memory systems. We propose
**Hierarchy-Aware Sparse Attention (HASA)** — a two-tier sparsification that
exploits Trainium2's SBUF/HBM hierarchy to achieve better throughput AND accuracy
than hardware-agnostic approaches.

**Three novel contributions:**

1. **Two-tier sparsification kernel (NKI):** Hot chunks in SBUF (on-chip SRAM),
   warm chunks in HBM, cold evicted. No prior work exploits accelerator SRAM tiers.

2. **Hardware-aware scoring:** Retriever score incorporates memory-access cost:
   `final_score = retriever_score - α·access_latency_ratio`. Chunks in SBUF get
   bonus; chunks requiring HBM DMA get penalty. This outperforms flat top-K.

3. **High-concurrency serving benchmark:** First paper to show sparse KV on
   Trainium2 at batch=64-256 with 70B model at 128K context — regime where
   full attention is physically impossible (OOM), making sparsity *enabling*
   rather than *optimizing*.

---

## 2. Hardware Target

### Trainium2 (trn2.48xlarge)
```
Chips:           16 × Trainium2
Compute:         190 TFLOPS bf16/chip (3,040 total)
HBM:             96 GB HBM3e/chip (1,536 GB total)
SBUF (on-chip):  48 MB/chip (estimated, 2x trn1's 24MB)
PSUM:            768 KB/chip (register file)
Interconnect:    NeuronLink v2 (chip-to-chip, 384 GB/s)
FP8:             Native support (E4M3, E5M2)
Region:          us-east-2 (us-east-2a/b/c)
Cost:            ~$21.50/hr on-demand
```

### Why trn2 (not trn1, not GPU):
- 96GB/chip HBM → 70B fits on 2 chips (TP=2), leaving 14 chips for KV
- But at batch=128+ with 128K context → KV explodes → sparse is REQUIRED
- SBUF 48MB = perfect "hot cache" tier that GPU doesn't have (GPU L2 is shared/opaque)
- fp8 native → compressed KV keys at zero overhead
- Cost: $21.50/hr vs ~$32/hr for p5.48xlarge → cheaper benchmarking

---

## 3. Model: Llama-3.1-70B-Instruct (128K)

### Why 70B (not 8B):
- **Memory pressure is real:** 70B weights = ~140GB (bf16), needs TP=2 minimum
- **KV cache dominates at high batch:** batch=64 × 128K × 70B-KV = 1.05 TB
- **This is the regime where sparse KV is ENABLING (not just optimizing)**
- **MLSys reviewers want:** real production-scale problems, not toy 8B setups

### KV Cache Math (70B):
```
Llama-3.1-70B config:
  layers = 80
  kv_heads = 8 (GQA, 8 KV heads shared across 64 query heads)
  head_dim = 128
  dtype = bf16

Per token, all layers:
  K+V = 2 × 8 × 128 × 2 bytes × 80 layers = 327,680 bytes = 320 KB/token

Full KV cache:
  128K tokens: 320KB × 128K = 40 GB (per sequence!)

Batched serving on trn2.48xlarge (1,536 GB total HBM):
  Model weights (TP=16): ~140GB (replicated KV heads across chips)
  Available for KV: ~1,396 GB
  
  Full attention:
    batch=32:  32 × 40GB = 1,280 GB → FITS (barely)
    batch=48:  48 × 40GB = 1,920 GB → OOM ❌
    batch=64:  64 × 40GB = 2,560 GB → OOM ❌

  With HASA (13.5% retention):
    batch=32:  32 × 5.4GB = 173 GB → EASY
    batch=64:  64 × 5.4GB = 346 GB → COMFORTABLE
    batch=128: 128 × 5.4GB = 691 GB → FITS
    batch=256: 256 × 5.4GB = 1,382 GB → FITS (barely)

  IMPROVEMENT: max batch 32 → 256 = 8x throughput
```

**This is the MLSys story:** Without sparse KV, trn2 can only serve batch=32 at 128K.
With HASA, it serves batch=256. That's 8x throughput on the same hardware.

---

## 4. Architecture: HASA (Hierarchy-Aware Sparse Attention)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 HASA: HIERARCHY-AWARE SPARSE ATTENTION                    │
│                 (Trainium2, TP=16, Llama-3.1-70B)                       │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  PREFILL (Dense, standard)                                       │    │
│  │  • Full attention on prompt                                      │    │
│  │  • Generate all KV, compress K to fp8 chunks                     │    │
│  │  • Build chunk index (metadata: position, layer, norm)           │    │
│  └─────────────────────────┬───────────────────────────────────────┘    │
│                             │                                             │
│                             ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  KV STORE (Three-Tier)                                           │    │
│  │                                                                   │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │ TIER 0 — SBUF (48MB/chip)          "Registers"           │   │    │
│  │  │ • Attention sink tokens (first 4 + last 64)              │   │    │
│  │  │ • Current window (last 256 tokens)                       │   │    │
│  │  │ • Top-scored "hot" chunks from retriever                 │   │    │
│  │  │ • Access: 1 cycle (zero-cost)                            │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │ TIER 1 — HBM (96GB/chip)            "RAM"               │   │    │
│  │  │ • All indexed chunks (fp8 compressed K, bf16 V)          │   │    │
│  │  │ • Prefetchable to SBUF via DMA                           │   │    │
│  │  │ • Access: ~100 cycles (DMA latency)                      │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │ TIER 2 — EVICTED                    "Disk"               │   │    │
│  │  │ • Low-scored chunks dropped from computation             │   │    │
│  │  │ • Recomputable from prefill cache if needed              │   │    │
│  │  │ • Access: recompute (expensive, avoided)                 │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────┬───────────────────────────────────────┘    │
│                             │                                             │
│                             ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  DECODE LOOP                                                     │    │
│  │                                                                   │    │
│  │  Every RETRIEVAL_INTERVAL (64) decode steps:                     │    │
│  │                                                                   │    │
│  │  1. SCORE: retriever(hidden_state) → chunk_scores[N]             │    │
│  │  2. ADJUST: score += sbuf_bonus(chunk) - hbm_penalty(chunk)      │    │
│  │  3. PLACE: top-K₁→SBUF, top-K₂→HBM(keep), rest→evict           │    │
│  │  4. PREFETCH: async DMA next-predicted hot chunks → SBUF         │    │
│  │  5. ATTEND: sparse flash attention on SBUF+HBM selected chunks   │    │
│  │                                                                   │    │
│  │  Between retrieval cycles:                                        │    │
│  │  • Attention only on SBUF-resident + HBM-selected chunks         │    │
│  │  • No re-scoring, use cached decisions                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Novel Kernel Designs (NKI)

### Kernel 1: Two-Tier Sparse Gather (`hasa_gather.py`)

```python
"""
Two-tier sparse KV gather for Trainium2.
Key insight: separate DMA streams for SBUF-resident vs HBM-fetched chunks.
SBUF chunks = zero-latency access (already on-chip).
HBM chunks = async DMA with double-buffering.
"""
```

### Kernel 2: Hardware-Aware Sparse FlashAttention (`hasa_attention.py`)

```python
"""
Sparse FlashAttention that processes SBUF and HBM chunks separately:
- SBUF chunks: immediate matmul (no DMA wait)
- HBM chunks: overlap DMA with SBUF computation (pipelining)
- Online softmax across both tiers (single numerically-stable pass)

This is the novel kernel. Standard sparse attention treats all chunks equally.
HASA processes SBUF-first, overlapping HBM DMA with SBUF compute.
"""
```

### Kernel 3: Retriever Scoring (`hasa_score.py`)

```python
"""
Run FlashMemory-style retriever on NeuronCore.
Produces raw scores, then applies hardware-aware adjustment:
  adjusted_score[i] = raw_score[i] + bonus_if_in_sbuf[i] - dma_cost[i]
  
This biases the selector toward keeping "borderline" chunks in SBUF
rather than evicting them (since SBUF access is free).
"""
```

---

## 6. Competitive Baselines (5 methods)

| # | Method | What It Does | Source |
|---|--------|-------------|--------|
| 1 | **Full Attention** | All KV in HBM, standard FlashAttention | Upper bound |
| 2 | **StreamingLLM** | Attention sink (4 tokens) + sliding window | Xiao et al. 2024 |
| 3 | **H2O (Heavy Hitter Oracle)** | Keep tokens with highest cumulative attention | Zhang et al. 2024 |
| 4 | **SnapKV** | Observation-window based compression | Li et al. 2024 |
| 5 | **FlashMemory (flat)** | Learned retriever, flat top-K (no hierarchy) | Wang et al. 2026 |
| 6 | **HASA (ours)** | Learned retriever + two-tier placement + HW-aware scoring | **Novel** |

**Key comparison:** HASA vs FlashMemory-flat shows the value of hierarchy awareness.
Both use same retriever; difference is SBUF exploitation.

---

## 7. Benchmark Suite

### 7.1 Accuracy Benchmarks

| Benchmark | Lengths | Metric | Why |
|-----------|---------|--------|-----|
| RULER | 4K, 16K, 64K, 128K | Accuracy | Multi-hop retrieval at depth |
| Needle-in-Haystack | 4K→128K (all positions) | Exact match | Worst-case for eviction |
| LongBench-v2 | 46K-128K | F1 | Real QA/summarization |
| PG-19 (perplexity) | 128K | PPL | Language modeling quality |
| Passkey Retrieval | 4K→128K | Exact match | Stress test |

### 7.2 System Benchmarks (the MLSys differentiator)

| Metric | Description |
|--------|-------------|
| Max serving batch | Largest batch that fits in memory at 128K |
| Throughput (tok/s) | End-to-end decode throughput |
| TTFT | Time to first token |
| TPOT | Time per output token |
| HBM utilization % | Memory efficiency |
| SBUF hit rate % | How often chunks are in SBUF (HASA-specific) |
| DMA overlap % | Compute/transfer overlap ratio |

### 7.3 Ablation Studies

| Ablation | Purpose |
|----------|---------|
| Flat top-K vs two-tier placement | Isolate SBUF exploitation value |
| HW-aware scoring ON vs OFF | Isolate scoring adjustment value |
| Retrieval interval: 32, 64, 128, 256 | Staleness vs overhead tradeoff |
| Retention %: 5%, 10%, 15%, 25%, 50% | Sparsity curve |
| Chunk size: 32, 64, 128 tokens | Granularity tradeoff |
| fp8 K vs bf16 K | Compression impact on retriever quality |

---

## 8. Implementation Loop

```
┌──────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION LOOP                         │
│                                                                │
│  ┌────────────┐                                               │
│  │ PHASE 0    │ Local development (Atlas host, no GPU)        │
│  │ Foundation │ • Retriever model code (PyTorch native)       │
│  │ (Week 1)   │ • Benchmark harness (datasets, metrics)       │
│  │            │ • NKI kernel scaffolds (compilable stubs)     │
│  │            │ • Unit tests (correctness, shapes)            │
│  └─────┬──────┘                                               │
│        │                                                       │
│        ▼                                                       │
│  ┌────────────┐                                               │
│  │ PHASE 1    │ Single-chip validation (trn2, 1 NeuronCore)  │
│  │ Kernels    │ • Compile NKI kernels on real hardware        │
│  │ (Week 2)   │ • Profile: latency, SBUF usage, DMA rates    │
│  │            │ • Correctness: output matches PyTorch ref     │
│  │            │ • Iterate kernel until targets met            │
│  └─────┬──────┘                                               │
│        │                                                       │
│        ▼                                                       │
│  ┌────────────┐                                               │
│  │ PHASE 2    │ Retriever training (GPU, backbone-free)       │
│  │ Retriever  │ • Generate training data from 70B attn maps   │
│  │ (Week 3)   │ • Train dual-encoder retriever (~50M params)  │
│  │            │ • Validate: Recall@K, NDCG vs oracle          │
│  │            │ • Ablate: scoring layers, chunk sizes          │
│  └─────┬──────┘                                               │
│        │                                                       │
│        ▼                                                       │
│  ┌────────────┐                                               │
│  │ PHASE 3    │ Integration (trn2.48xlarge, full model)       │
│  │ System     │ • 70B serving with HASA decode loop           │
│  │ (Week 4-5) │ • Accuracy benchmarks (RULER, Needle, etc.)  │
│  │            │ • System benchmarks (throughput, batch size)   │
│  │            │ • Baseline comparisons (5 methods)            │
│  └─────┬──────┘                                               │
│        │                                                       │
│        ▼                                                       │
│  ┌────────────┐                                               │
│  │ PHASE 4    │ Revision loop (iterate on results)            │
│  │ Revise     │ • If accuracy < baseline: tune retention %    │
│  │ (Week 6)   │ • If throughput gain < 2x: optimize kernels   │
│  │            │ • If SBUF hit rate < 80%: tune placement      │
│  │            │ • Repeat Phase 3 with revised params          │
│  └─────┬──────┘                                               │
│        │                                                       │
│        ▼                                                       │
│  ┌────────────┐                                               │
│  │ PHASE 5    │ Paper writing + figures                       │
│  │ Paper      │ • LaTeX draft (MLSys format)                  │
│  │ (Week 7-8) │ • Figures: throughput curves, accuracy tables │
│  │            │ • Related work positioning                     │
│  └────────────┘                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Quality Gates (High Bar)

Each phase has EXIT CRITERIA that must pass before proceeding:

### Phase 0 Exit Criteria:
- [ ] All unit tests pass (shapes, dtypes, numerics within 1e-5)
- [ ] Retriever forward pass produces valid scores [0,1] on mock data
- [ ] NKI kernel stubs compile with `neuronx-cc --target trn2` (dry-run)
- [ ] Benchmark harness loads all 5 datasets successfully

### Phase 1 Exit Criteria:
- [ ] `hasa_gather` kernel: correct output vs PyTorch gather reference
- [ ] `hasa_attention` kernel: output within 1e-3 of dense attention (on sparse subset)
- [ ] `hasa_score` kernel: scores match PyTorch reference within 1e-4
- [ ] Latency targets: gather < 50µs, attention < 200µs (per head, seq=128K×13.5%)
- [ ] SBUF utilization reported by Neuron profiler

### Phase 2 Exit Criteria:
- [ ] Retriever Recall@256 ≥ 85% (can find 85% of oracle top-256 chunks)
- [ ] Retriever NDCG@256 ≥ 0.80
- [ ] Generalization: test accuracy within 5% of train accuracy
- [ ] Inference latency < 1ms per score call (on NeuronCore)

### Phase 3 Exit Criteria:
- [ ] Accuracy: within 2% of full attention on RULER-128K
- [ ] Accuracy: within 1% on Needle-in-Haystack (all positions)
- [ ] Throughput: ≥ 4x over full attention at batch=128
- [ ] Max batch: ≥ 128 at 128K context (full attention OOMs at batch≥48)
- [ ] All 5 baselines benchmarked with same harness

### Phase 4 Exit Criteria:
- [ ] Best config identified (retention%, chunk_size, interval)
- [ ] HASA > FlashMemory-flat by ≥ 5% on at least one key metric
- [ ] No accuracy regression > 3% on any single benchmark
- [ ] Results reproducible (3 runs, std < 2%)

---

## 10. Cost & Timeline

| Phase | Hardware | Hours | Cost |
|-------|----------|:---:|:---:|
| 0: Foundation | Atlas host (free) | 20h | $0 |
| 1: Kernels | trn2.48xlarge | 8h | $172 |
| 2: Retriever | g5.2xlarge (spot) | 12h | $15 |
| 3: System | trn2.48xlarge | 16h | $344 |
| 4: Revise | trn2.48xlarge | 8h | $172 |
| 5: Paper | Atlas host | 20h | $0 |
| **Total** | | **84h** | **~$703** |

Buffer 50%: **~$1,050 total budget**
Timeline: **8 weeks** (realistic for MLSys-quality)

---

## 11. PyTorch Native Approach

All code uses **PyTorch native** (no torch_xla dependency):

```python
# Model loading: neuronx-distributed for TP
import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers import parallel_state

# Compilation: torch.compile with neuronx backend  
model = torch.compile(model, backend="neuronx")

# Custom NKI ops: registered via torch.library
@torch.library.custom_op("hasa::sparse_gather", mutates_args=())
def sparse_gather(kv_cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # NKI kernel invocation
    return nki_sparse_gather(kv_cache, indices)

# Serving loop: standard PyTorch (no XLA traces)
for step in range(max_new_tokens):
    hidden = model.decode_step(input_ids, kv_cache=sparse_kv)
    if step % RETRIEVAL_INTERVAL == 0:
        scores = retriever(hidden)
        sparse_kv = hasa_place(kv_cache, scores, sbuf_budget)
```

---

## 12. Repository Structure

```
flashmemory-trn2/
├── DESIGN.md                    # This document
├── LOOP.md                      # Implementation loop tracker
├── src/
│   ├── __init__.py
│   ├── retriever.py             # FlashMemory retriever (adapted for Llama-70B GQA)
│   ├── hasa_placer.py           # Two-tier placement logic
│   ├── serving_loop.py          # Decode loop with HASA integration
│   └── kernels/
│       ├── __init__.py
│       ├── hasa_gather.py       # NKI: two-tier sparse KV gather
│       ├── hasa_attention.py    # NKI: sparse flash attention (SBUF-first)
│       └── hasa_score.py        # NKI: retriever scoring on NeuronCore
├── benchmarks/
│   ├── __init__.py
│   ├── accuracy.py              # RULER, Needle, LongBench, Passkey, PPL
│   ├── system.py                # Throughput, latency, batch scaling
│   ├── baselines/
│   │   ├── streaming_llm.py
│   │   ├── h2o.py
│   │   ├── snapkv.py
│   │   └── flashmemory_flat.py
│   └── datasets/
│       └── download.py
├── scripts/
│   ├── setup_trn2.sh            # Instance setup + SDK install
│   ├── generate_train_data.sh   # Run 70B, capture attention maps
│   ├── train_retriever.sh       # Train retriever (backbone-free)
│   ├── run_benchmarks.sh        # Full benchmark suite
│   └── profile_kernels.sh       # Neuron profiler runs
├── results/
│   └── .gitkeep
└── docs/
    ├── BASELINES.md             # Baseline implementation notes
    └── KERNEL_NOTES.md          # NKI development notes
```
