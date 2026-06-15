# FlashMemory on Trainium — Specific Implementation Design

> Sparse KV Attention for Long-Context Serving on AWS Trainium

---

## 1. Goal

Port FlashMemory's Lookahead Sparse Attention (LSA) approach to **Trainium**,
benchmark against full-attention baseline, and demonstrate that agent-driven
sparse KV management enables long-context serving on memory-constrained hardware.

**Success criteria:**
- KV cache reduced to ≤15% of full baseline (matching FlashMemory's 13.5%)
- Accuracy within ±1% of full attention on RULER/LongBench
- Latency per token ≤ full attention (sparse should be faster)
- Running on trn1.32xlarge (512GB HBM total, 16 NeuronCores)

---

## 2. Model Selection: **Llama-3.1-8B-128K**

### Why Llama-3.1-8B:

| Criteria | Llama-3.1-8B | DeepSeek-V4 (671B) | Qwen3-8B |
|----------|:---:|:---:|:---:|
| Fits on trn1.32xlarge | ✅ easily | ❌ too large | ✅ |
| Long context native | ✅ 128K | ✅ 128K+ | ✅ 128K |
| NeuronX SDK support | ✅ verified | ❌ MoE not fully supported | ⚠️ partial |
| Open weights | ✅ Meta license | ⚠️ custom license | ✅ Apache |
| Community baseline data | ✅ extensive | ✅ | ✅ |
| GQA (grouped query attention) | ✅ 8 KV heads | MLA (different) | ✅ 8 KV heads |
| Benchmark availability | ✅ RULER, LongBench, Needle | ✅ | ✅ |

**Decision:** Llama-3.1-8B-Instruct (128K context)
- 8B params = fits entirely on trn1.32xlarge with room for KV cache experiments
- GQA with 8 KV heads = manageable KV cache size for experimentation
- 128K native context = no need for positional extrapolation
- Excellent NeuronX transformers support (inference compilation verified)
- Most widely benchmarked = easiest to compare results

### KV Cache Math (Llama-3.1-8B):

```
Config:
  layers = 32
  kv_heads = 8 (GQA)
  head_dim = 128
  dtype = bf16 (2 bytes)

Per token KV:
  K: 8 heads × 128 dim × 2 bytes = 2,048 bytes
  V: 8 heads × 128 dim × 2 bytes = 2,048 bytes
  Total per token per layer: 4,096 bytes
  Total per token all layers: 4,096 × 32 = 131,072 bytes = 128 KB

Full KV cache at various sequence lengths:
  16K tokens:  128KB × 16K = 2 GB
  32K tokens:  128KB × 32K = 4 GB
  64K tokens:  128KB × 64K = 8 GB
  128K tokens: 128KB × 128K = 16 GB

Trainium trn1.32xlarge:
  Total HBM: 512GB (16 chips × 32GB)
  Model weights: ~16GB (8B × bf16)
  Available for KV: ~496GB across chips (tensor parallel)
  Per chip (TP=16): ~31GB available for KV
  → 128K full KV = 1GB per chip (fits easily)
  → BUT: batched serving (batch=32) = 32GB per chip → TIGHT

With FlashMemory (13.5% retention):
  128K × batch=32: 32GB → 4.3GB per chip → COMFORTABLE
  128K × batch=128: 128GB → 17.3GB per chip → NOW POSSIBLE
```

**Key insight:** FlashMemory doesn't just save memory — it enables **4x larger batch sizes**
on Trainium, directly translating to 4x higher throughput.

---

## 3. Architecture: FlashMemory Indexer on Trainium

### 3.1 Overall System

```
┌────────────────────────────────────────────────────────────────────┐
│                    TRAINIUM SERVING SYSTEM                           │
│                    (trn1.32xlarge, TP=16)                           │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 PREFILL PHASE (Dense)                          │  │
│  │  Full attention on prompt → generates all KV cache + hidden    │  │
│  │  Compress K: fp8 quantize + store chunk metadata               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 KV CACHE STORE                                 │  │
│  │                                                                │  │
│  │  Hot tier (SBUF, 24MB/chip):     Top-scored chunks (live)     │  │
│  │  Warm tier (HBM, 32GB/chip):     Indexed chunks (prefetchable)│  │
│  │  Cold tier (CPU DRAM, offload):  Evicted chunks (recomputable) │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │          DECODE LOOP (Sparse, every 64 tokens)                 │  │
│  │                                                                │  │
│  │  ┌─────────────────┐     ┌──────────────────────────────┐    │  │
│  │  │ RETRIEVER       │     │ SPARSE ATTENTION KERNEL       │    │  │
│  │  │ (FlashMemory    │     │ (NKI-native)                  │    │  │
│  │  │  Indexer)       │     │                                │    │  │
│  │  │                 │ ──► │ Only attend to selected chunks  │    │  │
│  │  │ Input: hidden   │     │ Mask rest to -inf              │    │  │
│  │  │ Output: top-K   │     │ Standard softmax on subset     │    │  │
│  │  │   chunk indices │     │                                │    │  │
│  │  └─────────────────┘     └──────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Retriever Adaptation for Llama-3.1

FlashMemory's retriever is designed for DeepSeek-V4's CSA (Compressed Sparse Attention).
We adapt for Llama-3.1's GQA:

**Original (DeepSeek-V4 CSA):**
- 128 heads, 128 dim, Q_LORA_RANK=2048
- Compressed K as fp8 (132 bytes per chunk)
- Per-layer scoring on layers 10, 12, 20

**Adapted (Llama-3.1 GQA):**
- 8 KV heads, 128 dim
- Compressed K as fp8 (132 bytes per chunk, same format)
- Per-layer scoring on layers 8, 16, 24 (spread across 32 layers)
- Simpler architecture: 8 heads instead of 128 → smaller indexer

```python
# Adapted retriever for Llama-3.1 GQA
class LlamaFlashMemoryRetriever:
    """
    Lightweight retriever for Llama-3.1 GQA KV-cache sparsification.
    
    Architecture (per scoring layer):
      hidden [B, 4096]
        → wq_a (4096 → 1024)        # smaller lora rank for 8-head GQA
        → RMSNorm
        → wq_b (1024 → 8 * 128)     # 8 KV heads × 128 dim
        → RoPE (last 64 dims)
        → q [B, 8, 128]
      
      compressed_k [B, N_chunks, 132] (uint8 fp8)
        → dequant → k [B, N_chunks, 128]
      
      score = sigmoid(mean_heads(relu(k @ q^T)))  # [B, N_chunks]
    """
    
    config = {
        "n_kv_heads": 8,
        "head_dim": 128,
        "q_lora_rank": 1024,       # smaller than DSV4's 2048
        "rope_dim": 64,
        "rope_base": 500000,       # Llama-3.1 RoPE base
        "scoring_layers": [8, 16, 24],
        "chunk_size": 64,          # tokens per chunk
        "retrieval_interval": 64,  # re-score every 64 decode steps
    }
```

### 3.3 NKI Kernel Design

Two custom NKI kernels needed:

**Kernel 1: Sparse KV Gather**
```python
@nki.jit
def sparse_kv_gather(
    kv_cache,       # [num_chunks, chunk_size, kv_heads, head_dim] in HBM
    keep_indices,   # [top_k] selected chunk indices
    output_buf,     # [top_k, chunk_size, kv_heads, head_dim] in SBUF
):
    """
    Gather only selected KV chunks from HBM into SBUF.
    This is the memory-saving kernel: only loads 13.5% of KV.
    
    Key optimization: DMA prefetch next chunk while processing current.
    """
    for i in nl.affine_range(top_k):
        chunk_idx = nl.load(keep_indices[i])
        # Async DMA: load from HBM to SBUF
        chunk_data = nl.load(kv_cache[chunk_idx])
        nl.store(output_buf[i], chunk_data)
```

**Kernel 2: Sparse Attention with Mask**
```python
@nki.jit  
def sparse_flash_attention(
    query,          # [batch, heads, head_dim]
    sparse_k,       # [top_k * chunk_size, kv_heads, head_dim] (gathered)
    sparse_v,       # [top_k * chunk_size, kv_heads, head_dim] (gathered)
):
    """
    FlashAttention on sparse KV subset.
    Same tiling as our proven attention kernel, but on smaller KV set.
    
    At 13.5% retention with 128K context:
      Full attention: 128K tokens
      Sparse attention: ~17K tokens → fits in SBUF easily
    """
    # Standard flash attention tiling on the sparse subset
    # Reuse NKI-Agent v12 attention kernel architecture
    # But input is only the selected chunks (much smaller)
    ...
```

**Kernel 3: Retriever Scoring (inference of indexer on NeuronCore)**
```python
@nki.jit
def retriever_score(
    hidden,         # [batch, 4096] current decode hidden state
    wq_a,          # [4096, 1024] projection
    wq_b,          # [1024, 1024] (8*128) projection  
    q_norm,        # [1024] RMSNorm weights
    compressed_k,  # [num_chunks, 132] uint8 fp8
):
    """
    Run FlashMemory retriever entirely on NeuronCore.
    Small enough to run without separate model load.
    
    Retriever params: ~25M (tiny vs 8B backbone)
    Can share NeuronCore with backbone decode step.
    """
    # Project query
    q_lora = nl.matmul(hidden, wq_a)        # [B, 1024]
    q_norm_out = rms_norm(q_lora, q_norm)
    q = nl.matmul(q_norm_out, wq_b)         # [B, 1024] = [B, 8*128]
    q = reshape_and_rope(q)                   # [B, 8, 128]
    
    # Dequant compressed keys
    k = dequant_fp8(compressed_k)             # [N, 128]
    
    # Score
    scores = nl.matmul(k, q.transpose())      # [N, 8]
    scores = nl.relu(scores)
    scores = scores.mean(dim=-1)              # [N]
    scores = nl.sigmoid(scores)               # [N] in [0,1]
    
    return scores
```

---

## 4. Training the Retriever (for Llama-3.1)

FlashMemory's key insight: **backbone-free decoupled training**.
We don't need to load Llama-3.1-8B to train the retriever.

### Training Data Generation (one-time, on GPU):

```
Step 1: Run Llama-3.1-8B on long-context datasets (LongBench, PG19, etc.)
Step 2: For each decode step, record:
        - hidden state [4096]
        - which KV chunks got high attention scores (ground truth)
Step 3: Save as retrieval training data:
        (query=hidden, positive=high-attention chunks, negative=low-attention)
```

### Retriever Training (cheap, backbone-free):

```
Architecture: Dual encoder
  Query encoder: Linear(4096→1024) + Norm + Linear(1024→1024) + RoPE
  Key encoder: FP8 dequant + identity (keys are already embeddings)
  
Loss: InfoNCE (contrastive)
  Positive: chunks that received >threshold attention weight
  Negative: random chunks from same sequence
  
Hardware: Single GPU or even CPU (retriever is only ~25M params)
Training time: ~2-4 hours on single A10G
Cost: ~$4-8 (spot A10G)
```

### Alternative: Transfer from FlashMemory-DSV4

The released FlashMemory weights are for DeepSeek-V4 (128 heads, MLA).
We can **distill** rather than train from scratch:

```
Option A: Train from scratch on Llama-3.1 attention patterns (cleaner)
Option B: Fine-tune FlashMemory-DSV4 weights with head projection adapter
         (faster, may work since scoring mechanism is similar)

Recommended: Option A (2-4 hours, guarantees optimal results for Llama GQA)
```

---

## 5. Benchmark Plan

### 5.1 Hardware Setup

```
Primary:   trn1.32xlarge (16 NeuronCores, 512GB HBM) — $21.50/hr
Fallback:  trn1.2xlarge  (1 NeuronCore, 32GB HBM)   — $1.34/hr (for single-chip tests)
Baseline:  inf2.xlarge   (1 NeuronCore, no sparse)   — comparison

Estimated total cost: $150-300 (including training data gen on g5)
```

### 5.2 Benchmarks

| Benchmark | Context Length | What It Tests | Metric |
|-----------|:---:|---|---|
| RULER (4K-128K) | 4K→128K | Synthetic retrieval at various depths | Accuracy |
| LongBench-v2 | 46K-493K | Real long-doc QA, summarization | F1/ROUGE |
| Needle in a Haystack | 4K→128K | Single-fact retrieval at all positions | Accuracy |
| Passkey Retrieval | 4K→128K | Random passkey buried in noise | Exact match |
| Perplexity (PG19) | 128K | Language modeling quality | PPL |

### 5.3 Ablation Matrix

| Experiment | KV Retention | Method | Purpose |
|------------|:---:|---|---|
| Baseline-Full | 100% | Standard attention | Upper bound |
| Baseline-Random | 13.5% | Random chunk selection | Lower bound |
| FlashMemory-Port | 13.5% | Trained retriever (our port) | Main result |
| FM-Sweep-5% | 5% | Aggressive sparsity | How far can we push? |
| FM-Sweep-25% | 25% | Conservative sparsity | Easy win baseline |
| FM-Sweep-50% | 50% | Moderate sparsity | Diminishing returns? |
| NKI-Optimized | 13.5% | + NKI sparse gather kernel | Latency improvement |
| Agent-Selected | Variable | RL agent decides retention % | Future work |

### 5.4 Metrics to Report

**Accuracy metrics:**
- Task accuracy (per benchmark)
- Accuracy delta vs full attention
- Accuracy at each context length (scaling curve)

**Efficiency metrics:**
- KV cache memory usage (GB)
- Tokens per second (throughput)
- Time to first token (TTFT)
- Time per output token (TPOT)
- Maximum batch size at 128K context
- Peak HBM utilization %

**Trainium-specific metrics:**
- SBUF utilization %
- HBM bandwidth utilization %
- NeuronCore compute utilization %
- DMA transfer overhead (retriever → kernel)

---

## 6. Expected Results (Hypothesis)

Based on FlashMemory paper + Trainium characteristics:

| Metric | Full Attention | FlashMemory (ours) | Improvement |
|--------|:---:|:---:|:---:|
| KV Cache (128K, batch=1) | 16 GB | 2.2 GB | 7.3x less |
| Max batch (128K, trn1.32xl) | ~30 | ~120+ | 4x throughput |
| RULER accuracy | baseline | ±1% | maintained |
| Tokens/sec (decode) | X | ~1.1-1.3X | faster (less memory traffic) |
| TTFT (128K prefill) | Y | Y (same, dense prefill) | unchanged |

**Key selling point:** On Trainium (32GB/chip), FlashMemory is MORE critical
than on H100 (80GB). The same technique that saves memory on GPU becomes
**essential enabling technology** on Trainium for long-context serving.

---

## 7. Implementation Timeline

### Week 1: Setup & Baseline
- [ ] Compile Llama-3.1-8B on trn1.32xlarge (NeuronX transformers)
- [ ] Run full-attention baseline on RULER + Needle
- [ ] Profile KV cache memory at 16K/32K/64K/128K
- [ ] Record baseline throughput numbers

### Week 2: Retriever Training
- [ ] Generate training data: run Llama-3.1-8B on long docs, capture attention patterns
- [ ] Implement Llama-adapted FlashMemory retriever (~25M params)
- [ ] Train retriever (contrastive loss, 2-4 hours on g5)
- [ ] Validate retriever quality: does top-K match ground truth attention?

### Week 3: Integration & NKI Kernels
- [ ] Write NKI sparse_kv_gather kernel
- [ ] Write NKI sparse_attention kernel (based on our proven v12 attention)
- [ ] Integrate retriever into decode loop (score every 64 steps)
- [ ] End-to-end test: sparse decode generates coherent text

### Week 4: Benchmark & Results
- [ ] Full benchmark suite (RULER, LongBench, Needle, Passkey, PPL)
- [ ] Ablation sweep (5%, 13.5%, 25%, 50% retention)
- [ ] Latency profiling (with vs without NKI optimization)
- [ ] Write results + figures

### Week 5: Blog & Paper Section
- [ ] Blog: "Sparse KV Attention on Trainium: 4x Throughput at 128K Context"
- [ ] Add results to MLSys 2027 paper (if applicable)
- [ ] Push code to research-papers/flashmemory-trainium/

---

## 8. Cost Estimate

| Item | Instance | Hours | Cost |
|------|----------|:---:|:---:|
| Training data gen | g5.2xlarge | 4h | $5 |
| Retriever training | g5.2xlarge | 4h | $5 |
| Baseline profiling | trn1.32xlarge | 4h | $86 |
| Integration & debug | trn1.2xlarge | 8h | $11 |
| Full benchmark | trn1.32xlarge | 6h | $129 |
| Ablation sweep | trn1.32xlarge | 4h | $86 |
| **Total** | | **30h** | **~$322** |

Buffer 50% for retries: **~$500 total budget**

---

## 9. Deliverables

1. **Code:** `research-papers/flashmemory-trainium/`
   - `retriever/` — Llama-adapted FlashMemory retriever
   - `nki_kernels/` — sparse_kv_gather.py, sparse_attention.py
   - `benchmark/` — evaluation scripts
   - `results/` — JSON + figures

2. **Blog:** "Sparse KV Attention on Trainium: Enabling 128K Context with 4x Throughput"
   - Positioning: Trainium's memory constraints → FlashMemory as enabling tech
   - NKI-native implementation → agent-driven optimization angle
   - Benchmark data vs full attention baseline

3. **Paper contribution:** Results feed into MLSys 2027 (cross-hardware section)
   or standalone workshop paper

4. **KEY-FINDINGS.md:** Hardware env, benchmark data, vs paper claims, patches, cost

---

## 10. Risks & Mitigations

| Risk | Probability | Mitigation |
|------|:---:|---|
| Llama-3.1 on NeuronX compilation issues | 20% | Fallback: use Llama-3-8B (shorter context but verified) |
| Retriever quality insufficient for Llama GQA | 25% | More training data; increase scoring layers from 3→6 |
| NKI sparse gather kernel complexity | 15% | Simplified version: just mask attention, don't physically evict |
| 128K context OOM on trn1.2xlarge | 30% | Use trn1.32xlarge (TP=16) for large-context tests |
| Results don't match FlashMemory paper | 20% | Expected: their model is DSV4 (MLA), ours is GQA — different characteristics. Report honestly. |
| Retrieval interval (64) suboptimal for Llama | 15% | Ablate: 32, 64, 128 intervals |

---

## 11. Future: Agent-Driven Extension

Once the baseline FlashMemory port works, the **agentic** extension:

```
Phase A (this design): Fixed retriever, static top-K selection
Phase B (future):      RL-trained agent decides:
                       - Dynamic retention % per layer (not fixed 13.5%)
                       - Adaptive retrieval interval (not fixed 64)
                       - Context-dependent chunk scoring
                       - Hardware-aware: adjust based on current memory pressure

Agent Reward:
  R = α·(batch_throughput / baseline_throughput)
    + β·(accuracy / baseline_accuracy)  
    - γ·(memory_violations)
    - λ·(latency_spikes > threshold)

Governance (Cedar):
  DENY: evict_chunk IF chunk.is_system_prompt
  DENY: retention_ratio < 5% (minimum safety)
  DENY: retrieval_interval > 256 (staleness risk)
  LIMIT: SBUF_usage < 95%
```

This is where "Agentic AI for AI Infra" fully manifests —
the agent doesn't just USE the infrastructure, it OPTIMIZES it in real-time.

---

## 12. One-Liner Pitch

> "FlashMemory proves you only need 13.5% of KV cache. We prove it works on Trainium —
> enabling 4x throughput at 128K context — and let an agent find the optimal sparse pattern."

---

*Design by Atlas 🔭 | 2026-06-15*
*Model: Llama-3.1-8B-Instruct (128K) on trn1.32xlarge*
*Budget: ~$500 | Timeline: 5 weeks*
*Integrates: FlashMemory (2606.09079) + NKI-Agent v12 attention kernel + Trainium memory hierarchy*
