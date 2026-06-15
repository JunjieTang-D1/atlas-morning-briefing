# Agentic AI for AI Infrastructure — Design Document

> "The best infrastructure doesn't just run models — it improves itself."

---

## 1. Thesis

**Use agentic AI to solve AI infrastructure problems.**

Traditional AI infra (kernels, compilers, memory management, serving) is built by
human experts writing hand-optimized code. This doesn't scale — each new hardware
generation (Trainium, Blackwell, TPU v6) requires re-optimization from scratch.

Our thesis: **AI agents can automate AI infrastructure optimization**, from kernel
generation to memory management to serving efficiency. The agent becomes the
infrastructure engineer.

```
Traditional:    Human Expert → Hand-Tuned Kernel → One Hardware Target
Our Approach:   Agent + RL → Auto-Generated Kernel → Any Hardware Target
```

---

## 2. Positioning — Why This Is Unique

```
                    Manual Optimization          Agent-Driven Optimization
                    ──────────────────          ─────────────────────────
  Single Hardware   │ FlashAttention (GPU)     │ NKI-Agent (Trainium) ← YOU
                    │ NeuronMM (Trainium)      │
                    │                          │
  Cross-Hardware    │ Triton (GPU-only)        │ AtX Kernel Migration ← YOU
                    │ TVM (compile-only)       │
                    │                          │
  Memory/Serving    │ FlashMemory (GPU)        │ Agentic KV Manager   ← YOU (NEW)
                    │ PagedAttention           │   (Trainium-native)
                    │                          │
  RL-Trained        │ (nobody)                 │ Governance-Constrained ← YOU (NEW)
  Infra Agent       │                          │   RL for Infra Agents
```

**Your moat:** Nobody else combines (1) Trainium-native NKI expertise + (2) agent-driven
kernel generation + (3) RL training with governance constraints. Each piece exists
separately; the integration is novel.

---

## 3. Architecture — The Agentic Infra Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AGENTIC INFRA PLATFORM                           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LAYER 4: RL TRAINING LOOP                       │    │
│  │                                                               │    │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐ │    │
│  │  │ Trajectory   │  │ Reward        │  │ Governance       │ │    │
│  │  │ Collector    │  │ Computer      │  │ Constraints      │ │    │
│  │  │ (Polar-style)│  │ (perf+safety) │  │ (Cedar policies) │ │    │
│  │  └──────────────┘  └───────────────┘  └──────────────────┘ │    │
│  │                                                               │    │
│  │  Training Signal: R(τ) = α·Speedup + β·Correctness - λ·Violations │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↕                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LAYER 3: AGENT BRAIN                             │    │
│  │                                                               │    │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐ │    │
│  │  │ Kernel       │  │ Memory        │  │ Compiler         │ │    │
│  │  │ Generator    │  │ Optimizer     │  │ Strategist       │ │    │
│  │  │ Agent        │  │ Agent         │  │ Agent            │ │    │
│  │  │              │  │               │  │                  │ │    │
│  │  │ "Write NKI   │  │ "Decide KV    │  │ "Choose compile  │ │    │
│  │  │  kernel code"│  │  eviction"    │  │  flags & tiling" │ │    │
│  │  └──────────────┘  └───────────────┘  └──────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↕                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LAYER 2: HARDWARE ABSTRACTION                    │    │
│  │                                                               │    │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐ │    │
│  │  │ NKI          │  │ CUDA/Triton   │  │ XLA/StableHLO    │ │    │
│  │  │ (Trainium)   │  │ (GPU)         │  │ (TPU)            │ │    │
│  │  └──────────────┘  └───────────────┘  └──────────────────┘ │    │
│  │                                                               │    │
│  │  Unified Interface: compile(kernel) → binary + profile        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↕                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LAYER 1: HARDWARE                                │    │
│  │                                                               │    │
│  │  Trainium (trn1/trn2)  │  GPU (H100/B200)  │  TPU (v5/v6)  │    │
│  │  32GB HBM, 24MB SBUF   │  80GB HBM         │  32GB HBM     │    │
│  │  NeuronCore v2          │  SM + Tensor Core  │  MXU          │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Three Agent Domains

### 4.1 Kernel Generator Agent (PROVEN — NKI-Agent + Attention-Forge)

**Status:** ✅ Demonstrated. Agent generates NKI attention kernels matching human expert.

**What it does:**
- Takes a high-level spec (e.g., "flash attention, seq_len=4096, head_dim=128")
- Generates NKI kernel code (Python + NKI API calls)
- Compiles on Trainium, profiles, iterates until performance target met

**Key results (proven):**
- v12 agent-generated kernel matches v7 (human expert) at all sequence lengths
- 512: 19µs, 1K: 39µs, 4K: 400µs, 8K: 1526µs, 16K: 6236µs
- Agent discovered K-scaling trick (novel: saves 4MB SBUF at 16K)

**Next evolution:**
- Cross-hardware: same agent generates CUDA + NKI from single spec (AtX paper)
- RL training: GRPO on kernel generation trajectories (governance: no unsafe memory access)
- Expand beyond attention: matmul, conv, normalization, activation kernels

```
Spec: "FlashAttention, causal, bf16, seq=4096, head=128"
      ↓
Agent Brain (LLM + NKI knowledge)
      ↓
Generated: attention_v12.py (NKI kernel code)
      ↓
Compile → Profile → Reward = speedup_vs_baseline
      ↓
Iterate until: latency ≤ target OR budget exhausted
```

---

### 4.2 Memory Optimizer Agent (NEW — FlashMemory-inspired)

**Status:** 💡 Design phase. Inspired by FlashMemory-DeepSeek-V4.

**What it does:**
- Decides which KV cache chunks to keep in HBM vs offload/evict
- Predicts future attention patterns (lookahead)
- Optimizes for Trainium's specific memory hierarchy (HBM → SBUF → PSUM)

**Why Trainium-specific matters:**
```
Trainium Memory Hierarchy:
  PSUM:  384KB  (fastest, register-like)    ← hot attention scores
  SBUF:  24MB   (fast, on-chip SRAM)        ← active KV chunks
  HBM:   32GB   (slow, off-chip DRAM)       ← full KV cache
  
GPU Memory Hierarchy (for comparison):
  Registers → L1/Shared Mem (256KB) → L2 (50MB) → HBM (80GB)
```

Trainium's 24MB SBUF is the key constraint. FlashMemory's 13.5% KV retention
means at seq_len=128K with 128 heads × 128 dim × bf16:
- Full KV: ~4GB per layer → doesn't fit in HBM for many layers
- 13.5%: ~540MB per layer → fits with batching
- Critical chunks in SBUF: ~24MB → agent must pick the TOP 0.5% to keep hot

**Architecture:**

```
┌─────────────────────────────────────────────────────┐
│           MEMORY OPTIMIZER AGENT                      │
│                                                       │
│  Input:                                               │
│    • Current query embedding                          │
│    • KV chunk index (metadata of all chunks)          │
│    • Memory pressure signal (HBM utilization %)       │
│    • Historical access patterns                       │
│                                                       │
│  Decision:                                            │
│    • Which KV chunks → SBUF (hot, immediate access)   │
│    • Which KV chunks → HBM (warm, prefetchable)       │
│    • Which KV chunks → evict (cold, recompute if needed)│
│                                                       │
│  Constraint (Governance):                             │
│    • accuracy_drop < 1% vs full attention             │
│    • SBUF utilization < 95% (leave headroom)          │
│    • No eviction of "anchor" tokens (BOS, system)     │
│    • Prefetch must complete before needed             │
│                                                       │
│  Reward:                                              │
│    R = α·(1 - HBM_usage/capacity)                    │
│      + β·(accuracy vs full_attention)                 │
│      - γ·(prefetch_miss_rate)                         │
│      - λ·(governance_violations)                      │
└─────────────────────────────────────────────────────┘
```

**Trainium-Native Implementation:**

```python
# NKI kernel: sparse KV gather based on agent's index selection
@nki.jit
def sparse_kv_gather(kv_cache_hbm, index_mask, sbuf_budget):
    """
    Agent provides index_mask (which chunks to load).
    Kernel implements efficient sparse gather into SBUF.
    """
    # Agent selected top-K chunk indices
    selected = nl.load(index_mask)  # small tensor
    
    # Gather only selected KV chunks from HBM → SBUF
    for chunk_id in nl.affine_range(selected.shape[0]):
        kv_chunk = nl.load(kv_cache_hbm[selected[chunk_id]])
        # ... attention computation on sparse subset
```

**Training the Memory Agent:**
- Use FlashMemory's dual-encoder as initialization (distill → small model)
- Fine-tune with RL: reward = (accuracy_maintained × memory_saved)
- Governance constraint: never evict tokens that cause >1% accuracy loss
- Hardware-aware: reward shaped by Trainium SBUF/HBM latency ratios

---

### 4.3 Compiler Strategist Agent (FUTURE)

**Status:** 🔮 Roadmap. Builds on NeuronX Compiler knowledge.

**What it does:**
- Chooses optimal compiler flags, tiling strategies, fusion patterns
- Adapts compilation strategy per model architecture (MoE vs dense vs MLA)
- Learns from compilation outcomes (latency, memory, correctness)

**Why it matters:**
- NeuronX Compiler (neuronx-cc) has 100+ flags
- Optimal settings vary by model, batch size, sequence length
- Currently: human trial-and-error or defaults
- Agent: learns optimal compilation strategy per workload

**Deferred to Phase 3** — requires deeper NeuronX CC integration.

---

## 5. Unified Training Framework

All three agents share a common RL training infrastructure:

```
┌────────────────────────────────────────────────────────────────┐
│                UNIFIED RL TRAINING LOOP                          │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│  │ Kernel   │    │ Memory   │    │ Compiler │                 │
│  │ Agent    │    │ Agent    │    │ Agent    │                 │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                 │
│       │               │               │                         │
│       ▼               ▼               ▼                         │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              TRAJECTORY COLLECTOR                     │       │
│  │  (Polar-style: proxy LLM API calls, capture actions) │       │
│  └─────────────────────────────────────────────────────┘       │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              REWARD COMPUTATION                       │       │
│  │                                                       │       │
│  │  Kernel Agent:                                        │       │
│  │    R = speedup_vs_baseline × correctness_score        │       │
│  │                                                       │       │
│  │  Memory Agent:                                        │       │
│  │    R = memory_saved × accuracy_preserved              │       │
│  │                                                       │       │
│  │  Compiler Agent:                                      │       │
│  │    R = compile_time_reduction × output_quality        │       │
│  │                                                       │       │
│  │  ALL: - λ · governance_violations                     │       │
│  └─────────────────────────────────────────────────────┘       │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              GOVERNANCE HARNESS (Cedar)               │       │
│  │                                                       │       │
│  │  Kernel: no unsafe memory access, no SBUF overflow    │       │
│  │  Memory: no accuracy drop >1%, no anchor eviction     │       │
│  │  Compiler: no invalid flag combinations, timeout < 5min│      │
│  │                                                       │       │
│  │  Violations → negative reward signal (not just block)  │       │
│  │  = "The harness trains the agent, not just constrains" │       │
│  └─────────────────────────────────────────────────────┘       │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              GRPO UPDATE                              │       │
│  │  Group-relative policy optimization on trajectories   │       │
│  │  Constrained variant: governance as reward shaping    │       │
│  └─────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. Trainium-Specific Advantages

Why this platform is **differentiated** on Trainium vs GPU:

| Aspect | GPU Ecosystem | Trainium + Agentic Infra |
|--------|--------------|--------------------------|
| Kernel authoring | Triton/CUDA (mature, many experts) | NKI (new, few experts) → agent fills the gap |
| Memory management | PagedAttention, FlashMemory | Sparse KV on SBUF (24MB sweet spot) → agent-optimized |
| Compiler | nvcc (decades of optimization) | neuronx-cc (young, many knobs) → agent explores faster |
| Long context | 80GB HBM = brute force works | 32GB HBM = MUST be clever → agent-driven sparsity |
| Cost | $2-4/hr (H100) | $1.34/hr (trn1) → cheaper experiments, faster RL loops |
| Competition | Crowded (NVIDIA, Triton community) | Blue ocean (few NKI experts in the world) |

**Key insight:** Trainium's constraints (smaller HBM, younger compiler, fewer experts)
make it the IDEAL platform for agentic optimization — the agent provides what the
ecosystem lacks.

---

## 7. Connection to Existing Work

```
YOUR PORTFOLIO:
                                                          
  ┌─────────────────┐     ┌─────────────────┐     ┌────────────────┐
  │ NKI-Agent       │     │ Attention-Forge  │     │ AtX (Cross-HW) │
  │ (Blog, proven)  │     │ (MLSys paper)    │     │ (Blog, submitted)│
  │                 │     │                  │     │                │
  │ Agent generates │     │ Agent evolves    │     │ Same agent,    │
  │ NKI kernels     │     │ attention kernel │     │ CUDA + NKI     │
  └────────┬────────┘     └────────┬─────────┘     └───────┬────────┘
           │                       │                        │
           └───────────────────────┼────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   AGENTIC AI FOR AI INFRA   │
                    │   (This Design Document)     │
                    │                              │
                    │   Unified vision:            │
                    │   • Kernel Gen (proven)      │
                    │   • Memory Opt (new)         │ ← FlashMemory angle
                    │   • Compiler Strategy (future)│
                    │   • RL Training (AgentCore)  │ ← Governance RL
                    │   • Trainium-native (unique) │
                    └──────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      DELIVERABLES            │
                    │                              │
                    │  • MLSys 2027 paper (ready)  │
                    │  • Blog: Sparse KV on Trn    │
                    │  • AgentCore Science paper   │
                    │  • Open-source toolkit       │
                    └─────────────────────────────┘
```

---

## 8. Roadmap

### Phase 1: Kernel Generation (✅ DONE)
- NKI-Agent: proven, blog published
- Attention-Forge: MLSys paper ready
- AtX cross-hardware: blog submitted

### Phase 2: Memory Optimization (Q3 2026)
- Design sparse KV gather NKI kernel
- Port FlashMemory's indexer concept to Trainium
- Benchmark: DeepSeek-V4 style model, 128K context on trn1
- Blog: "Sparse KV Attention on Trainium"

### Phase 3: RL Training Loop (Q3-Q4 2026)
- Implement governance-constrained GRPO for kernel agent
- Train memory optimizer agent via RL (reward = memory_saved × accuracy)
- AgentCore Science paper (H1→H4 hypotheses from research brief)
- Workshop submission (KDD / DL4C)

### Phase 4: Unified Platform (Q4 2026 - Q1 2027)
- Unify kernel + memory + compiler agents into single platform
- Open-source release: "Agentic Infra Toolkit for Trainium"
- Full paper: unified agentic infrastructure optimization

---

## 9. Paper & Content Pipeline

| # | Deliverable | Timeline | Status |
|---|-------------|----------|--------|
| 1 | MLSys 2027: Cross-HW Kernel Migration | Ready (submission TBD) | 🟢 |
| 2 | Blog: Sparse KV on Trainium (FlashMemory port) | Q3 2026 | 💡 New |
| 3 | AgentCore Science: Governance RL for Infra Agents | Q3-Q4 2026 | 💡 Concept |
| 4 | Blog: RL-Trained Kernel Agent (GRPO results) | Q4 2026 | 🔮 Future |
| 5 | Full paper: Agentic AI for AI Infrastructure | Q1 2027 | 🔮 Future |

---

## 10. Key Differentiators (Elevator Pitch)

**For LinkedIn / conferences:**
> "I build AI agents that write AI infrastructure code. My kernel generation agent
> matches human experts on AWS Trainium. Next: agents that optimize memory management
> and compiler strategies — trained with RL under governance constraints."

**For AWS internal:**
> "Agentic AI for Trainium optimization — agents auto-generate NKI kernels,
> manage KV cache sparsity, and tune compiler flags. Reduces the expert-dependency
> bottleneck for Trainium adoption."

**For academic (paper abstract):**
> "We present a unified framework for training AI agents to optimize AI infrastructure.
> Our governance-constrained RL approach trains agents that generate hardware kernels,
> manage memory hierarchies, and select compiler strategies — achieving expert-level
> performance while provably respecting safety constraints."

---

## 11. Research Foundation

| Paper | Contribution to This Work |
|-------|---------------------------|
| FlashMemory-DeepSeek-V4 (2606.09079) | Sparse KV indexing works (+0.6% acc, 13.5% memory) |
| FlashAttention-4 (2603.05451) | Dense compute optimization baseline |
| Polar (2605.24220) | RL training infrastructure for agents (GRPO proxy) |
| Safe RLHF / CS-RLHF (2310.12773, 2510.03520) | Constrained RL methodology |
| NKI-Agent (yours, blog) | Agent kernel generation proven on Trainium |
| AtX Cross-Hardware (yours, MLSys) | Same agent, multiple hardware targets |
| Marc Brooker "Box" (Jan 2026) | Safety = external harness, not internal prompt |
| ATOM (2605.26178) | Nucleus-Electron architecture for agent pools |
| IBM Policy Guards (2507.16459) | Policy compilation into runtime guards |

---

## 12. Risk & Mitigation

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| FlashMemory's indexer doesn't port cleanly to Trainium | 25% | NKI sparse gather is feasible (proven primitives exist); fallback: simplified top-K selection |
| RL training too expensive for infra agents | 20% | Start small (Qwen3.5-4B); trn1 cheaper than H100 for training |
| Governance constraints hurt infra performance | 30% | H1 hypothesis — if false, pivot to depth-scaling (H3) |
| Similar work appears before publish | 15% | Trainium-native angle is niche enough; NKI expertise is rare |
| Trainium adoption stalls | 10% | Framework is hardware-agnostic at Layer 3; Trainium is primary but not only target |

---

## 13. One-Liner Vision

> **"AI agents that build AI infrastructure — trained safely, deployed on Trainium, better than human experts."**

---

*Design by Atlas 🔭 | 2026-06-15*
*Integrates: NKI-Agent (2026-03), Attention-Forge (2026-03), AtX (2026-05), FlashMemory analysis (2026-06), AgentCore Science (2026-06)*
