# Implementation Loop Tracker — HASA on Trainium2

## Status: 🔵 PHASE 0 (Foundation)

---

## Loop Definition

```
WHILE paper_not_submitted:
    design_hypothesis()
    implement_code()
    run_benchmark()
    analyze_results()
    IF results_meet_exit_criteria():
        advance_to_next_phase()
    ELSE:
        revise_design()
        CONTINUE
```

---

## Phase 0: Foundation (Local, No GPU)

### Tasks
- [x] DESIGN.md written (MLSys-grade, Option A, trn2, 70B, high-concurrency)
- [x] LOOP.md written (this file)
- [ ] `src/retriever.py` — Llama-70B adapted retriever
- [ ] `src/hasa_placer.py` — two-tier placement logic
- [ ] `src/serving_loop.py` — decode loop skeleton
- [ ] `src/kernels/hasa_gather.py` — NKI kernel (compilable stub)
- [ ] `src/kernels/hasa_attention.py` — NKI kernel (compilable stub)
- [ ] `src/kernels/hasa_score.py` — NKI kernel (compilable stub)
- [ ] `benchmarks/accuracy.py` — benchmark harness
- [ ] `benchmarks/system.py` — throughput measurement
- [ ] `benchmarks/baselines/` — 4 baseline implementations
- [ ] Unit tests passing
- [ ] Scripts ready

### Exit Criteria
- [ ] All unit tests pass (shapes, dtypes, numerics within 1e-5)
- [ ] Retriever forward pass produces valid scores [0,1] on mock data
- [ ] NKI kernel stubs compile with neuronx-cc --target trn2 (dry-run)
- [ ] Benchmark harness loads all 5 datasets

### Decision Points
- If NKI trn2 target not available in SDK: fallback to trn1 kernels + trn2 profiling
- If Llama-70B too large for attention map extraction: use Llama-8B for retriever training, evaluate on 70B

---

## Phase 1: Kernel Development (trn2, single chip)

### Tasks
- [ ] SSH into trn2.48xlarge instance
- [ ] Install Neuron SDK 2.22+
- [ ] Compile hasa_gather.py on real hardware
- [ ] Compile hasa_attention.py on real hardware  
- [ ] Compile hasa_score.py on real hardware
- [ ] Correctness tests vs PyTorch reference
- [ ] Latency profiling (Neuron profiler)
- [ ] Iterate until exit criteria met

### Exit Criteria
- [ ] hasa_gather: correct output, < 50µs
- [ ] hasa_attention: within 1e-3 of dense, < 200µs
- [ ] hasa_score: matches PyTorch within 1e-4
- [ ] SBUF utilization ≤ 90%

### Estimated Cost: ~$172 (8h × $21.50)

---

## Phase 2: Retriever Training (GPU)

### Tasks
- [ ] Generate training data (run 70B on g5/p4d, capture attention)
- [ ] Implement training loop (InfoNCE, dual encoder)
- [ ] Train retriever (~50M params, 2-4 hours)
- [ ] Evaluate: Recall@K, NDCG, generalization
- [ ] Export to NKI-compatible format

### Exit Criteria
- [ ] Recall@256 ≥ 85%
- [ ] NDCG@256 ≥ 0.80
- [ ] Test/train gap < 5%
- [ ] Inference < 1ms on NeuronCore

### Estimated Cost: ~$15 (12h spot g5)

---

## Phase 3: Full System Benchmark (trn2.48xlarge)

### Tasks
- [ ] Load 70B model (TP=16)
- [ ] Integrate HASA decode loop
- [ ] Run accuracy benchmarks (5 benchmarks × 4 lengths)
- [ ] Run system benchmarks (throughput, batch scaling)
- [ ] Run all baselines (StreamingLLM, H2O, SnapKV, FlashMemory-flat)
- [ ] Generate comparison tables + figures

### Exit Criteria
- [ ] Within 2% accuracy of full attention (RULER-128K)
- [ ] ≥ 4x throughput at batch=128
- [ ] Max batch ≥ 128 at 128K
- [ ] HASA > FlashMemory-flat on ≥1 key metric by ≥5%

### Estimated Cost: ~$344 (16h × $21.50)

---

## Phase 4: Revision (iterate)

### Trigger Conditions:
- Accuracy below target → increase retention%, adjust scoring layers
- Throughput below target → optimize DMA pipelining, reduce retriever overhead
- SBUF hit rate below 80% → tune placement bias, increase hot-chunk budget
- Baseline comparison unfavorable → investigate why, may need architectural change

### Estimated: 1-2 revision cycles, ~$172 each

---

## Phase 5: Paper

### Deliverables:
- [ ] LaTeX (MLSys 2027 format, 10 pages)
- [ ] Figures: throughput scaling, accuracy tables, ablation plots
- [ ] Supplementary: kernel code listings, training details

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-15 | Option A (hardware-aware) over B (RL agent) | Higher acceptance probability, executable in 8 weeks |
| 2026-06-15 | trn2.48xlarge (not trn1) | 96GB/chip enables 70B, real memory pressure |
| 2026-06-15 | Llama-3.1-70B (not 8B) | MLSys needs production-scale, 8B has no memory pressure |
| 2026-06-15 | PyTorch native (no torch_xla) | Modern Neuron SDK approach, cleaner code |
| 2026-06-15 | 5 baselines (not just full attn) | MLSys reviewers require competitive comparison |

---

## Risk Register

| Risk | Mitigation | Status |
|------|-----------|--------|
| trn2 NKI compile fails | Fallback to trn1 kernels | ⚪ Not yet tested |
| 70B attention map extraction OOM | Use 8B for training, 70B for eval | ⚪ Not yet tested |
| Retriever quality insufficient | Add more scoring layers, larger model | ⚪ Not yet tested |
| SBUF size assumption wrong (not 48MB) | Adjust tier budgets dynamically | ⚪ Not yet tested |
| trn2 quota denied | Request limit increase, start with trn1 | ⚪ Not yet tested |
