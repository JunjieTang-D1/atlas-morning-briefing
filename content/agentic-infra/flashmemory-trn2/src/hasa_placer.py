"""
HASA Two-Tier Placement Logic.

Decides which KV chunks go to which memory tier based on retriever scores
and hardware constraints (SBUF size, HBM capacity, batch pressure).
"""

import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PlacementConfig:
    """Hardware-aware placement configuration for Trainium2."""
    
    # Hardware specs (trn2.48xlarge, per chip)
    sbuf_size_bytes: int = 48 * 1024 * 1024    # 48 MB SBUF per chip
    hbm_size_bytes: int = 96 * 1024 * 1024 * 1024  # 96 GB HBM per chip
    n_chips: int = 16                            # Chips in trn2.48xlarge
    
    # Model dimensions (Llama-3.1-70B)
    n_kv_heads: int = 8
    head_dim: int = 128
    n_layers: int = 80
    chunk_size: int = 64        # Tokens per chunk
    dtype_bytes: int = 2        # bf16
    
    # Placement policy
    sink_tokens: int = 4        # Always keep first N tokens (attention sink)
    window_tokens: int = 256    # Always keep last N tokens (sliding window)
    min_retention: float = 0.05 # Never drop below 5% retention (safety)
    max_sbuf_util: float = 0.90 # Don't fill SBUF beyond 90% (leave headroom)
    
    # Scoring adjustments
    sbuf_residency_bonus: float = 0.05   # Bonus for chunks already in SBUF
    eviction_hysteresis: float = 0.02    # Don't evict unless score drops by this much


class HASAPlacer:
    """Two-tier placement engine for KV cache chunks.
    
    Tier 0 (SBUF): On-chip SRAM. Zero-latency access. Limited capacity.
    Tier 1 (HBM): Off-chip DRAM. DMA latency. Large capacity.
    Tier 2 (Evicted): Not in memory. Chunks masked to -inf in attention.
    """
    
    def __init__(self, config: PlacementConfig):
        self.config = config
        self._bytes_per_chunk = self._compute_chunk_bytes()
    
    def _compute_chunk_bytes(self) -> int:
        """Bytes per KV chunk (K+V, one layer, bf16)."""
        # One chunk = chunk_size tokens × n_kv_heads × head_dim × 2 (K+V) × dtype_bytes
        return (
            self.config.chunk_size 
            * self.config.n_kv_heads 
            * self.config.head_dim 
            * 2  # K and V
            * self.config.dtype_bytes
        )
    
    def compute_budgets(
        self,
        seq_len: int,
        batch_size: int,
        tp_degree: int = 16,
    ) -> Tuple[int, int, int]:
        """Compute per-sequence chunk budgets given hardware constraints.
        
        Returns:
            sbuf_budget: chunks per sequence in SBUF
            hbm_budget: chunks per sequence in HBM
            total_chunks: total chunks per sequence
        """
        total_chunks = seq_len // self.config.chunk_size
        
        # SBUF budget (per chip, shared across batch)
        # We store ONE layer's KV in SBUF (the layer being computed)
        usable_sbuf = int(self.config.sbuf_size_bytes * self.config.max_sbuf_util)
        # Chunks that fit in SBUF across the batch on this chip
        chunks_in_sbuf_total = usable_sbuf // self._bytes_per_chunk
        # Per sequence (divide by batch_size / tp_degree for local batch)
        local_batch = max(1, batch_size // tp_degree)
        sbuf_per_seq = chunks_in_sbuf_total // local_batch
        sbuf_budget = min(sbuf_per_seq, total_chunks)
        
        # HBM budget (retention-based)
        # Total HBM available for KV (after model weights)
        model_bytes = 140 * 1024 * 1024 * 1024  # ~140GB for 70B bf16
        kv_hbm_total = (
            self.config.hbm_size_bytes * self.config.n_chips 
            - model_bytes
        )
        # KV per sequence at full retention
        full_kv_per_seq = total_chunks * self._bytes_per_chunk * self.config.n_layers
        # Max sequences that fit
        max_batch_full = kv_hbm_total // full_kv_per_seq
        
        # Target retention to fit desired batch
        if batch_size <= max_batch_full:
            # Full attention fits — but we still use sparse for throughput
            hbm_retention = 0.135  # Default FlashMemory-style
        else:
            # Must sparsify to fit batch
            hbm_retention = (kv_hbm_total / batch_size) / (full_kv_per_seq)
            hbm_retention = max(hbm_retention, self.config.min_retention)
        
        hbm_budget = int(total_chunks * hbm_retention) - sbuf_budget
        hbm_budget = max(hbm_budget, 0)
        
        return sbuf_budget, hbm_budget, total_chunks
    
    def place(
        self,
        scores: torch.Tensor,          # [B, N_chunks] retriever scores
        seq_len: int,
        batch_size: int,
        current_tier: Optional[torch.Tensor] = None,  # [B, N] 0=evicted, 1=HBM, 2=SBUF
    ) -> torch.Tensor:
        """Assign chunks to tiers.
        
        Returns:
            placement: [B, N_chunks] with values 0 (evict), 1 (HBM), 2 (SBUF)
        """
        B, N = scores.shape
        sbuf_budget, hbm_budget, total_chunks = self.compute_budgets(seq_len, batch_size)
        
        # Ensure mandatory chunks (sink + window) are always in SBUF
        adjusted_scores = scores.clone()
        
        # Sink tokens (first few chunks)
        n_sink_chunks = max(1, self.config.sink_tokens // self.config.chunk_size)
        adjusted_scores[:, :n_sink_chunks] += 10.0  # Guaranteed top score
        
        # Window tokens (last few chunks)
        n_window_chunks = max(1, self.config.window_tokens // self.config.chunk_size)
        adjusted_scores[:, -n_window_chunks:] += 10.0  # Guaranteed top score
        
        # Residency bonus (hysteresis: prefer keeping chunks where they are)
        if current_tier is not None:
            sbuf_resident = (current_tier == 2).float()
            adjusted_scores += self.config.sbuf_residency_bonus * sbuf_resident
        
        # Placement decision
        placement = torch.zeros(B, N, dtype=torch.long, device=scores.device)
        
        # Top sbuf_budget → SBUF (tier 2)
        _, sbuf_idx = adjusted_scores.topk(sbuf_budget, dim=-1)
        placement.scatter_(1, sbuf_idx, 2)
        
        # Next hbm_budget → HBM (tier 1)
        remaining = adjusted_scores.clone()
        remaining.scatter_(1, sbuf_idx, -float('inf'))
        _, hbm_idx = remaining.topk(hbm_budget, dim=-1)
        placement.scatter_(1, hbm_idx, 1)
        
        # Rest stays 0 (evicted)
        return placement
    
    def summary(self, seq_len: int, batch_size: int) -> str:
        """Human-readable placement summary."""
        sbuf_b, hbm_b, total = self.compute_budgets(seq_len, batch_size)
        evicted = total - sbuf_b - hbm_b
        
        return (
            f"HASA Placement (seq={seq_len}, batch={batch_size}):\n"
            f"  Total chunks: {total} ({total * self.config.chunk_size} tokens)\n"
            f"  SBUF tier:    {sbuf_b} chunks ({sbuf_b/total*100:.1f}%) — zero-latency\n"
            f"  HBM tier:     {hbm_b} chunks ({hbm_b/total*100:.1f}%) — DMA fetch\n"
            f"  Evicted:      {evicted} chunks ({evicted/total*100:.1f}%) — not used\n"
            f"  Memory saved: {evicted/total*100:.1f}%\n"
            f"  Bytes/chunk:  {self._bytes_per_chunk:,}\n"
        )


if __name__ == "__main__":
    config = PlacementConfig()
    placer = HASAPlacer(config)
    
    # Show placement at various batch sizes
    for batch in [32, 64, 128, 256]:
        print(placer.summary(seq_len=131072, batch_size=batch))
        print()
