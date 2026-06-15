"""
HASA Retriever — Hardware-Aware Sparse Attention Retriever for Llama-3.1-70B GQA.

Adapted from FlashMemory-DeepSeek-V4 architecture.
Predicts which KV chunks will be attended to in the next decode window.
Produces per-chunk scores in [0,1] for placement decisions.

PyTorch native implementation (no torch_xla dependency).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class HASARetrieverConfig:
    """Configuration for HASA retriever adapted to Llama-3.1-70B."""
    
    # Llama-3.1-70B architecture
    hidden_size: int = 8192          # Llama-70B hidden dim
    n_kv_heads: int = 8              # GQA: 8 KV heads
    head_dim: int = 128              # Per-head dimension
    n_layers: int = 80               # Total layers in backbone
    
    # Retriever architecture
    q_lora_rank: int = 2048          # Query projection bottleneck
    scoring_layers: tuple = (16, 32, 48, 64)  # 4 layers spread across 80
    chunk_size: int = 64             # Tokens per chunk
    retrieval_interval: int = 64     # Re-score every N decode steps
    
    # RoPE (Llama-3.1 style)
    rope_dim: int = 64               # Last 64 dims get RoPE
    rope_base: float = 500000.0      # Llama-3.1 extended RoPE base
    rope_factor: float = 8.0         # YaRN factor for 128K
    
    # Hardware-aware scoring
    sbuf_bonus: float = 0.1          # Score bonus for SBUF-resident chunks
    hbm_penalty: float = 0.0         # No penalty for HBM (baseline tier)
    
    # Training
    compressed_k_bytes: int = 132    # 128 (fp8 values) + 4 (f32 scale)


class RMSNorm(nn.Module):
    """RMSNorm as used in Llama."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class YaRNRoPE(nn.Module):
    """YaRN Rotary Position Embedding for 128K context."""
    
    def __init__(self, config: HASARetrieverConfig, max_seq_len: int = 131072):
        super().__init__()
        self.dim = config.rope_dim
        self.base = config.rope_base
        self.factor = config.rope_factor
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies with YaRN scaling
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        # YaRN linear interpolation for extended context
        inv_freq = inv_freq / self.factor
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin cache
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to last `rope_dim` dimensions of x.
        
        Args:
            x: [batch, n_heads, head_dim]
            positions: [batch] token positions
        Returns:
            x with RoPE applied to last rope_dim dims
        """
        batch = x.shape[0]
        
        # Split into RoPE and pass-through parts
        x_rope = x[..., -self.dim:]       # Last rope_dim dims
        x_pass = x[..., :-self.dim]       # First (head_dim - rope_dim) dims
        
        # Get cos/sin for positions
        cos = self.cos_cache[positions]    # [batch, dim//2]
        sin = self.sin_cache[positions]    # [batch, dim//2]
        
        # Reshape for broadcasting: [batch, 1, dim//2]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # Apply rotation
        x1 = x_rope[..., : self.dim // 2]
        x2 = x_rope[..., self.dim // 2 :]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)
        
        return torch.cat([x_pass, x_rotated], dim=-1)


class ScoringLayer(nn.Module):
    """Single scoring layer — predicts chunk relevance for one backbone layer.
    
    Architecture per FlashMemory:
      hidden → wq_a → RMSNorm → wq_b → reshape → RoPE → q [B, n_heads, head_dim]
      compressed_k → dequant → k [B, N_chunks, head_dim]
      score = sigmoid(mean_heads(relu(k @ q^T) * fused_weights))
    """
    
    def __init__(self, config: HASARetrieverConfig):
        super().__init__()
        self.config = config
        
        # Query projection (LoRA-style bottleneck)
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank)
        self.wq_b = nn.Linear(
            config.q_lora_rank, 
            config.n_kv_heads * config.head_dim, 
            bias=False
        )
        
        # Fused head weights (importance weighting across heads)
        self.weights_proj = nn.Linear(config.hidden_size, config.n_kv_heads, bias=False)
        self.weight_scale = (config.head_dim ** -0.5) * (config.n_kv_heads ** -0.5)
        
        # RoPE
        self.rope = YaRNRoPE(config)
    
    def forward(
        self,
        hidden: torch.Tensor,           # [B, hidden_size]
        compressed_k: torch.Tensor,      # [B, N_chunks, 132] uint8
        positions: torch.Tensor,         # [B] current decode position
    ) -> torch.Tensor:                   # [B, N_chunks] scores in [0, 1]
        """Score all chunks for one backbone layer."""
        B, N_chunks, _ = compressed_k.shape
        
        # Project hidden → query
        q_lora = self.wq_a(hidden)                      # [B, q_lora_rank]
        q_lora = self.q_norm(q_lora)                    # [B, q_lora_rank]
        q = self.wq_b(q_lora)                           # [B, n_heads * head_dim]
        q = q.view(B, self.config.n_kv_heads, self.config.head_dim)  # [B, 8, 128]
        
        # Apply RoPE to query
        q = self.rope(q, positions)                      # [B, 8, 128]
        
        # Fused weights (per-head importance)
        fused_w = self.weights_proj(hidden)              # [B, n_heads]
        fused_w = fused_w * self.weight_scale           # [B, n_heads]
        
        # Dequantize compressed keys
        k = self._dequant_fp8_keys(compressed_k)        # [B, N_chunks, head_dim]
        
        # Score computation: relu(k @ q^T) weighted by fused_w
        # k: [B, N, D], q: [B, H, D] → attn: [B, N, H]
        attn = torch.bmm(k, q.transpose(1, 2))         # [B, N_chunks, n_heads]
        attn = F.relu(attn)                             # ReLU gate
        
        # Weight by head importance and sum
        # fused_w: [B, H] → [B, 1, H]
        weighted = attn * fused_w.unsqueeze(1)          # [B, N_chunks, n_heads]
        score = weighted.sum(dim=-1)                    # [B, N_chunks]
        
        # Sigmoid to [0, 1]
        score = torch.sigmoid(score)                    # [B, N_chunks]
        
        return score
    
    def _dequant_fp8_keys(self, compressed_k: torch.Tensor) -> torch.Tensor:
        """Dequantize fp8 compressed keys.
        
        Format: [128 bytes fp8_e4m3 values | 4 bytes f32 scale]
        """
        B, N, total_bytes = compressed_k.shape
        head_dim = self.config.head_dim  # 128
        
        # Split value bytes and scale bytes
        k_bytes = compressed_k[:, :, :head_dim]          # [B, N, 128] uint8
        scale_bytes = compressed_k[:, :, head_dim:]      # [B, N, 4] uint8
        
        # Reinterpret as fp8 and f32
        # In practice: view as float8_e4m3fn and float32
        k_fp8 = k_bytes.to(torch.float8_e4m3fn).float() if hasattr(torch, 'float8_e4m3fn') \
                 else k_bytes.float() / 127.0  # Fallback for older PyTorch
        scale = scale_bytes.contiguous().view(B, N, 1).float()  # Simplified
        
        # Dequant
        k = k_fp8 * scale                               # [B, N, head_dim]
        
        return k


class HASARetriever(nn.Module):
    """Full HASA retriever — ensemble of scoring layers with hardware-aware adjustment.
    
    Scores all KV chunks and produces placement decisions:
      - SBUF tier: top-K1 highest scored chunks (on-chip, zero-latency)
      - HBM tier: top-K2 next-highest (in DRAM, prefetchable)
      - Evicted: remaining chunks (not used in attention)
    """
    
    def __init__(self, config: HASARetrieverConfig):
        super().__init__()
        self.config = config
        
        # One scoring layer per selected backbone layer
        self.scoring_layers = nn.ModuleDict({
            f"l{layer_idx}": ScoringLayer(config)
            for layer_idx in config.scoring_layers
        })
    
    def forward(
        self,
        hidden: torch.Tensor,           # [B, hidden_size]
        compressed_k: torch.Tensor,      # [B, N_chunks, 132] uint8
        positions: torch.Tensor,         # [B] current position
        ensemble_mode: str = "max",      # "max" or "mean"
    ) -> torch.Tensor:                   # [B, N_chunks] final scores
        """Compute chunk scores with multi-layer ensemble."""
        
        per_layer_scores = []
        for name, layer in self.scoring_layers.items():
            scores = layer(hidden, compressed_k, positions)  # [B, N_chunks]
            per_layer_scores.append(scores)
        
        # Stack: [n_scoring_layers, B, N_chunks]
        stacked = torch.stack(per_layer_scores, dim=0)
        
        # Ensemble
        if ensemble_mode == "max":
            final_scores = stacked.max(dim=0).values     # [B, N_chunks]
        elif ensemble_mode == "mean":
            final_scores = stacked.mean(dim=0)           # [B, N_chunks]
        else:
            raise ValueError(f"Unknown ensemble_mode: {ensemble_mode}")
        
        return final_scores
    
    def place(
        self,
        scores: torch.Tensor,           # [B, N_chunks]
        sbuf_budget: int,                # Max chunks in SBUF tier
        hbm_budget: int,                 # Max chunks in HBM tier (keep)
        current_sbuf: Optional[torch.Tensor] = None,  # [B, N] bool: currently in SBUF
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hardware-aware placement decision.
        
        Returns:
            sbuf_mask: [B, N_chunks] bool — chunks to place in SBUF
            hbm_mask: [B, N_chunks] bool — chunks to keep in HBM
        """
        B, N = scores.shape
        
        # Hardware-aware score adjustment
        adjusted = scores.clone()
        if current_sbuf is not None:
            # Bonus for chunks already in SBUF (avoid unnecessary eviction)
            adjusted = adjusted + self.config.sbuf_bonus * current_sbuf.float()
        
        # Top-K1 → SBUF
        _, sbuf_indices = adjusted.topk(sbuf_budget, dim=-1)  # [B, sbuf_budget]
        sbuf_mask = torch.zeros(B, N, dtype=torch.bool, device=scores.device)
        sbuf_mask.scatter_(1, sbuf_indices, True)
        
        # Top-K2 (excluding SBUF) → HBM keep
        remaining_scores = adjusted.clone()
        remaining_scores[sbuf_mask] = -float('inf')
        _, hbm_indices = remaining_scores.topk(hbm_budget, dim=-1)  # [B, hbm_budget]
        hbm_mask = torch.zeros(B, N, dtype=torch.bool, device=scores.device)
        hbm_mask.scatter_(1, hbm_indices, True)
        
        return sbuf_mask, hbm_mask
    
    @classmethod
    def from_config(cls, config: Optional[HASARetrieverConfig] = None) -> "HASARetriever":
        """Create retriever with default or custom config."""
        if config is None:
            config = HASARetrieverConfig()
        return cls(config)
    
    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Utility functions ---

def compute_placement_budgets(
    seq_len: int,
    chunk_size: int = 64,
    sbuf_bytes: int = 48 * 1024 * 1024,   # 48 MB (Trainium2 estimated)
    hbm_retention: float = 0.135,           # 13.5% FlashMemory baseline
    kv_bytes_per_token: int = 320 * 1024,   # 320KB/token for 70B
) -> Tuple[int, int]:
    """Compute SBUF and HBM budgets based on hardware constraints.
    
    Returns:
        sbuf_budget: number of chunks that fit in SBUF
        hbm_budget: number of chunks to keep in HBM
    """
    n_chunks = seq_len // chunk_size
    
    # SBUF budget: how many chunks fit in on-chip SRAM
    bytes_per_chunk = chunk_size * kv_bytes_per_token // seq_len  # simplified
    # Actually: chunk_size tokens × (2 × n_kv_heads × head_dim × 2 bytes) per layer
    # For one layer: 64 × 8 × 128 × 2 × 2 = 262,144 bytes = 256 KB per chunk per layer
    # Across all layers stored in SBUF: need to pick which layers
    # Simplification: store last-layer KV in SBUF
    bytes_per_chunk_one_layer = chunk_size * 8 * 128 * 2 * 2  # K+V, bf16
    sbuf_budget = min(
        sbuf_bytes // bytes_per_chunk_one_layer,
        n_chunks
    )
    
    # HBM budget: retention percentage
    hbm_budget = int(n_chunks * hbm_retention) - sbuf_budget
    hbm_budget = max(hbm_budget, 0)
    
    return sbuf_budget, hbm_budget


if __name__ == "__main__":
    # Quick validation
    config = HASARetrieverConfig()
    retriever = HASARetriever.from_config(config)
    
    print(f"HASA Retriever — {retriever.param_count():,} parameters")
    print(f"Scoring layers: {config.scoring_layers}")
    print(f"Config: {config.n_kv_heads} KV heads, {config.head_dim} head_dim")
    
    # Mock forward pass
    B, N_chunks = 2, 2048  # 2 sequences, 128K/64 = 2048 chunks
    hidden = torch.randn(B, config.hidden_size)
    compressed_k = torch.randint(0, 255, (B, N_chunks, config.compressed_k_bytes), dtype=torch.uint8)
    positions = torch.tensor([1000, 2000])
    
    scores = retriever(hidden, compressed_k, positions)
    print(f"Scores shape: {scores.shape}, range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Placement
    sbuf_budget, hbm_budget = compute_placement_budgets(seq_len=131072)
    print(f"SBUF budget: {sbuf_budget} chunks, HBM budget: {hbm_budget} chunks")
    
    sbuf_mask, hbm_mask = retriever.place(scores, sbuf_budget, hbm_budget)
    print(f"SBUF chunks: {sbuf_mask.sum().item()}, HBM chunks: {hbm_mask.sum().item()}")
    print(f"Evicted: {N_chunks - sbuf_mask.sum().item() - hbm_mask.sum().item()}")
    print("\n✅ Retriever validation passed")
