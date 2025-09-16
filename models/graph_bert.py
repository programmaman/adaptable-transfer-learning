# graph_bert.py
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional additive attention bias.
    - Accepts: x [B,S,d], key_padding_mask [B,S] (True for valid tokens), attn_bias [B,H,S,S] (additive).
    - Returns: context [B,S,d]
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                    # [B,S,d]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B,S] bool; True = valid token
        attn_bias: Optional[torch.Tensor] = None          # [B,H,S,S] additive bias
    ) -> torch.Tensor:
        B, S, d = x.shape
        H, Dh = self.n_heads, self.d_head

        # Projections
        Q = self.q_proj(x)  # [B,S,d]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to heads
        Q = Q.view(B, S, H, Dh).transpose(1, 2)  # [B,H,S,Dh]
        K = K.view(B, S, H, Dh).transpose(1, 2)  # [B,H,S,Dh]
        V = V.view(B, S, H, Dh).transpose(1, 2)  # [B,H,S,Dh]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Dh ** 0.5)  # [B,H,S,S]

        # Additive attention bias (e.g., SPD buckets, edge bias)
        if attn_bias is not None:
            # Expect [B,H,S,S] (or broadcastable)
            scores = scores + attn_bias

        # Key padding mask: mask out padded keys by adding -inf to their columns
        if key_padding_mask is not None:
            # key_padding_mask: True for valid, False for pad
            # We want to mask pads -> set scores[..., :, pad_positions] = -inf
            pad_mask = ~key_padding_mask  # True where pad
            # shape to [B,1,1,S] to broadcast across heads and query positions
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B,H,S,S]
        attn = self.attn_drop(attn)

        context = torch.matmul(attn, V)   # [B,H,S,Dh]
        context = context.transpose(1, 2).contiguous().view(B, S, d)  # [B,S,d]
        context = self.o_proj(context)
        context = self.proj_drop(context)
        return context


class TransformerEncoderLayer(nn.Module):
    """
    Pre-LN Transformer encoder layer with BiasSelfAttention.
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BiasSelfAttention(d_model, n_heads, dropout=dropout)
        self.drop_path1 = nn.Dropout(dropout)

        hidden_dim = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                       # [B,S,d]
        key_padding_mask: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.norm1(x)
        h = self.attn(h, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        x = x + self.drop_path1(h)

        # FFN with pre-norm
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class GraphBERT(nn.Module):
    """
    GraphBERT: Transformer over subgraph 'token' sets with structure-aware embeddings and optional attention bias.

    Forward expects precomputed, padded mini-batches of subgraphs:
      - x:          [B, S, feat_dim] float   (node/token features)
      - mask:       [B, S] bool              (True=valid token, False=pad)
      - role_ids:   [B, S] long              (discrete structural role buckets; e.g., degree bins)
      - attn_bias:  [B, H, S, S] float or None  (additive attention bias: SPD buckets, edge bias, etc.)
      - anchor_pos: [B, S, A] long or None   (bucketed distances to A anchors; summed embedding)
      - center_idx: [B] long or None         (index of center token per subgraph)
      - return_token_repr: bool (optional)

    Returns:
      - logits: [B, num_classes] (classification on pooled representation—center token if provided, else masked mean)
      - if return_token_repr=True: dict with {"logits", "token_repr", "pooled"}
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        *,
        num_role_buckets: int = 16,
        num_dist_buckets: int = 6,   # e.g., SPD buckets 0..5+ (clip to 5)
        use_anchor_pos: bool = False,
        num_anchors: int = 8,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.use_anchor_pos = use_anchor_pos
        self.num_anchors = num_anchors
        self.num_dist_buckets = num_dist_buckets

        # Feature projection to model dimension
        self.feat_proj = nn.Linear(feat_dim, d_model)

        # Structural role embedding (e.g., degree bins, k-core bins)
        self.role_emb = nn.Embedding(num_role_buckets, d_model)

        # Optional anchor-distance embedding: summed across anchors
        if use_anchor_pos:
            self.anchor_emb = nn.Embedding(num_dist_buckets, d_model)
        else:
            self.anchor_emb = None

        # Encoder stack
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    # ----- helper methods -----

    def _sum_embeddings(
        self,
        x_proj: torch.Tensor,          # [B,S,d]
        role_ids: torch.Tensor,        # [B,S]
        anchor_pos: Optional[torch.Tensor]  # [B,S,A] or None
    ) -> torch.Tensor:
        h = x_proj + self.role_emb(role_ids)  # [B,S,d]
        if self.anchor_emb is not None and anchor_pos is not None:
            # Sum anchor-distance embeddings across anchors
            # anchor_pos[b,s,a] ∈ [0, num_dist_buckets)
            B, S, A = anchor_pos.shape
            anchor_flat = anchor_pos.view(B * S * A)
            anchor_e = self.anchor_emb(anchor_flat).view(B, S, A, -1)  # [B,S,A,d]
            anchor_sum = anchor_e.sum(dim=2)  # [B,S,d]
            h = h + anchor_sum
        return h

    def _encode(
        self,
        h: torch.Tensor,               # [B,S,d]
        mask: torch.Tensor,            # [B,S] bool (True=valid)
        attn_bias: Optional[torch.Tensor]  # [B,H,S,S] or broadcastable, or None
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, key_padding_mask=mask, attn_bias=attn_bias)
        h = self.final_norm(h)
        return h  # [B,S,d]

    @staticmethod
    def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B,S,d], mask: [B,S] True=valid
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)  # [B,1,1]
        h_masked = h * mask.unsqueeze(-1)                                  # [B,S,d]
        return h_masked.sum(dim=1, keepdim=False) / denom.squeeze(-1)      # [B,d]

    @staticmethod
    def _gather_center(h: torch.Tensor, center_idx: torch.Tensor) -> torch.Tensor:
        # h: [B,S,d], center_idx: [B]
        B, S, d = h.shape
        idx = center_idx.view(B, 1, 1).expand(B, 1, d)  # [B,1,d]
        return h.gather(dim=1, index=idx).squeeze(1)    # [B,d]

    # ----- public forward -----

    def forward(
        self,
        x: torch.Tensor,                          # [B,S,feat_dim]
        mask: torch.Tensor,                       # [B,S] bool (True=valid)
        role_ids: torch.Tensor,                   # [B,S] long
        attn_bias: Optional[torch.Tensor] = None, # [B,H,S,S] float or None
        anchor_pos: Optional[torch.Tensor] = None,# [B,S,A] long or None
        center_idx: Optional[torch.Tensor] = None,# [B] long or None
        return_token_repr: bool = False
    ) -> torch.Tensor | Dict[str, Any]:
        """
        Model-only forward. All graph-derived encodings must be precomputed outside.
        """
        # Project features and add structure-aware embeddings
        x_proj = self.feat_proj(x)                         # [B,S,d]
        h = self._sum_embeddings(x_proj, role_ids, anchor_pos)  # [B,S,d]

        # Encode with Transformer + optional attention bias
        token_repr = self._encode(h, mask=mask, attn_bias=attn_bias)  # [B,S,d]

        # Pool: center token if provided, else masked mean
        if center_idx is not None:
            pooled = self._gather_center(token_repr, center_idx)       # [B,d]
        else:
            pooled = self._masked_mean(token_repr, mask)               # [B,d]

        logits = self.head(pooled)                                      # [B,C]

        if return_token_repr:
            return {
                "logits": logits,
                "token_repr": token_repr,
                "pooled": pooled
            }
        return logits


# ---------- Minimal shape sanity test (optional) ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, F = 2, 16, 8
    d_model, H, L, C = 128, 4, 3, 5

    model = GraphBERT(
        feat_dim=F,
        num_classes=C,
        d_model=d_model,
        n_heads=H,
        n_layers=L,
        dropout=0.1,
        num_role_buckets=16,
        num_dist_buckets=6,
        use_anchor_pos=True,
        num_anchors=8,
    )

    x = torch.randn(B, S, F)
    mask = torch.ones(B, S, dtype=torch.bool)
    role_ids = torch.randint(0, 16, (B, S))
    # Example: zero bias to start; later, add SPD/edge biases
    attn_bias = torch.zeros(B, H, S, S)
    anchor_pos = torch.randint(0, 6, (B, S, 8))
    center_idx = torch.zeros(B, dtype=torch.long)

    out = model(
        x=x,
        mask=mask,
        role_ids=role_ids,
        attn_bias=attn_bias,
        anchor_pos=anchor_pos,
        center_idx=center_idx,
        return_token_repr=True
    )
    print(out["logits"].shape, out["token_repr"].shape, out["pooled"].shape)  # [B,C], [B,S,d], [B,d]
