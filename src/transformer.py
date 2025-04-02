"""
transformer.py
--------------
Full Seq2Seq Transformer built from scratch with PyTorch.
Implements: Multi-Head Attention, Positional Encoding,
            Feed-Forward Networks, Encoder & Decoder stacks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────
#  POSITIONAL ENCODING
# ──────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in 'Attention Is All You Need'.
    Injects position information into token embeddings so the model understands
    token order (transformers have no recurrence).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the sinusoidal position matrix once (not learnable)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices → sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  indices → cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────
#  MULTI-HEAD ATTENTION
# ──────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Attention.
    Runs `num_heads` attention functions in parallel over d_k-dimensional
    subspaces, then concatenates and projects back to d_model.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads  # dimension per head

        # Linear projections for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq, d_model) → (batch, heads, seq, d_k)."""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _scaled_dot_product(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V), attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)

        # Project and split into heads
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        # Attention per head
        x, self.attn_weights = self._scaled_dot_product(Q, K, V, mask)

        # Concatenate heads and project
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(x)


# ──────────────────────────────────────────────────────────────
#  FEED-FORWARD NETWORK
# ──────────────────────────────────────────────────────────────
class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network: two linear layers with GELU activation.
    Applied identically to each position independently.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
#  ENCODER LAYER
# ──────────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer:
      1. Multi-head self-attention
      2. Add & Norm (residual connection)
      3. Feed-forward network
      4. Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention + residual
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ──────────────────────────────────────────────────────────────
#  DECODER LAYER
# ──────────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer:
      1. Masked multi-head self-attention (causal mask prevents future peeking)
      2. Add & Norm
      3. Cross-attention over encoder output
      4. Add & Norm
      5. Feed-forward network
      6. Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Masked self-attention (decoder can only attend to past tokens)
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # 2. Cross-attention: queries from decoder, keys/values from encoder
        attn2 = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


# ──────────────────────────────────────────────────────────────
#  FULL SEQ2SEQ TRANSFORMER
# ──────────────────────────────────────────────────────────────
class Seq2SeqTransformer(nn.Module):
    """
    Complete encoder-decoder Transformer for sequence-to-sequence tasks.
    Can be used for summarization, translation, or any text→text task.

    Architecture follows 'Attention Is All You Need' (Vaswani et al., 2017)
    with GELU activations instead of ReLU.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # Embeddings (shared positional encoding for both sides)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding  = PositionalEncoding(d_model, max_len, dropout)

        # Encoder & Decoder stacks
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )

        # Final projection to vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for all weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Mask builders ──────────────────────────────────────────
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Padding mask: (batch, 1, 1, src_len) — True where token is real."""
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Causal + padding mask: (batch, 1, tgt_len, tgt_len)."""
        tgt_len  = tgt.size(1)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal   = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool)
        )
        return pad_mask & causal

    # ── Encode & Decode (exposed separately for beam search) ───
    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    # ── Full forward pass ──────────────────────────────────────
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.output_projection(dec_output)  # logits: (batch, tgt_len, tgt_vocab)

    # ── Greedy decoding for inference ─────────────────────────
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 200,
    ) -> torch.Tensor:
        """
        Simple greedy decoding: at each step pick the highest-probability token.
        For better quality, use beam_decode() in pipeline.py.
        """
        src_mask   = self.make_src_mask(src)
        enc_output = self.encode(src, src_mask)

        tgt = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(tgt)
            dec_out  = self.decode(tgt, enc_output, src_mask, tgt_mask)
            logits   = self.output_projection(dec_out[:, -1, :])  # last token
            next_tok = logits.argmax(dim=-1, keepdim=True)
            tgt      = torch.cat([tgt, next_tok], dim=1)

            if (next_tok == eos_idx).all():
                break

        return tgt

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"Seq2SeqTransformer("
            f"d_model={self.d_model}, "
            f"params={self.count_parameters():,})"
        )
