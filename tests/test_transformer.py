"""
test_transformer.py
-------------------
Unit tests for all core Transformer components.
Run with: pytest tests/ -v
"""

import math
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import (
    PositionalEncoding,
    MultiHeadAttention,
    FeedForwardNetwork,
    EncoderLayer,
    DecoderLayer,
    Seq2SeqTransformer,
)


# ──────────────────────────────────────────────────────────────
#  POSITIONAL ENCODING
# ──────────────────────────────────────────────────────────────
class TestPositionalEncoding:
    def test_output_shape(self):
        pe  = PositionalEncoding(d_model=64, max_len=100)
        x   = torch.zeros(2, 10, 64)
        out = pe(x)
        assert out.shape == (2, 10, 64), "Output shape mismatch"

    def test_not_all_zeros_after_encoding(self):
        pe  = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x   = torch.zeros(1, 5, 64)
        out = pe(x)
        assert not torch.all(out == 0), "PE should inject non-zero values"

    def test_different_positions_different_encodings(self):
        pe  = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        pe_matrix = pe.pe.squeeze(0)  # (max_len, d_model)
        assert not torch.allclose(pe_matrix[0], pe_matrix[1]), \
            "Adjacent positions should have different encodings"


# ──────────────────────────────────────────────────────────────
#  MULTI-HEAD ATTENTION
# ──────────────────────────────────────────────────────────────
class TestMultiHeadAttention:
    def test_output_shape(self):
        mha    = MultiHeadAttention(d_model=64, num_heads=8)
        x      = torch.randn(2, 10, 64)
        output = mha(x, x, x)
        assert output.shape == (2, 10, 64), "Attention output shape mismatch"

    def test_invalid_heads_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=8)  # 65 not divisible by 8

    def test_attention_with_mask(self):
        mha  = MultiHeadAttention(d_model=32, num_heads=4)
        x    = torch.randn(1, 6, 32)
        mask = torch.ones(1, 1, 1, 6, dtype=torch.bool)
        mask[0, 0, 0, 3:] = 0  # mask last 3 positions
        out = mha(x, x, x, mask)
        assert out.shape == (1, 6, 32)

    def test_attn_weights_stored(self):
        mha = MultiHeadAttention(d_model=32, num_heads=4)
        x   = torch.randn(1, 5, 32)
        _   = mha(x, x, x)
        assert hasattr(mha, "attn_weights"), "Attention weights should be stored"
        assert mha.attn_weights.shape == (1, 4, 5, 5)


# ──────────────────────────────────────────────────────────────
#  FEED-FORWARD NETWORK
# ──────────────────────────────────────────────────────────────
class TestFeedForwardNetwork:
    def test_output_shape(self):
        ffn = FeedForwardNetwork(d_model=64, d_ff=256)
        x   = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_learns_nonlinearity(self):
        ffn = FeedForwardNetwork(d_model=16, d_ff=64, dropout=0.0)
        ffn.eval()
        x1 = torch.zeros(1, 1, 16)
        x2 = torch.ones(1, 1, 16)
        assert not torch.allclose(ffn(x1), ffn(x2))


# ──────────────────────────────────────────────────────────────
#  ENCODER LAYER
# ──────────────────────────────────────────────────────────────
class TestEncoderLayer:
    def test_output_shape(self):
        layer = EncoderLayer(d_model=64, num_heads=8, d_ff=256)
        x     = torch.randn(2, 12, 64)
        out   = layer(x)
        assert out.shape == (2, 12, 64)

    def test_with_padding_mask(self):
        layer = EncoderLayer(d_model=32, num_heads=4, d_ff=128)
        x    = torch.randn(2, 8, 32)
        mask = torch.ones(2, 1, 1, 8, dtype=torch.bool)
        out  = layer(x, src_mask=mask)
        assert out.shape == (2, 8, 32)


# ──────────────────────────────────────────────────────────────
#  DECODER LAYER
# ──────────────────────────────────────────────────────────────
class TestDecoderLayer:
    def test_output_shape(self):
        layer      = DecoderLayer(d_model=64, num_heads=8, d_ff=256)
        tgt        = torch.randn(2, 5, 64)
        enc_output = torch.randn(2, 10, 64)
        out        = layer(tgt, enc_output)
        assert out.shape == (2, 5, 64)


# ──────────────────────────────────────────────────────────────
#  FULL SEQ2SEQ TRANSFORMER
# ──────────────────────────────────────────────────────────────
class TestSeq2SeqTransformer:
    @pytest.fixture
    def small_model(self):
        return Seq2SeqTransformer(
            src_vocab_size=200, tgt_vocab_size=200,
            d_model=32, num_heads=4,
            num_encoder_layers=2, num_decoder_layers=2,
            d_ff=64, max_len=50
        )

    def test_forward_output_shape(self, small_model):
        src    = torch.randint(1, 200, (2, 10))
        tgt    = torch.randint(1, 200, (2, 7))
        logits = small_model(src, tgt)
        assert logits.shape == (2, 7, 200), "Output logits shape mismatch"

    def test_src_mask_shape(self, small_model):
        src  = torch.tensor([[1, 2, 3, 0, 0]])  # last 2 are padding
        mask = small_model.make_src_mask(src)
        assert mask.shape == (1, 1, 1, 5)
        assert mask[0, 0, 0, 3] == False  # padding should be masked

    def test_tgt_mask_is_causal(self, small_model):
        tgt  = torch.tensor([[1, 2, 3, 4]])
        mask = small_model.make_tgt_mask(tgt)
        assert mask.shape == (1, 1, 4, 4)
        # Upper triangle should be False (future positions masked)
        assert mask[0, 0, 0, 1] == False  # position 0 cannot attend to position 1

    def test_parameter_count(self, small_model):
        count = small_model.count_parameters()
        assert count > 0

    def test_greedy_decode(self, small_model):
        small_model.eval()
        src = torch.randint(1, 200, (1, 8))
        out = small_model.greedy_decode(src, bos_idx=1, eos_idx=2, max_len=20)
        assert out.size(0) == 1     # batch dimension preserved
        assert out[0, 0] == 1       # first token should be BOS

    def test_weight_initialization(self, small_model):
        for name, param in small_model.named_parameters():
            if param.dim() > 1:
                # Xavier uniform: values should not be extreme
                assert param.abs().max() < 10.0, f"Extreme weights in {name}"


# ──────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────
class TestUtils:
    def test_bleu_identical(self):
        from src.utils import bleu_score
        result = bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert result["bleu"] > 90.0, "Identical strings should have high BLEU"

    def test_bleu_completely_different(self):
        from src.utils import bleu_score
        result = bleu_score("apple banana cherry", "dog elephant fox")
        assert result["bleu"] == 0.0

    def test_rouge_l_identical(self):
        from src.utils import rouge_l
        result = rouge_l("the quick brown fox", "the quick brown fox")
        assert result["rouge_l_f1"] == 1.0

    def test_rouge_l_partial(self):
        from src.utils import rouge_l
        result = rouge_l("the quick fox", "the quick brown fox")
        assert 0.0 < result["rouge_l_f1"] < 1.0

    def test_clean_text(self):
        from src.utils import clean_text
        dirty = "Hello   \t world\x00\x01"
        clean = clean_text(dirty)
        assert "\t" not in clean
        assert "\x00" not in clean
        assert "Hello" in clean


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
