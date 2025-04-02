# 🌐 Multilingual Summarization & Translation Pipeline

A production-grade NLP pipeline that auto-detects the language of any input document, summarizes it using a hierarchical chunking strategy with mBART, and translates the summary to a target language using Helsinki-NLP MarianMT — all built on top of a **custom Transformer architecture implemented from scratch in PyTorch**.

---

## ✨ What Makes This Different

| Feature                  | This Project                  | Basic HuggingFace Wrapper  |
| ------------------------ | ----------------------------- | -------------------------- |
| Transformer from scratch | ✅ Full encoder-decoder       | ❌ Black-box               |
| Long doc handling        | ✅ Hierarchical chunking      | ❌ Truncates at 512 tokens |
| Language detection       | ✅ Auto-detect via langdetect | ❌ Manual                  |
| Translation              | ✅ Dynamic model loading      | ❌ Fixed pair              |
| Pivot routing            | ✅ English pivot fallback     | ❌ Fails                   |
| Evaluation               | ✅ BLEU + ROUGE-L             | ❌ None                    |
| CLI                      | ✅ Full argparse interface    | ❌ None                    |

---

## 🏗️ Architecture

```
Input Document (any language, any length)
        │
        ▼
┌─────────────────────┐
│  Language Detection │  langdetect → ISO 639-1 code
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│         Hierarchical Summarization      │
│                                         │
│  Text → [Chunk₁][Chunk₂]...[ChunkN]    │
│            ↓       ↓          ↓        │
│          Sum₁    Sum₂  ...  SumN       │
│            └───────┬───────────┘       │
│                 Merge                  │
│                    ↓                   │
│           Final Compression            │
│         (mBART-large-cc25)             │
└──────────────────┬──────────────────── ┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Neural Translation              │
│                                         │
│   Helsinki-NLP MarianMT                 │
│   src → tgt  (or  src → en → tgt)      │
└──────────────────┬──────────────────── ┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│     Evaluation (optional)               │
│   BLEU-4  ·  ROUGE-L  ·  Length ratio  │
└─────────────────────────────────────────┘
```

### Custom Transformer Components (Built From Scratch)

```
Seq2SeqTransformer
├── PositionalEncoding      — sinusoidal position injection
├── MultiHeadAttention      — parallel scaled dot-product attention
│   └── scaled_dot_product  — Q·Kᵀ/√dₖ · V with masking
├── FeedForwardNetwork      — position-wise MLP with GELU
├── EncoderLayer × N        — self-attn + FFN + residuals + LayerNorm
├── DecoderLayer × N        — masked-self-attn + cross-attn + FFN
└── Output Projection       — linear → vocab logits
```

---

## 📁 Project Structure

```
multilingual_summarization_pipeline/
├── src/
│   ├── __init__.py
│   ├── transformer.py        # Full Seq2Seq Transformer from scratch
│   ├── summarizer.py         # mBART hierarchical summarizer
│   ├── translator.py         # Helsinki-NLP MarianMT translator
│   ├── pipeline.py           # End-to-end orchestrator
│   └── utils.py              # Detection, BLEU, ROUGE-L, helpers
├── notebooks/
│   └── exploration.ipynb     # Interactive walkthrough
├── data/
│   └── sample_texts/
│       ├── english_ai.txt
│       ├── french_climate.txt
│       └── arabic_education.txt
├── tests/
│   └── test_transformer.py   # Unit tests (pytest)
├── configs/
│   └── model_config.yaml     # All hyperparameters
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash

pip install -r requirements.txt
```

### 2. Set up HuggingFace Token (Optional but Recommended)

The models download automatically without a token, but setting one up removes the warning message and gives you faster download speeds.

**Step 1** — Create a free account at [huggingface.co](https://huggingface.co)

**Step 2** — Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → click **New Token** → select **Read** access → click **Generate** → copy the token

**Step 3** — Run this in your terminal:

```bash
hf auth login
```

Paste your token when prompted. Done — the warning will never appear again.

> **Note:** If `hf` command is not found, run `pip install -U huggingface_hub` first.

### 3. Run via CLI

```bash
# Summarize an English file → translate to Arabic
python main.py --file data/sample_texts/english_ai.txt --target arabic

# Summarize French text → translate to English
python main.py --file data/sample_texts/french_climate.txt --target english

# Inline text
python main.py --text "Your long document here..." --target spanish

# Save output to JSON
python main.py --file data/sample_texts/english_ai.txt --target french --save results/output.json

# Evaluate against a reference
python main.py --file data/sample_texts/english_ai.txt \
               --target arabic \
               --reference "يحول الذكاء الاصطناعي الصناعات..."
```

### 4. Use as Python API

```python
from src.pipeline import MultilingualPipeline

pipeline = MultilingualPipeline()

result = pipeline.run(
    text=open("my_document.txt").read(),
    target_language="arabic",
    verbose=True
)

print(result["summarization"]["final_summary"])
print(result["translation"]["translated_text"])
```

### 5. Run the notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 🧠 Custom Transformer Usage

```python
from src.transformer import Seq2SeqTransformer
import torch

model = Seq2SeqTransformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    d_model=256,
    num_heads=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_ff=1024,
)

print(f"Parameters: {model.count_parameters():,}")

# Training forward pass
src    = torch.randint(1, 32000, (4, 50))   # (batch=4, src_len=50)
tgt    = torch.randint(1, 32000, (4, 30))   # (batch=4, tgt_len=30)
logits = model(src, tgt)                     # → (4, 30, 32000)

# Inference
output = model.greedy_decode(src[:1], bos_idx=1, eos_idx=2, max_len=100)
```

---

## 🌍 Supported Languages

| Language | Code | Direct Translation To                                                            |
| -------- | ---- | -------------------------------------------------------------------------------- |
| English  | en   | Arabic, French, Spanish, German, Chinese, Japanese, Russian, Portuguese, Italian |
| Arabic   | ar   | English, French                                                                  |
| French   | fr   | English, Spanish, German, Arabic, Italian                                        |
| Spanish  | es   | English, French, Portuguese                                                      |
| German   | de   | English, French, Russian                                                         |
| Chinese  | zh   | English (pivot for others)                                                       |
| Russian  | ru   | English, German                                                                  |
| Japanese | ja   | English (pivot for others)                                                       |
| Korean   | ko   | English (pivot for others)                                                       |

**All other pairs** are handled automatically via English pivot routing.

---

## 🧪 Tests

```bash
# Run all unit tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📊 Evaluation Metrics

| Metric            | Description                           | Range                   |
| ----------------- | ------------------------------------- | ----------------------- |
| BLEU-4            | 4-gram precision with brevity penalty | 0–100 (higher = better) |
| ROUGE-L           | Longest common subsequence F1         | 0–1 (higher = better)   |
| Compression Ratio | Summary length / original length      | ~0.05–0.3 typical       |

---

## 💡 Key Concepts Demonstrated

- **Positional Encoding** — sinusoidal embeddings inject token order into attention
- **Multi-Head Attention** — parallel attention over d_k-dimensional subspaces
- **Causal Masking** — decoder cannot attend to future tokens during generation
- **Cross-Attention** — decoder queries the encoder's contextualized representations
- **Hierarchical Summarization** — chunk → summarize → merge → compress for unlimited doc length
- **Pivot Translation** — routes through English when a direct language pair model doesn't exist
- **Xavier Initialization** — stable gradient flow at the start of training

---

## 📖 References

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Liu et al. (2020) — [Multilingual Denoising Pre-training for Neural Machine Translation (mBART)](https://arxiv.org/abs/2001.08210)
- Helsinki-NLP — [OPUS-MT Translation Models](https://github.com/Helsinki-NLP/Opus-MT)
- Papineni et al. (2002) — [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)

---

## 📄 License

MIT License — free to use, modify, and distribute.
