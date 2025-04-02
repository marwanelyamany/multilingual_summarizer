"""
utils.py
--------
Utility functions:
  - Language auto-detection
  - BLEU / ROUGE evaluation
  - Pretty-printing results
  - Text preprocessing helpers
"""

import re
import math
from typing import List, Dict, Optional
from collections import Counter


# ──────────────────────────────────────────────────────────────
#  LANGUAGE DETECTION
# ──────────────────────────────────────────────────────────────
def detect_language(text: str) -> Dict[str, str]:
    """
    Auto-detect the language of a text using langdetect.
    Returns ISO 639-1 code and a confidence label.
    """
    try:
        from langdetect import detect, detect_langs
        code  = detect(text[:1000])          # sample first 1000 chars
        langs = detect_langs(text[:1000])    # with probabilities
        prob  = next((l.prob for l in langs if l.lang == code), 0.0)
        return {
            "code":       code,
            "confidence": round(prob, 3),
            "label":      _code_to_name(code),
        }
    except Exception as e:
        print(f"[LangDetect] Detection failed: {e}. Defaulting to English.")
        return {"code": "en", "confidence": 0.0, "label": "English"}


def _code_to_name(code: str) -> str:
    CODE_TO_NAME = {
        "en": "English",  "ar": "Arabic",   "fr": "French",
        "es": "Spanish",  "de": "German",   "zh": "Chinese",
        "ja": "Japanese", "ru": "Russian",  "pt": "Portuguese",
        "it": "Italian",  "nl": "Dutch",    "tr": "Turkish",
        "ko": "Korean",   "hi": "Hindi",
    }
    return CODE_TO_NAME.get(code, code.upper())


# ──────────────────────────────────────────────────────────────
#  BLEU SCORE (sentence-level, no dependencies)
# ──────────────────────────────────────────────────────────────
def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> Dict:
    """
    Compute sentence-level BLEU score (no sacrebleu dependency needed).
    Uses modified n-gram precision with brevity penalty.
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens:
        return {"bleu": 0.0, "precisions": [], "brevity_penalty": 0.0}

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hyp_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        # Clipped precision
        clipped  = sum(min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items())
        total    = sum(hyp_ngrams.values())
        precisions.append(clipped / total if total else 0.0)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        log_avg  = sum(math.log(p) for p in precisions) / len(precisions)
        geo_mean = math.exp(log_avg)

    # Brevity penalty
    bp = (
        1.0 if len(hyp_tokens) >= len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    )

    return {
        "bleu":            round(geo_mean * bp * 100, 2),
        "precisions":      [round(p * 100, 2) for p in precisions],
        "brevity_penalty": round(bp, 4),
        "hypothesis_len":  len(hyp_tokens),
        "reference_len":   len(ref_tokens),
    }


# ──────────────────────────────────────────────────────────────
#  ROUGE-L  (longest common subsequence)
# ──────────────────────────────────────────────────────────────
def rouge_l(hypothesis: str, reference: str) -> Dict:
    """Compute ROUGE-L F1 score based on longest common subsequence."""
    def lcs_length(a: List, b: List) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    lcs = lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0
    recall    = lcs / len(ref_tokens) if ref_tokens else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "rouge_l_f1":        round(f1, 4),
        "rouge_l_precision": round(precision, 4),
        "rouge_l_recall":    round(recall, 4),
        "lcs_length":        lcs,
    }


# ──────────────────────────────────────────────────────────────
#  TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Basic text cleanup: normalize whitespace, remove control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    text = re.sub(r"\s+", " ", text)                                  # collapse whitespace
    text = text.strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    try:
        import nltk
        return len(nltk.sent_tokenize(text))
    except Exception:
        return text.count(".") + text.count("!") + text.count("?")


# ──────────────────────────────────────────────────────────────
#  PRETTY PRINTING
# ──────────────────────────────────────────────────────────────
def print_pipeline_result(result: Dict) -> None:
    """Pretty-print the full pipeline output."""
    sep = "=" * 65

    print(f"\n{sep}")
    print("  MULTILINGUAL SUMMARIZATION & TRANSLATION — RESULTS")
    print(sep)

    print(f"\n  Source language    : {result.get('detected_language', {}).get('label', 'Unknown')}"
          f" [{result.get('detected_language', {}).get('code', '?')}]"
          f"  (confidence: {result.get('detected_language', {}).get('confidence', 0):.1%})")
    print(f"  Target language    : {result.get('target_language', 'N/A')}")
    print(f"  Original length    : {result.get('summarization', {}).get('original_word_count', '?')} words")
    print(f"  Summary length     : {word_count(result.get('summarization', {}).get('final_summary', ''))} words")
    print(f"  Compression ratio  : {result.get('summarization', {}).get('compression_ratio', '?')}")
    print(f"  Translation strat. : {result.get('translation', {}).get('strategy', 'N/A')}")

    print(f"\n{'─' * 65}")
    print("  SUMMARY (source language)")
    print(f"{'─' * 65}")
    print(f"\n  {result.get('summarization', {}).get('final_summary', '[No summary generated]')}\n")

    print(f"{'─' * 65}")
    print(f"  TRANSLATED SUMMARY ({result.get('target_language', '')})")
    print(f"{'─' * 65}")
    print(f"\n  {result.get('translation', {}).get('translated_text', '[No translation generated]')}\n")

    if "evaluation" in result:
        ev = result["evaluation"]
        print(f"{'─' * 65}")
        print("  EVALUATION METRICS")
        print(f"{'─' * 65}")
        print(f"  BLEU score   : {ev.get('bleu', {}).get('bleu', 'N/A')}")
        print(f"  ROUGE-L F1   : {ev.get('rouge_l', {}).get('rouge_l_f1', 'N/A')}")

    print(f"\n{sep}\n")
