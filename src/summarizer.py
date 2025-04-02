"""
summarizer.py
-------------
Multilingual summarization using mBART-large with hierarchical chunking.
Handles documents of any length via a chunk → summarize → merge → compress strategy.
"""

import torch
import warnings
from typing import List, Dict, Optional
warnings.filterwarnings("ignore")

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


# ── mBART language code map ──────────────────────────────────────────────────
MBART_LANG_CODES: Dict[str, str] = {
    "en": "en_XX", "ar": "ar_AR", "fr": "fr_XX", "es": "es_XX",
    "de": "de_DE", "zh": "zh_CN", "ja": "ja_XX", "ru": "ru_RU",
    "pt": "pt_XX", "it": "it_IT", "nl": "nl_XX", "tr": "tr_TR",
    "ko": "ko_KR", "hi": "hi_IN", "fi": "fi_FI", "cs": "cs_CZ",
}

# Default to en_XX if lang not in our map
def _get_mbart_code(lang_code: str) -> str:
    return MBART_LANG_CODES.get(lang_code.lower(), "en_XX")


class MultilingualSummarizer:
    """
    Wraps facebook/mbart-large-cc25 for multilingual extractive-abstractive summarization.

    Strategy:
      1. Sentence-boundary-aware chunking (respects meaning units)
      2. Per-chunk summarization with beam search
      3. Partial summaries merged
      4. Optional final compression pass if merged text is still long
    """

    MODEL_NAME = "facebook/mbart-large-cc25"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Summarizer] Loading {self.MODEL_NAME} on {self.device}...")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.MODEL_NAME)
        self.model     = MBartForConditionalGeneration.from_pretrained(self.MODEL_NAME)
        self.model     = self.model.to(self.device)
        self.model.eval()
        print("[Summarizer] Model ready ✓")

    # ── Chunking ─────────────────────────────────────────────────────────────
    @staticmethod
    def chunk_text(text: str, max_words_per_chunk: int = 350) -> List[str]:
        """
        Split text into sentence-boundary-respecting chunks.
        Never cuts in the middle of a sentence.
        """
        sentences = nltk.sent_tokenize(text)
        chunks, current, cur_len = [], [], 0

        for sent in sentences:
            word_count = len(sent.split())
            if cur_len + word_count > max_words_per_chunk and current:
                chunks.append(" ".join(current))
                current, cur_len = [], 0
            current.append(sent)
            cur_len += word_count

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ── Single-chunk summarization ────────────────────────────────────────────
    def _summarize_chunk(
        self,
        text: str,
        src_lang_code: str,
        max_summary_len: int = 160,
        min_summary_len: int  = 40,
        num_beams: int         = 4,
    ) -> str:
        """Summarize a single text chunk using mBART beam search."""
        mbart_src = _get_mbart_code(src_lang_code)
        self.tokenizer.src_lang = mbart_src

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        forced_bos = self.tokenizer.lang_code_to_id[mbart_src]

        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=max_summary_len,
                min_length=min_summary_len,
                num_beams=num_beams,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # ── Hierarchical summarization ────────────────────────────────────────────
    def summarize(
        self,
        text: str,
        src_lang_code: str = "en",
        max_chunk_words: int = 350,
        max_final_len: int   = 200,
        min_final_len: int   = 50,
        verbose: bool        = True,
    ) -> Dict:
        """
        Full hierarchical summarization pipeline.

        Returns
        -------
        dict with keys:
          - original_word_count   : int
          - num_chunks            : int
          - chunks                : List[str]
          - partial_summaries     : List[str]
          - merged_summary        : str
          - final_summary         : str
          - compression_ratio     : float  (final words / original words)
          - src_lang_code         : str
        """
        word_count = len(text.split())
        if verbose:
            print(f"\n[Summarizer] Input: {word_count} words | lang=[{src_lang_code}]")

        # Step 1 — Chunk
        chunks = self.chunk_text(text, max_chunk_words)
        if verbose:
            print(f"[Summarizer] Split into {len(chunks)} chunk(s)")

        # Step 2 — Summarize each chunk
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"[Summarizer] Summarizing chunk {i + 1}/{len(chunks)} ...")
            summary = self._summarize_chunk(chunk, src_lang_code)
            partial_summaries.append(summary)

        # Step 3 — Merge partial summaries
        merged = " ".join(partial_summaries)

        # Step 4 — Final compression pass if still too long
        if len(merged.split()) > 180 and len(chunks) > 1:
            if verbose:
                print("[Summarizer] Running final compression pass...")
            final_summary = self._summarize_chunk(
                merged, src_lang_code,
                max_summary_len=max_final_len,
                min_summary_len=min_final_len,
            )
        else:
            final_summary = merged

        compression_ratio = round(len(final_summary.split()) / max(word_count, 1), 3)
        if verbose:
            print(f"[Summarizer] Done — {len(final_summary.split())} words "
                  f"(compression: {compression_ratio})")

        return {
            "original_word_count": word_count,
            "num_chunks":          len(chunks),
            "chunks":              chunks,
            "partial_summaries":   partial_summaries,
            "merged_summary":      merged,
            "final_summary":       final_summary,
            "compression_ratio":   compression_ratio,
            "src_lang_code":       src_lang_code,
        }
