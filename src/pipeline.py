"""
pipeline.py
-----------
Main orchestrator for the Multilingual Summarization & Translation Pipeline.

Flow:
  Input text
    → Language auto-detection
    → Hierarchical summarization (mBART)
    → Neural translation (Helsinki-NLP MarianMT)
    → Evaluation (BLEU, ROUGE-L)
    → Structured results dict
"""

import os
import json
import torch
from typing import Dict, Optional
from pathlib import Path

from src.summarizer import MultilingualSummarizer
from src.translator  import Translator, LANGUAGE_NAME_TO_CODE
from src.utils       import detect_language, bleu_score, rouge_l, clean_text, print_pipeline_result


class MultilingualPipeline:
    """
    End-to-end pipeline: auto-detect → summarize → translate → evaluate.

    Parameters
    ----------
    device : str, optional
        'cuda' or 'cpu'. Auto-detected if not specified.
    lazy_load : bool
        If True, models load on first use (saves memory if only using one component).
    """

    def __init__(self, device: Optional[str] = None, lazy_load: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Running on: {self.device}")

        if lazy_load:
            self._summarizer = None
            self._translator = None
        else:
            self._summarizer = MultilingualSummarizer(self.device)
            self._translator = Translator(self.device)

    # ── Lazy getters ───────────────────────────────────────────────────────
    @property
    def summarizer(self) -> MultilingualSummarizer:
        if self._summarizer is None:
            self._summarizer = MultilingualSummarizer(self.device)
        return self._summarizer

    @property
    def translator(self) -> Translator:
        if self._translator is None:
            self._translator = Translator(self.device)
        return self._translator

    # ── Core pipeline ──────────────────────────────────────────────────────
    def run(
        self,
        text: str,
        target_language: str = "english",
        src_lang_override: Optional[str] = None,
        reference_summary: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full pipeline on a piece of text.

        Parameters
        ----------
        text : str
            Input document (any length, any supported language).
        target_language : str
            Language name or ISO code for the output translation.
        src_lang_override : str, optional
            Force a source language instead of auto-detecting.
        reference_summary : str, optional
            If provided, compute BLEU & ROUGE-L against this reference.
        verbose : bool
            Print step-by-step progress.

        Returns
        -------
        dict — full result with keys:
            detected_language, target_language, summarization,
            translation, evaluation (if reference provided)
        """
        text = clean_text(text)
        if not text:
            raise ValueError("Input text is empty after cleaning.")

        # ── Step 1: Language detection ─────────────────────────────────────
        if src_lang_override:
            lang_code = LANGUAGE_NAME_TO_CODE.get(src_lang_override.lower(), src_lang_override)
            detected  = {"code": lang_code, "confidence": 1.0, "label": src_lang_override}
        else:
            if verbose:
                print("\n[Pipeline] Step 1/3 — Detecting source language...")
            detected  = detect_language(text)
            lang_code = detected["code"]
            if verbose:
                print(f"[Pipeline] Detected: {detected['label']} "
                      f"({detected['confidence']:.1%} confidence)")

        # ── Step 2: Hierarchical summarization ─────────────────────────────
        if verbose:
            print("\n[Pipeline] Step 2/3 — Summarizing...")
        summarization_result = self.summarizer.summarize(
            text=text,
            src_lang_code=lang_code,
            verbose=verbose,
        )

        # ── Step 3: Translate summary ──────────────────────────────────────
        if verbose:
            print(f"\n[Pipeline] Step 3/3 — Translating to [{target_language}]...")
        tgt_code = LANGUAGE_NAME_TO_CODE.get(target_language.lower(), target_language.lower())
        translation_result = self.translator.translate(
            text=summarization_result["final_summary"],
            src_lang=lang_code,
            tgt_lang=tgt_code,
        )

        # ── Assemble result ────────────────────────────────────────────────
        result = {
            "detected_language": detected,
            "target_language":   target_language,
            "summarization":     summarization_result,
            "translation":       translation_result,
        }

        # ── Optional evaluation ────────────────────────────────────────────
        if reference_summary:
            hyp = translation_result["translated_text"]
            result["evaluation"] = {
                "bleu":   bleu_score(hyp, reference_summary),
                "rouge_l": rouge_l(hyp, reference_summary),
            }

        if verbose:
            print_pipeline_result(result)

        return result

    # ── Batch processing ───────────────────────────────────────────────────
    def run_batch(
        self,
        texts: list,
        target_language: str = "english",
        verbose: bool = False,
    ) -> list:
        """Run the pipeline on a list of documents. Returns list of result dicts."""
        results = []
        for i, text in enumerate(texts):
            print(f"\n[Pipeline] Processing document {i + 1}/{len(texts)} ...")
            result = self.run(text, target_language=target_language, verbose=verbose)
            results.append(result)
        return results

    # ── Save / load results ────────────────────────────────────────────────
    @staticmethod
    def save_result(result: Dict, path: str) -> None:
        """Save pipeline result to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Make serializable (remove any tensor objects)
        serializable = json.loads(json.dumps(result, default=str))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"[Pipeline] Result saved to {path}")

    @staticmethod
    def load_result(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
