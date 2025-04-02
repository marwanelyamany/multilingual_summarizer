"""
translator.py
-------------
Neural machine translation using Helsinki-NLP MarianMT models.
Dynamically loads the correct model for any supported language pair.
Falls back to English-pivot translation when a direct model isn't available.
"""

import torch
import warnings
from typing import Dict, Optional, Tuple
warnings.filterwarnings("ignore")

from transformers import MarianMTModel, MarianTokenizer


# ── Language maps ──────────────────────────────────────────────────────────
LANGUAGE_NAME_TO_CODE: Dict[str, str] = {
    "english":    "en", "arabic":     "ar", "french":     "fr",
    "spanish":    "es", "german":     "de", "chinese":    "zh",
    "japanese":   "ja", "russian":    "ru", "portuguese": "pt",
    "italian":    "it", "dutch":      "nl", "turkish":    "tr",
    "korean":     "ko", "hindi":      "hi", "finnish":    "fi",
    "czech":      "cs", "polish":     "pl", "swedish":    "sv",
    "romanian":   "ro", "ukrainian":  "uk",
}

# Language pairs where we must pivot through English
# (no direct Helsinki-NLP model exists)
PIVOT_PAIRS = {
    ("ar", "fr"), ("ar", "de"), ("ar", "zh"), ("ar", "ja"),
    ("zh", "ar"), ("zh", "fr"), ("zh", "de"), ("ja", "ar"),
    ("ko", "ar"), ("hi", "ar"),
}


class Translator:
    """
    Neural machine translation engine using Helsinki-NLP MarianMT.

    Features:
    - Dynamic model loading per language pair (cached in memory)
    - Automatic English-pivot translation for unsupported direct pairs
    - Batch-aware tokenization with proper padding
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_cache: Dict[str, Tuple[MarianTokenizer, MarianMTModel]] = {}
        print(f"[Translator] Initialized on {self.device} ✓")

    # ── Internal helpers ───────────────────────────────────────────────────
    def _load_model(self, src: str, tgt: str) -> Optional[Tuple]:
        """Load (and cache) a Helsinki-NLP MarianMT model for a language pair."""
        cache_key = f"{src}-{tgt}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        print(f"[Translator] Loading {model_name} ...")
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model     = MarianMTModel.from_pretrained(model_name).to(self.device)
            model.eval()
            self._model_cache[cache_key] = (tokenizer, model)
            print(f"[Translator] {model_name} loaded ✓")
            return tokenizer, model
        except Exception as e:
            print(f"[Translator] Could not load {model_name}: {e}")
            return None

    def _translate_direct(
        self,
        text: str,
        src: str,
        tgt: str,
        max_length: int = 512,
        num_beams: int  = 5,
    ) -> Optional[str]:
        """Attempt direct src→tgt translation. Returns None if model unavailable."""
        result = self._load_model(src, tgt)
        if result is None:
            return None

        tokenizer, model = result
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            translated_ids = model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    # ── Public API ─────────────────────────────────────────────────────────
    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 512,
        num_beams: int  = 5,
    ) -> Dict:
        """
        Translate text from src_lang to tgt_lang.

        Supports both ISO 639-1 codes ('en', 'fr') and
        full language names ('english', 'french').

        Returns
        -------
        dict with keys:
          - translated_text : str
          - src_lang        : str  (ISO code)
          - tgt_lang        : str  (ISO code)
          - strategy        : str  ('direct' | 'pivot_en' | 'no_op')
          - pivot_text      : str | None  (intermediate English, if pivot used)
        """
        # Normalize to ISO codes
        src = LANGUAGE_NAME_TO_CODE.get(src_lang.lower(), src_lang.lower())
        tgt = LANGUAGE_NAME_TO_CODE.get(tgt_lang.lower(), tgt_lang.lower())

        # No-op if same language
        if src == tgt:
            return {
                "translated_text": text,
                "src_lang": src, "tgt_lang": tgt,
                "strategy": "no_op", "pivot_text": None,
            }

        # Try direct translation
        direct = self._translate_direct(text, src, tgt, max_length, num_beams)
        if direct is not None:
            return {
                "translated_text": direct,
                "src_lang": src, "tgt_lang": tgt,
                "strategy": "direct", "pivot_text": None,
            }

        # Pivot through English
        print(f"[Translator] Direct model not found — pivoting through English")

        pivot_text = None
        if src != "en":
            pivot_text = self._translate_direct(text, src, "en", max_length, num_beams)
            if pivot_text is None:
                return {
                    "translated_text": f"[Translation unavailable: {src}→en failed]",
                    "src_lang": src, "tgt_lang": tgt,
                    "strategy": "failed", "pivot_text": None,
                }
        else:
            pivot_text = text

        if tgt != "en":
            final = self._translate_direct(pivot_text, "en", tgt, max_length, num_beams)
            if final is None:
                return {
                    "translated_text": f"[Translation unavailable: en→{tgt} failed]",
                    "src_lang": src, "tgt_lang": tgt,
                    "strategy": "failed", "pivot_text": pivot_text,
                }
        else:
            final = pivot_text

        return {
            "translated_text": final,
            "src_lang": src, "tgt_lang": tgt,
            "strategy": "pivot_en", "pivot_text": pivot_text,
        }

    @staticmethod
    def supported_languages() -> Dict[str, str]:
        """Return the full name → code mapping."""
        return LANGUAGE_NAME_TO_CODE.copy()
