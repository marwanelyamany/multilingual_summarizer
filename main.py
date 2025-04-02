"""
main.py
-------
Command-line entry point for the Multilingual Summarization & Translation Pipeline.

Usage examples
--------------
# Summarize a file and translate to Arabic:
    python main.py --file data/sample_texts/english_ai.txt --target arabic

# Summarize inline text and translate to French:
    python main.py --text "Your long document here..." --target french

# Force source language (skip auto-detection):
    python main.py --file doc.txt --target spanish --src german

# Save output to JSON:
    python main.py --file doc.txt --target arabic --save results/output.json

# Evaluate against a reference translation:
    python main.py --file doc.txt --target french --reference "La référence ici..."
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import MultilingualPipeline
from src.translator import LANGUAGE_NAME_TO_CODE


SUPPORTED_LANGUAGES = sorted(LANGUAGE_NAME_TO_CODE.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multilingual Summarization & Translation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f", type=str,
        help="Path to input text file (.txt)"
    )
    input_group.add_argument(
        "--text", "-t", type=str,
        help="Input text as a string (wrap in quotes)"
    )

    # Language options
    parser.add_argument(
        "--target", "-tgt", type=str, default="english",
        choices=SUPPORTED_LANGUAGES,
        help="Target language for translation (default: english)"
    )
    parser.add_argument(
        "--src", "-s", type=str, default=None,
        help="Source language override (default: auto-detect)"
    )

    # Evaluation
    parser.add_argument(
        "--reference", "-r", type=str, default=None,
        help="Reference translation for BLEU/ROUGE-L evaluation"
    )

    # Output
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save results to this JSON file path"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress step-by-step progress output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load input text ────────────────────────────────────────────────────
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"[Error] File not found: {args.file}")
            sys.exit(1)
        text = p.read_text(encoding="utf-8")
        print(f"[Main] Loaded '{p.name}' ({len(text.split())} words)")
    else:
        text = args.text

    # ── Run pipeline ───────────────────────────────────────────────────────
    pipeline = MultilingualPipeline()
    result   = pipeline.run(
        text=text,
        target_language=args.target,
        src_lang_override=args.src,
        reference_summary=args.reference,
        verbose=not args.quiet,
    )

    # ── Save if requested ──────────────────────────────────────────────────
    if args.save:
        MultilingualPipeline.save_result(result, args.save)

    return result


if __name__ == "__main__":
    main()
