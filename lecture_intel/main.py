#!/usr/bin/env python3
"""
Lecture Intelligence System — CLI Entry Point

Transcribe and correct lecture recordings using local AI models.
All processing is offline; no network required after initial model download.

LLM: Phi-4 (default) | Llama 3.1 8B | Gemma 3 12B
ASR: faster-whisper large-v3 (mlx-whisper pending macOS 26.4 compat)

Usage:
  python main.py -i lecture.mp3
  python main.py -i lecture.m4a --course auto --format md --format srt
  python main.py -i lecture.wav --course statistics --structure
  python main.py -i ./lectures/ --batch -o ./output --verbose
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click

# ============================================================
# Course type presets with domain keywords
# ============================================================

COURSE_PRESETS: dict[str, dict] = {
    "deep_learning": {
        "label": "Deep Learning / Neural Networks",
        "keywords": [
            "neural network", "deep learning", "attention", "transformer",
            "gradient descent", "backpropagation", "loss function",
            "activation", "ReLU", "softmax", "embedding", "tokenization",
            "fine-tuning", "pre-training", "BERT", "GPT", "LLM",
            "CNN", "RNN", "LSTM", "seq2seq", "encoder", "decoder",
        ],
    },
    "machine_learning": {
        "label": "Machine Learning",
        "keywords": [
            "regression", "classification", "clustering", "SVM",
            "random forest", "decision tree", "XGBoost",
            "feature engineering", "cross-validation", "overfitting",
            "ensemble", "boosting", "bagging",
        ],
    },
    "statistics": {
        "label": "Statistics",
        "keywords": [
            "p-value", "hypothesis testing", "confidence interval",
            "regression", "ANOVA", "chi-square", "t-test",
            "normal distribution", "standard deviation", "variance",
            "Bayesian", "MCMC", "likelihood", "prior", "posterior",
        ],
    },
    "r_language": {
        "label": "R Language",
        "keywords": [
            "ggplot2", "dplyr", "tidyverse", "data.frame", "tibble",
            "lm()", "glm()", "apply", "R Markdown", "Shiny", "CRAN",
            "vector", "matrix", "list", "factor",
        ],
    },
    "python": {
        "label": "Python Programming",
        "keywords": [
            "NumPy", "Pandas", "Matplotlib", "Scikit-learn", "PyTorch",
            "Jupyter", "list comprehension", "decorator",
            "class", "inheritance", "lambda", "virtual environment",
        ],
    },
    "mathematics": {
        "label": "Mathematics",
        "keywords": [
            "derivative", "integral", "matrix", "eigenvalue",
            "gradient", "Hessian", "convex", "optimization",
            "theorem", "proof", "linear algebra", "calculus",
        ],
    },
    "data_science": {
        "label": "Data Science",
        "keywords": [
            "EDA", "exploratory data analysis", "feature selection",
            "data cleaning", "visualization", "correlation matrix",
            "missing values", "outlier", "normalization",
        ],
    },
    "general": {
        "label": "General / Mixed",
        "keywords": [],
    },
}


# ============================================================
# CLI
# ============================================================

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input", "-i", required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input audio file or directory (with --batch).",
)
@click.option(
    "--output", "-o", default="./output",
    type=click.Path(path_type=Path),
    help="Output directory (default: ./output).",
)
@click.option(
    "--course", "-c", default="general",
    type=click.Choice(["auto"] + list(COURSE_PRESETS.keys())),
    help="Course type for domain context. Use 'auto' for automatic detection.",
)
@click.option(
    "--enhance/--no-enhance", default=True,
    help="Enable/disable LLM text correction (default: enabled).",
)
@click.option(
    "--enhance-audio/--no-enhance-audio", default=True,
    help="Enable/disable audio enhancement with DeepFilterNet (P1-A).",
)
@click.option(
    "--structure/--no-structure", default=False,
    help="Enable lecture structuring with key point highlighting (P2-B).",
)
@click.option(
    "--diarize/--no-diarize", default=False,
    help="Enable speaker diarization — requires HF token (P2-A).",
)
@click.option(
    "--asr", default="auto",
    type=click.Choice(["auto", "faster-whisper"]),
    help="ASR engine (default: auto-detect).",
)
@click.option(
    "--model", "-m", default="large-v3",
    type=click.Choice(["large-v3", "large-v2", "medium", "small"]),
    help="Whisper model size (default: large-v3).",
)
@click.option(
    "--format", "-f", "formats", multiple=True,
    type=click.Choice(["txt", "md", "docx", "srt", "json"]),
    help="Output formats (can specify multiple).",
)
@click.option(
    "--batch/--no-batch", default=False,
    help="Batch mode: process all audio files in --input directory.",
)
@click.option(
    "--verbose", "-v", is_flag=True,
    help="Enable verbose/debug logging.",
)
@click.option(
    "--config", "config_path", default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom config.yaml (default: ./config.yaml).",
)
@click.option(
    "--keywords", default=None, type=str,
    help="Comma-separated domain keywords for LLM correction.",
)
def main(
    input: Path, output: Path, course: str, enhance: bool,
    enhance_audio: bool, structure: bool, diarize: bool,
    asr: str, model: str, formats: tuple[str, ...],
    batch: bool, verbose: bool, config_path: Optional[Path],
    keywords: Optional[str],
):
    """
    Lecture Intelligence System — Local AI Lecture Transcription

    \b
    Examples:
      python main.py -i lecture.mp3
      python main.py -i lecture.m4a --course auto --format md --format srt
      python main.py -i lecture.wav --course statistics --structure
      python main.py -i ./lectures/ --batch -o ./output --verbose
      python main.py -i lecture.mp3 --course deep_learning --enhance-audio
    """
    setup_logging(verbose)
    logger = logging.getLogger("main")
    logger.info("Lecture Intelligence System — Starting")

    # --- Resolve config ---
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        click.echo(
            f"Error: Config file not found at {config_path}", err=True,
        )
        sys.exit(1)

    # --- Validate input ---
    if batch and not input.is_dir():
        click.echo(
            f"Error: --batch mode requires --input to be a directory.", err=True,
        )
        sys.exit(1)
    if not batch and not input.is_file():
        click.echo(
            f"Error: --input must be a file (use --batch for directories).", err=True,
        )
        sys.exit(1)

    # --- Resolve course type ---
    use_auto_classify = course == "auto"
    if use_auto_classify:
        course_for_llm = "general"  # will be overridden by classifier
    else:
        course_for_llm = course

    course_preset = COURSE_PRESETS.get(course_for_llm, COURSE_PRESETS["general"])
    domain_kw = course_preset["keywords"].copy()
    if keywords:
        domain_kw.extend(k.strip() for k in keywords.split(",") if k.strip())

    # Resolve formats
    format_list = list(formats) if formats else []

    # --- Print configuration ---
    click.echo(f"\n{'='*60}")
    click.echo(f"  Lecture Intelligence System")
    click.echo(f"{'='*60}")
    click.echo(f"  Input:         {input}")
    click.echo(f"  Output:        {output}")
    click.echo(f"  Course:        {course}{' (auto-detect)' if use_auto_classify else ''}")
    click.echo(f"  ASR Engine:    {asr}")
    click.echo(f"  Model:         {model}")
    click.echo(f"  LLM Correct:   {'on' if enhance else 'off'}")
    click.echo(f"  Audio Enhance: {'on' if enhance_audio else 'off'} (P1-A)")
    click.echo(f"  Diarization:   {'on' if diarize else 'off'} (P2-A)")
    click.echo(f"  Structuring:   {'on' if structure else 'off'} (P2-B)")
    click.echo(f"  Formats:       {', '.join(format_list) if format_list else 'default'}")
    click.echo(f"  Batch:         {'yes' if batch else 'no'}")
    click.echo(f"{'='*60}\n")

    # --- Initialize pipeline ---
    from pipeline import Pipeline

    try:
        pipeline = Pipeline(config_path)
    except Exception as exc:
        click.echo(f"Error: Failed to initialize pipeline: {exc}", err=True)
        logger.exception("Pipeline initialization failed")
        sys.exit(1)

    # Apply CLI overrides to pipeline config
    if asr != "auto":
        pipeline.asr_engine.preferred_engine = asr
    if model:
        pipeline.asr_engine.model_name = model

    # P1/P2 switch overrides
    pipeline.pipeline_cfg["enable_audio_enhancement"] = enhance_audio
    pipeline.pipeline_cfg["enable_speaker_diarization"] = diarize
    pipeline.pipeline_cfg["enable_lecture_structuring"] = structure

    # --- Run ---
    try:
        if batch:
            _run_batch(pipeline, input, output, course_for_llm, domain_kw, enhance, format_list)
        else:
            _run_single(pipeline, input, output, course_for_llm, domain_kw, enhance, format_list)
    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        if verbose:
            logger.exception("Pipeline failed")
        sys.exit(1)


# ============================================================
# Helpers
# ============================================================

def _run_single(
    pipeline, input_path: Path, output_dir: Path,
    course: str, domain_kw: list[str], enhance: bool,
    formats: list[str],
) -> None:
    """Run pipeline on a single file."""
    result = pipeline.process(
        input_path=input_path, output_dir=output_dir,
        course_type=course, domain_keywords=domain_kw,
        enhance=enhance, formats=formats if formats else None,
    )
    if result.warnings:
        click.echo(f"\n⚠ {len(result.warnings)} warning(s):")
        for w in result.warnings[:5]:
            click.echo(f"  - {w}")
        if len(result.warnings) > 5:
            click.echo(f"  ... and {len(result.warnings) - 5} more")


def _run_batch(
    pipeline, input_dir: Path, output_dir: Path,
    course: str, domain_kw: list[str], enhance: bool,
    formats: list[str],
) -> None:
    """Run pipeline on all files in a directory."""
    results = pipeline.process_batch(
        input_dir=input_dir, output_dir=output_dir,
        course_type=course, domain_keywords=domain_kw,
        enhance=enhance, formats=formats if formats else None,
    )
    if not results:
        click.echo("No files were successfully processed.")
        sys.exit(1)

    total_time = sum(r.elapsed_sec for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    click.echo(f"\n{'='*60}")
    click.echo(f"  Batch Complete")
    click.echo(f"{'='*60}")
    click.echo(f"  Processed:    {len(results)} file(s)")
    click.echo(f"  Total time:   {total_time:.1f}s")
    if total_warnings:
        click.echo(f"  Warnings:     {total_warnings}")


def setup_logging(verbose: bool) -> None:
    """Configure logging with appropriate verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
