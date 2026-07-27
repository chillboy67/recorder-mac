"""
Module 7 — CourseClassifier (P1-B)

Two-stage course domain detection:
  Stage 1: keyword matching (fast, millisecond-level)
  Stage 2: BGE-M3 embedding similarity (slower, used when Stage 1 uncertain)

Outputs a ClassificationResult used by TerminologyCorrector to select
the appropriate domain dictionary.

Usage:
    classifier = CourseClassifier(config)
    result = classifier.process(transcript_sample[:2000])
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from modules import ClassificationResult

logger = logging.getLogger(__name__)

# ============================================================
# Course Categories with Keywords
# ============================================================

COURSE_CATEGORIES: dict[str, list[str]] = {
    "deep_learning": [
        "neural network", "backpropagation", "CNN", "RNN", "LSTM", "GRU",
        "transformer", "attention", "self-attention", "multi-head attention",
        "gradient descent", "epoch", "batch size", "loss function", "softmax",
        "relu", "dropout", "batch normalization", "ResNet", "BERT", "GPT",
        "embedding", "tokenizer", "fine-tuning", "pre-training",
    ],
    "machine_learning": [
        "regression", "classification", "clustering", "SVM", "support vector",
        "random forest", "decision tree", "XGBoost", "feature engineering",
        "cross-validation", "overfitting", "underfitting", "bias variance",
        "k-nearest", "naive bayes", "ensemble", "boosting", "bagging",
    ],
    "statistics": [
        "p-value", "hypothesis", "confidence interval", "t-test", "ANOVA",
        "chi-square", "normal distribution", "standard deviation", "variance",
        "correlation", "regression", "Bayesian", "prior", "posterior",
        "maximum likelihood", "MLE", "MCMC", "central limit theorem",
        "null hypothesis", "Type I error", "Type II error", "statistical significance",
    ],
    "r_language": [
        "ggplot", "ggplot2", "dplyr", "tidyverse", "tibble", "data.frame",
        "mutate", "filter", "select", "group_by", "summarize", "pipe",
        "lm(", "glm(", "CRAN", "R language", "RStudio",
        "vector", "factor", "data.table", "readr", "tidyr", "purrr",
    ],
    "python_programming": [
        "def ", "import", "class", "pandas", "numpy", "matplotlib",
        "list comprehension", "dictionary", "lambda", "decorator",
        "pip install", "virtual environment", "Jupyter", "scikit-learn",
        "seaborn", "plotly", "FastAPI", "Django", "asyncio",
        "Python", "data analysis",
    ],
    "mathematics": [
        "theorem", "proof", "matrix", "eigenvalue", "eigenvector",
        "integral", "derivative", "linear algebra", "calculus", "topology",
        "manifold", "differential equation", "Fourier", "Laplace",
        "determinant", "singular value", "convex", "optimization",
    ],
    "data_science": [
        "EDA", "exploratory data analysis", "feature selection", "pipeline",
        "preprocessing", "data cleaning", "visualization", "correlation matrix",
        "missing values", "outlier", "normalization", "standardization",
    ],
}

# Descriptions used for embedding comparison (Stage 2)
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "deep_learning": (
        "This lecture covers neural networks, transformers, attention mechanism, "
        "backpropagation, and deep learning architectures like CNN and RNN."
    ),
    "machine_learning": (
        "This lecture covers machine learning algorithms including SVM, decision "
        "trees, random forests, feature engineering, and model evaluation."
    ),
    "statistics": (
        "This lecture covers statistical hypothesis testing, probability distributions, "
        "p-values, confidence intervals, Bayesian inference, and regression analysis."
    ),
    "r_language": (
        "This lecture covers R programming language, ggplot2 visualization, dplyr "
        "data manipulation, tidyverse packages, and statistical modeling in R."
    ),
    "python_programming": (
        "This lecture covers Python programming, pandas, numpy, data analysis, "
        "functions, classes, and Python libraries."
    ),
    "mathematics": (
        "This lecture covers mathematical theorems, proofs, linear algebra, calculus, "
        "matrix operations, and mathematical analysis."
    ),
    "data_science": (
        "This lecture covers data science workflow including exploratory data "
        "analysis, data cleaning, feature selection, and visualization."
    ),
    "general_lecture": (
        "This is a general academic lecture covering various topics."
    ),
}


class CourseClassifier:
    """Two-stage course domain classifier."""

    def __init__(self, config: dict):
        cfg = config.get("classifier", {})
        self.model_name = cfg.get("embedding_model", "BAAI/bge-m3")
        self.confidence_threshold = cfg.get("confidence_threshold", 0.70)
        self.cache_dir = cfg.get("cache_dir", "./models/bge")
        self.use_keyword_precheck = cfg.get("use_keyword_precheck", True)

        self._embedding_model = None
        self._category_embeddings = None

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(self, transcript_sample: str) -> ClassificationResult:
        """
        Classify a transcript sample (first ~2000 chars recommended).

        Returns ClassificationResult with course label and confidence.
        """
        t0 = time.time()

        if not transcript_sample or not transcript_sample.strip():
            dt_ms = (time.time() - t0) * 1000
            return ClassificationResult(
                course="general_lecture", confidence=0.0,
                top3=[("general_lecture", 0.0)], domain_hints=[],
                method="none", processing_time_ms=dt_ms,
            )

        # Stage 1: Keyword pre-check (fast)
        if self.use_keyword_precheck:
            kw_result = self._keyword_classify(transcript_sample)
            if kw_result is not None and kw_result.confidence >= self.confidence_threshold:
                kw_result.processing_time_ms = (time.time() - t0) * 1000
                logger.info(
                    "CourseClassifier: %s (conf=%.2f, method=keyword, %.0fms)",
                    kw_result.course, kw_result.confidence,
                    kw_result.processing_time_ms,
                )
                return kw_result

        # Stage 2: BGE-M3 embedding similarity
        emb_result = self._embedding_classify(transcript_sample)
        emb_result.processing_time_ms = (time.time() - t0) * 1000
        logger.info(
            "CourseClassifier: %s (conf=%.2f, method=embedding, %.0fms)",
            emb_result.course, emb_result.confidence,
            emb_result.processing_time_ms,
        )
        return emb_result

    # -----------------------------------------------------------
    # Stage 1: Keyword Classification
    # -----------------------------------------------------------

    def _keyword_classify(self, text: str) -> Optional[ClassificationResult]:
        """Fast keyword-based classification. Returns None if uncertain."""
        text_lower = text.lower()
        scores: dict[str, int] = {}
        hits: dict[str, list[str]] = {}

        for category, keywords in COURSE_CATEGORIES.items():
            if not keywords:
                continue
            matched = [kw for kw in keywords if kw.lower() in text_lower]
            scores[category] = len(matched)
            hits[category] = matched

        if not scores:
            return None

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_score = top[0]
        second_score = top[1][1] if len(top) > 1 else 0

        # Requirements: ≥ 5 keyword hits, ≥ 3 ahead of runner-up
        if best_score >= 5 and (best_score - second_score) >= 3:
            confidence = min(0.70 + best_score * 0.02, 0.95)
            top3 = [(c, s) for c, s in top[:3]]
            domain_hints = hits.get(best_cat, [])
            return ClassificationResult(
                course=best_cat, confidence=confidence,
                top3=top3, domain_hints=domain_hints,
                method="keyword",
            )
        return None

    # -----------------------------------------------------------
    # Stage 2: BGE-M3 Embedding Classification
    # -----------------------------------------------------------

    def _embedding_classify(self, text: str) -> ClassificationResult:
        """Use BGE-M3 embeddings to classify by similarity to category descriptions."""
        try:
            model = self._get_embedding_model()
        except Exception as exc:
            logger.warning(
                "Failed to load BGE-M3 embedding model: %s. "
                "Falling back to keyword method.", exc,
            )
            return ClassificationResult(
                course="general_lecture", confidence=0.0,
                top3=[("general_lecture", 0.0)], domain_hints=[],
                method="keyword_fallback",
            )

        # Encode the transcript
        text_embedding = model.encode(text)

        # Encode or retrieve cached category embeddings
        category_embeddings = self._get_category_embeddings(model)

        # Compute cosine similarities
        import numpy as np

        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        scores = {}
        for cat, emb in category_embeddings.items():
            scores[cat] = cosine(text_embedding, emb)

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_score = top[0]

        # If below confidence threshold, default to general_lecture
        if best_score < self.confidence_threshold:
            return ClassificationResult(
                course="general_lecture", confidence=best_score,
                top3=[(c, s) for c, s in top[:3]],
                domain_hints=[], method="embedding",
            )

        return ClassificationResult(
            course=best_cat, confidence=best_score,
            top3=[(c, s) for c, s in top[:3]],
            domain_hints=[], method="embedding",
        )

    # -----------------------------------------------------------
    # Model Loading (Singleton)
    # -----------------------------------------------------------

    def _get_embedding_model(self):
        """Lazy-load BGE-M3 embedding model."""
        if self._embedding_model is not None:
            return self._embedding_model

        logger.info("Loading BGE-M3 embedding model (%s)...", self.model_name)
        t0 = time.time()

        try:
            from FlagEmbedding import FlagModel
            model = FlagModel(
                self.model_name,
                query_instruction_for_retrieval="",
                use_fp16=True,  # Apple Silicon friendly
            )
        except ImportError:
            raise RuntimeError(
                "FlagEmbedding is required for course classification. "
                "Install: pip install FlagEmbedding"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BGE-M3 model: {exc}. "
                "First run will download ~2.3GB. Check network or download manually."
            ) from exc

        self._embedding_model = model
        dt = time.time() - t0
        logger.info("BGE-M3 loaded in %.1fs", dt)
        return model

    def _get_category_embeddings(self, model) -> dict:
        """Get or cache embeddings for all category descriptions."""
        if self._category_embeddings is not None:
            return self._category_embeddings

        logger.info("Encoding category descriptions...")
        embeddings = {}
        for cat, desc in CATEGORY_DESCRIPTIONS.items():
            embeddings[cat] = model.encode(desc)
        self._category_embeddings = embeddings
        return embeddings


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("CourseClassifier — Self Test")
    print("=" * 60)

    config = {
        "classifier": {
            "embedding_model": "BAAI/bge-m3",
            "confidence_threshold": 0.70,
            "cache_dir": "./models/bge",
            "use_keyword_precheck": True,
        }
    }

    classifier = CourseClassifier(config)

    # Test with sample texts
    dl_text = (
        "Today we will discuss the attention mechanism in neural networks. "
        "The transformer architecture uses self-attention and multi-head attention. "
        "We also cover backpropagation, gradient descent, and how CNN and RNN "
        "differ from transformer models. BERT and GPT are examples of pre-trained "
        "models that use fine-tuning for downstream tasks. The ReLU activation "
        "function is used with batch normalization and dropout for regularization. "
        "We also discuss embedding layers, tokenization, and the seq2seq architecture."
    )

    stats_text = (
        "In this lecture we cover statistical hypothesis testing. The p-value "
        "measures statistical significance. We discuss the null hypothesis, "
        "Type I and Type II errors, and confidence intervals. The ANOVA test "
        "compares means across groups. We also cover Bayesian inference with "
        "prior and posterior distributions, maximum likelihood estimation, "
        "and the central limit theorem."
    )

    r_text = (
        "Today we'll use R language for data analysis. The tidyverse package "
        "includes ggplot2 for visualization and dplyr for data manipulation. "
        "We use the pipe operator to chain operations. The data.frame is a "
        "fundamental data structure. We'll call lm() for linear regression "
        "and use mutate, filter, and group_by from dplyr."
    )

    general_text = (
        "Today I want to talk about environmental protection. Rachel Carson "
        "wrote Silent Spring which raised awareness about pesticides. "
        "The book had a major impact on environmental policy."
    )

    for name, text in [("Deep Learning", dl_text), ("Statistics", stats_text),
                        ("R Language", r_text), ("General", general_text)]:
        print(f"\n--- {name} ---")
        print(f"Text: {text[:120]}...")
        result = classifier.process(text)
        print(f"  Course:     {result.course}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Method:     {result.method}")
        print(f"  Top3:       {[(c, f'{s:.2f}') for c, s in result.top3[:3]]}")
        if result.domain_hints:
            print(f"  Hints:      {result.domain_hints[:5]}")

    print("\n=== Test Complete ===")
