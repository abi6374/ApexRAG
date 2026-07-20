"""
agents/planner/ml_router.py — ML-Based DAGRouter.

Replaces the heuristic regex classifier with a fine-tuned LogisticRegression
on sentence embeddings from a frozen MiniLM model.

Architecture:
    Query text
      → sentence-transformers/all-MiniLM-L6-v2 (frozen, 384-dim)  [~5ms]
      → sklearn LogisticRegression (multi-label, 3 OvR classifiers) [<1ms]
      → frozenset({"entity", "citation", "policy"})

Falls back to the heuristic :class:`DAGRouter` if:
    - The `apex-rag[ml]` extras are not installed
    - The trained model file is not found on disk
    - The embedding model fails to load

Usage:
    router = MLRouter()
    needed = router.classify("Who is the CEO and what regulations apply?")
    # → {"entity", "policy"}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from apex_rag.agents.planner.dag_router import DAGRouter

logger = logging.getLogger("apex_rag.agents.planner.ml_router")

# Default path for the trained model (relative to project root)
_MODEL_DIR = Path(os.getenv("APEX_ML_MODEL_DIR", "models"))
_DEFAULT_MODEL_PATH = _MODEL_DIR / "dag_router_model.joblib"
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Multi-label label order — must match training script
_LABEL_NAMES = ["entity", "citation", "policy"]


class MLRouter(DAGRouter):
    """ML-powered DAGRouter with heuristic fallback.

    Loads a trained LogisticRegression model + frozen sentence transformer
    on first use.  If loading fails, falls back to the parent heuristic
    :meth:`classify` (which is always available).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._classifier: MultiOutputClassifier | None = None
        self._embedder: Any = None  # sentence-transformers model
        self._fallback_active = False  # True if we failed to load ML model
        self._loaded = False

    # ── Lazy loading ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Load the classifier and embedding model if not already done.

        Returns:
            True if ML backend is ready, False if fallback is active.
        """
        if self._loaded:
            return not self._fallback_active

        self._loaded = True

        # 1. Try to load the trained classifier
        if not self._model_path.exists():
            logger.info(
                "ML model not found at %s — using heuristic fallback. "
                "Run 'python scripts/train_dag_router.py' to train one.",
                self._model_path,
            )
            self._fallback_active = True
            return False

        try:
            self._classifier = joblib.load(str(self._model_path))
            logger.debug("Loaded ML classifier from %s", self._model_path)
        except Exception as exc:
            logger.warning(
                "Failed to load ML classifier (%s) — using heuristic fallback.",
                exc,
            )
            self._fallback_active = True
            return False

        # 2. Try to load the sentence transformer embedder
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(_EMBEDDING_MODEL_NAME)
            logger.debug("Loaded embedding model %s", _EMBEDDING_MODEL_NAME)
        except Exception as exc:
            logger.warning(
                "Failed to load embedding model (%s) — using heuristic fallback.",
                exc,
            )
            self._fallback_active = True
            return False

        self._fallback_active = False
        return True

    # ── Public interface ───────────────────────────────────────────────

    def classify(
        self,
        query: str,
        *,
        planner_data: dict[str, Any] | None = None,
    ) -> frozenset[str]:
        """Classify the query using ML, falling back to heuristic.

        Args:
            query:        The user's natural-language query.
            planner_data: Optional planner signals (used in both ML and
                          heuristic paths for enrichment).

        Returns:
            A frozenset of needed DAG projection tags.
        """
        # Try ML path
        if self._ensure_loaded():
            try:
                return self._ml_classify(query, planner_data=planner_data)
            except Exception as exc:
                logger.warning(
                    "ML classification failed (%s) — falling back to heuristic.",
                    exc,
                )

        # Fallback to heuristic (always available)
        return super().classify(query, planner_data=planner_data)

    def _ml_classify(
        self,
        query: str,
        *,
        planner_data: dict[str, Any] | None = None,
    ) -> frozenset[str]:
        """Run ML-based classification on the query."""
        # Embed the query
        emb: np.ndarray = self._embedder.encode(query, normalize_embeddings=True)
        emb_2d = emb.reshape(1, -1)  # shape (1, 384)

        # Predict multi-label via MultiOutputClassifier
        predictions = self._classifier.predict(emb_2d)  # shape (1, 3)
        needed: set[str] = set()

        for idx, label in enumerate(_LABEL_NAMES):
            if predictions[0, idx] == 1:
                needed.add(label)

        # ── Enrich with planner data signals (shared logic) ────────
        self._apply_planner_data(needed, planner_data)

        return frozenset(needed)

    def check_ml_ready(self) -> bool:
        """Check whether the ML backend is loaded and ready.

        Note: this is not a cheap property — it triggers model loading
        on first call.  Use it as a one-time check after construction.

        Returns:
            True if the ML backend is ready for inference.
        """
        return self._ensure_loaded() and not self._fallback_active

    def check_is_using_fallback(self) -> bool:
        """Check whether the heuristic fallback is active.

        Note: this is not a cheap property — it triggers model loading
        on first call.

        Returns:
            True if the fallback is active (ML not available).
        """
        self._ensure_loaded()
        return self._fallback_active
