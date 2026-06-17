"""
apex_rag.temporal — Temporal Intelligence Layer (Part 3).

Components:
    - TemporalExtractor:     3-strategy document date extraction (metadata, regex, LLM)
    - FreshnessScorer:       Exponential-decay freshness computation
    - TemporalContradictionDetector:  3-step conflict detection between nodes
"""
