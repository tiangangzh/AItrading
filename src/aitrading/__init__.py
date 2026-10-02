"""AI-assisted equity research pipeline.

Observation (natural language)
  -> typed ScreenSpec (LLM, validated against the feature catalog)
  -> deterministic feature engine over institutional data (technical + fundamental + positioning)
  -> screen + cross-sectional ranking
  -> narrative engine (transcripts, news, filings, research)
  -> explanation agent (LLM) with programmatic grounding checks
  -> auditable report.

The LLM never computes numbers that drive selection: it translates intent into a typed spec and
explains candidates using evidence the deterministic layers supply. Every claim it makes is checked
against the feature frame or the source documents before the report is written.
"""

__version__ = "0.1.0"
