"""Part 2 — Learning from Doctor Edits.

Provides:
- SimulatedReviewer: a stand-in "doctor" that applies a consistent, hidden editing
  policy to discharge-summary drafts, producing (draft, corrected) pairs.
- CorrectionMemory: persists accumulated (draft, corrected) pairs per section and
  injects them as few-shot examples into future drafting prompts.

Safety guarantee: the correction-memory path cannot weaken the Part 1 no-fabrication
guardrail because:
  1. Refinement only runs on sections that already have evidence_ids.
  2. Missing/Conflicting sections are never touched.
  3. The grounding validator runs *after* refinement and will reject any verified
     field that loses its evidence citations or references unavailable evidence.
  4. The refinement Gemini prompt explicitly prohibits adding new clinical facts.
"""

from .correction_memory import CorrectionMemory
from .simulated_reviewer import SimulatedReviewer

__all__ = ["CorrectionMemory", "SimulatedReviewer"]
