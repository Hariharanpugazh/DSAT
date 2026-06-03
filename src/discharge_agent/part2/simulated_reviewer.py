"""Simulated Reviewer — the stand-in "doctor" for Part 2.

This module applies a *consistent but hidden* editing policy to a discharge-summary
draft, producing the corrected version that a clinician would ideally produce.

WHY "HIDDEN": The agent never imports this module.  It only ever sees past
(draft, corrected) pairs through the CorrectionMemory injection.  That separation
makes the learning loop honest — the agent must generalise from examples, not from
the rules themselves.

POLICY INVARIANT — the reviewer never adds new clinical facts.  Every
transformation is purely structural or stylistic:
  • Bullet-point lists instead of semicolon-separated strings.
  • Sentence-case capitalisation for drug names already present.
  • Standardised condition labels ("Stable", "Improving", "Critical", "Fair")
    inferred *only* from words already in the draft.
  • "Follow up:" prefix when absent.
  • Stripping inline evidence-ID parentheses that belong in the JSON, not the
    human-readable summary.
  • Leaving "Missing - clinician review required" and "Conflicting - clinician
    review required" completely unchanged.

REWARD SIGNAL: section-level normalised edit similarity
  similarity(draft, corrected) = 1 - edit_distance(draft, corrected)
                                       / max(len(draft), len(corrected), 1)
Range [0, 1]; 1 = no edits needed (perfect first draft).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import NamedTuple

from discharge_agent.models import DischargeSummaryDraft, SummaryField


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVIDENCE_ID_RE = re.compile(r"\s*\(ev-[0-9a-f]+(,\s*ev-[0-9a-f]+)*\)", re.IGNORECASE)


def _strip_evidence_ids(text: str) -> str:
    """Remove inline evidence citations — they belong in the JSON, not the prose."""
    return _EVIDENCE_ID_RE.sub("", text).strip()


def _is_placeholder(text: str) -> bool:
    return "Missing" in text or "Conflicting" in text or "None documented" in text


def _semicolon_to_bullets(text: str) -> str:
    """Convert semicolon-separated items to a bullet list."""
    items = [i.strip() for i in text.split(";") if i.strip()]
    if len(items) <= 1:
        return text
    return "\n".join(f"• {i.rstrip('.')}" for i in items)


def _to_sentences(text: str) -> str:
    """Ensure each semicolon-separated clause ends with a period."""
    parts = [p.strip() for p in re.split(r";\s*", text) if p.strip()]
    result = ". ".join(p.rstrip(".") for p in parts)
    return result + "." if result and not result.endswith(".") else result


# ---------------------------------------------------------------------------
# Section-specific reviewers (the hidden policy)
# ---------------------------------------------------------------------------

def _review_patient_demographics(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    return _strip_evidence_ids(raw)


def _review_admission_date(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    return _strip_evidence_ids(raw)


def _review_discharge_date(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    return _strip_evidence_ids(raw)


def _review_principal_diagnosis(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw)
    # Capitalise first letter of each diagnosis when split by semicolons
    parts = [p.strip().capitalize() for p in cleaned.split(";") if p.strip()]
    return "; ".join(parts)


def _review_secondary_diagnoses(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw)
    parts = [p.strip().capitalize() for p in cleaned.split(";") if p.strip()]
    return "\n".join(f"• {p}" for p in parts) if len(parts) > 1 else cleaned


def _review_hospital_course(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw)
    return _to_sentences(cleaned)


def _review_procedures(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw)
    return _semicolon_to_bullets(cleaned)


def _review_discharge_medications(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw)
    meds = [m.strip() for m in cleaned.split(";") if m.strip()]
    formatted = []
    for med in meds:
        # Capitalise drug name (first token before a digit or space+digit)
        med = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), med)
        formatted.append(f"• {med}")
    return "\n".join(formatted)


def _review_allergies(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw)
    # Preserve "Not known" exactly — do not normalise to "No known allergies"
    if cleaned.lower() in {"not known", "nkda", "none known"}:
        return "Not known"
    return cleaned


def _review_follow_up_instructions(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw)
    parts = [p.strip() for p in cleaned.split(";") if p.strip()]
    normalised = []
    for part in parts:
        if not re.match(r"^(follow[- ]?up|review|return)", part, re.IGNORECASE):
            part = "Follow up: " + part
        normalised.append(part)
    return "\n".join(f"• {p}" for p in normalised) if len(normalised) > 1 else (normalised[0] if normalised else raw)


def _review_pending_results(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw)
    return _semicolon_to_bullets(cleaned)


def _review_discharge_condition(raw: str) -> str:
    if _is_placeholder(raw):
        return raw
    cleaned = _strip_evidence_ids(raw).lower()
    if "stable" in cleaned:
        return "Stable"
    if "improving" in cleaned or "improved" in cleaned:
        return "Improving"
    if "critical" in cleaned or "serious" in cleaned:
        return "Critical"
    if "fair" in cleaned:
        return "Fair"
    # Cannot normalise — return original stripped text
    return _strip_evidence_ids(raw)


_SECTION_REVIEWERS = {
    "patient_demographics": _review_patient_demographics,
    "admission_date": _review_admission_date,
    "discharge_date": _review_discharge_date,
    "principal_diagnosis": _review_principal_diagnosis,
    "secondary_diagnoses": _review_secondary_diagnoses,
    "hospital_course": _review_hospital_course,
    "procedures": _review_procedures,
    "discharge_medications": _review_discharge_medications,
    "allergies": _review_allergies,
    "follow_up_instructions": _review_follow_up_instructions,
    "pending_results": _review_pending_results,
    "discharge_condition": _review_discharge_condition,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SectionScore(NamedTuple):
    section: str
    draft: str
    corrected: str
    similarity: float  # [0, 1]; 1 = identical (no edits needed)


class ReviewResult(NamedTuple):
    corrected_sections: dict[str, str]   # section_name → corrected text
    scores: list[SectionScore]
    overall_similarity: float             # mean across non-placeholder sections


class SimulatedReviewer:
    """Applies a consistent hidden editing policy and returns corrected drafts.

    The agent never sees this class.  Only the correction_memory that stores the
    (draft, corrected) pairs is visible to the agent.
    """

    def review(self, summary: DischargeSummaryDraft) -> ReviewResult:
        corrected: dict[str, str] = {}
        scores: list[SectionScore] = []

        for section_name, reviewer_fn in _SECTION_REVIEWERS.items():
            field: SummaryField = getattr(summary, section_name)
            draft_text = field.value
            corrected_text = reviewer_fn(draft_text)
            corrected[section_name] = corrected_text

            # Only score non-placeholder sections (placeholders are always "correct")
            if not _is_placeholder(draft_text):
                similarity = _normalised_similarity(draft_text, corrected_text)
                scores.append(SectionScore(section_name, draft_text, corrected_text, similarity))

        if scores:
            overall = sum(s.similarity for s in scores) / len(scores)
        else:
            overall = 0.0

        return ReviewResult(corrected, scores, overall)


def _normalised_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio — identical strings → 1.0, completely different → 0.0."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def compute_similarity(draft: str, corrected: str) -> float:
    return _normalised_similarity(draft, corrected)
