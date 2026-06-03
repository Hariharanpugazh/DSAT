"""Safe deterministic refinement for Part 2 correction memory.

This module deliberately does not import the simulated reviewer. It learns only
small formatting preferences from stored (draft, corrected) examples and applies
them to later drafts without touching evidence IDs or field status.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from discharge_agent.models import DischargeSummaryDraft, FactStatus, SummaryField
from discharge_agent.part2.correction_memory import CorrectionExample, CorrectionMemory
from discharge_agent.tools import SUMMARY_FIELD_NAMES


_EVIDENCE_ID_RE = re.compile(r"\s*\(ev-[0-9a-f]+(?:,\s*ev-[0-9a-f]+)*\)", re.IGNORECASE)


@dataclass(frozen=True)
class RefinementReport:
    draft: DischargeSummaryDraft
    sections_changed: int
    changed_sections: list[str]


def refine_text_with_examples(
    section_name: str,
    draft_value: str,
    examples: list[CorrectionExample],
    *,
    allow_exact_replay: bool = False,
) -> str:
    """Apply conservative learned formatting to one section.

    The refiner is intentionally narrow: it only removes inline citation tags,
    converts semicolon-delimited lists to bullets when examples show that style,
    sentence-formats prose when examples show that style, and learns simple
    condition labels from previous before/after pairs.
    """
    if not examples or _is_placeholder(draft_value):
        return draft_value

    if allow_exact_replay:
        for ex in reversed(examples):
            if ex.draft.strip() == draft_value.strip():
                return ex.corrected

    refined = draft_value
    if _examples_strip_citations(examples):
        refined = _strip_evidence_ids(refined)

    if _examples_use_bullets(examples) and ";" in refined:
        refined = _semicolon_to_bullets(refined)
    elif _examples_use_sentences(examples) and ";" in refined:
        refined = _semicolon_to_sentences(refined)

    condition = _learn_condition_label(refined, examples)
    if condition:
        refined = condition

    return refined.strip() or draft_value


def refine_summary_with_memory(
    draft: DischargeSummaryDraft,
    memory: CorrectionMemory,
    *,
    allow_exact_replay: bool = False,
) -> RefinementReport:
    """Return a refined draft while preserving grounding metadata."""
    updates: dict[str, SummaryField] = {}
    changed: list[str] = []

    for field_name in SUMMARY_FIELD_NAMES:
        field: SummaryField = getattr(draft, field_name)
        if field.status in (FactStatus.missing, FactStatus.conflicting):
            updates[field_name] = field
            continue

        examples = memory.get_examples(field_name)
        refined_value = refine_text_with_examples(
            field_name,
            field.value,
            examples,
            allow_exact_replay=allow_exact_replay,
        )
        if refined_value != field.value:
            updates[field_name] = field.model_copy(update={"value": refined_value})
            changed.append(field_name)
        else:
            updates[field_name] = field

    return RefinementReport(
        draft=draft.model_copy(update=updates),
        sections_changed=len(changed),
        changed_sections=changed,
    )


def _is_placeholder(text: str) -> bool:
    lowered = text.lower()
    return (
        "missing" in lowered
        or "conflicting" in lowered
        or "clinician review required" in lowered
    )


def _strip_evidence_ids(text: str) -> str:
    return _EVIDENCE_ID_RE.sub("", text).strip()


def _examples_strip_citations(examples: list[CorrectionExample]) -> bool:
    return any(_EVIDENCE_ID_RE.search(ex.draft) and not _EVIDENCE_ID_RE.search(ex.corrected) for ex in examples)


def _examples_use_bullets(examples: list[CorrectionExample]) -> bool:
    return any(_has_semicolon_list(ex.draft) and _has_bullets(ex.corrected) for ex in examples)


def _examples_use_sentences(examples: list[CorrectionExample]) -> bool:
    return any(_has_semicolon_list(ex.draft) and ". " in ex.corrected and not _has_bullets(ex.corrected) for ex in examples)


def _has_semicolon_list(text: str) -> bool:
    return len([part for part in text.split(";") if part.strip()]) > 1


def _has_bullets(text: str) -> bool:
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(stripped_lines) and all(line.startswith(("-", "*", "•")) for line in stripped_lines)


def _semicolon_to_bullets(text: str) -> str:
    items = [item.strip().rstrip(".") for item in text.split(";") if item.strip()]
    if len(items) <= 1:
        return text
    return "\n".join(f"- {item}" for item in items)


def _semicolon_to_sentences(text: str) -> str:
    parts = [part.strip().rstrip(".") for part in text.split(";") if part.strip()]
    if len(parts) <= 1:
        return text
    return ". ".join(parts) + "."


def _learn_condition_label(draft_value: str, examples: list[CorrectionExample]) -> str | None:
    labels = {"stable", "improving", "critical", "fair"}
    learned: dict[str, str] = {}
    for ex in examples:
        corrected = ex.corrected.strip()
        if corrected.lower() in labels:
            for label in labels:
                if label in ex.draft.lower():
                    learned[label] = corrected
    for label, corrected in learned.items():
        if label in draft_value.lower():
            return corrected
    return None
