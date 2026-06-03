from __future__ import annotations

import os
from pathlib import Path

from .ingestion import inspect_pdfs
from .models import (
    AgentRunResult,
    AgentStep,
    EvidenceFact,
    FactStatus,
    MedicationRecord,
    ReviewFlag,
)
from .provider import ExtractionProvider
from .tools import (
    SUMMARY_FIELD_NAMES,
    build_summary,
    detect_conflicts,
    find_missing_sections,
    mock_drug_interaction_check,
    new_id,
    reconcile_medications,
    source_path_map,
    validate_grounding,
)
from .part2.correction_memory import CorrectionMemory, DEFAULT_MEMORY_PATH


class DischargeSummaryAgent:
    def __init__(
        self,
        provider: ExtractionProvider,
        max_steps: int | None = None,
        memory_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.max_steps = max_steps or int(os.getenv("MAX_AGENT_STEPS", "20"))
        # memory_path=None disables Part 2 refinement (Part 1 behaviour unchanged)
        self._memory_path = memory_path

    def run(self, patient_label: str, paths: list[Path]) -> AgentRunResult:
        state = AgentRunResult(patient_label=patient_label)
        path_map = source_path_map(paths)
        extracted = False
        conflicts_checked = False
        missing_checked = False
        escalation_done = False
        medications_checked = False
        interactions_checked = False
        drafted = False
        refinement_done = False  # Part 2: memory-based style refinement
        validated = False

        # Load correction memory once; None means no memory available
        _correction_memory: CorrectionMemory | None = None
        if self._memory_path is not None:
            _correction_memory = CorrectionMemory(self._memory_path)
        elif DEFAULT_MEMORY_PATH.exists():
            _correction_memory = CorrectionMemory(DEFAULT_MEMORY_PATH)

        for step_number in range(1, self.max_steps + 1):
            if not state.pages:
                action, rationale = "inspect_documents", "No document inventory exists."
            elif not extracted:
                action, rationale = "extract_page_evidence", "Pages are known; extract evidence before clinical decisions."
            elif not conflicts_checked:
                action, rationale = "detect_conflicts", "Evidence exists; check identity and clinical-context compatibility."
            elif not missing_checked:
                action, rationale = "find_missing_sections", "Conflicts are known; identify required sections that remain unsupported."
            elif any(flag.severity == "critical" for flag in state.review_flags) and not escalation_done:
                action, rationale = "flag_for_clinician_review", "Critical safety concerns require explicit clinician escalation."
            elif not medications_checked:
                action, rationale = "reconcile_medications", "Medication evidence must be compared before drafting."
            elif not interactions_checked and any(m.phase == "discharge" for m in state.medications):
                action, rationale = "check_drug_interactions", "Discharge medications need a safety lookup."
            elif not drafted:
                action, rationale = "draft_summary", "Available evidence has passed pre-draft safety checks."
            elif not refinement_done:
                # Only refine if correction memory has at least one section with examples
                _has_memory = (
                    _correction_memory is not None
                    and bool(_correction_memory.sections_with_examples())
                )
                if _has_memory:
                    action, rationale = (
                        "refine_draft_with_memory",
                        "Correction memory has examples; applying style refinement before grounding validation.",
                    )
                else:
                    refinement_done = True
                    action, rationale = "validate_grounding", "The draft must be rejected if any verified fact lacks evidence."
            elif not validated:
                action, rationale = "validate_grounding", "The draft must be rejected if any verified fact lacks evidence."
            else:
                action, rationale = "finish", "The draft and safety checks are complete."

            step = AgentStep(step_number=step_number, rationale=rationale, action=action)
            try:
                if action == "inspect_documents":
                    state.pages = inspect_pdfs(paths)
                    step.inputs = {"files": [p.name for p in paths]}
                    step.result = f"Inspected {len(state.pages)} pages."
                    step.next_decision = "Extract page-level evidence."

                elif action == "extract_page_evidence":
                    unreadable = 0
                    quota_exhausted = False
                    for page_index, page in enumerate(state.pages):
                        try:
                            extraction = self.provider.extract_page(page, path_map[page.source_file])
                            page.document_type = extraction.document_type
                            page.readable = extraction.readable
                            page.extraction_method = "embedded_text" if page.text else "vision"
                            if not extraction.readable:
                                unreadable += 1
                                continue
                            for category in ("patient_id", "name", "age", "sex"):
                                value = getattr(extraction.identity, category)
                                if value:
                                    normalized_category = "patient_name" if category == "name" else category
                                    state.evidence.append(
                                        EvidenceFact(
                                            evidence_id=new_id("ev"),
                                            category=normalized_category,
                                            value=value,
                                            source_file=page.source_file,
                                            page_number=page.page_number,
                                            excerpt=value,
                                        )
                                    )
                            for item in extraction.facts:
                                state.evidence.append(
                                    EvidenceFact(
                                        evidence_id=new_id("ev"),
                                        category=item.category,
                                        value=item.value,
                                        source_file=page.source_file,
                                        page_number=page.page_number,
                                        excerpt=item.excerpt,
                                        status=item.status,
                                        confidence=item.confidence,
                                    )
                                )
                            for item in extraction.medications:
                                evidence_id = new_id("ev")
                                state.evidence.append(
                                    EvidenceFact(
                                        evidence_id=evidence_id,
                                        category="medication",
                                        value=item.name,
                                        source_file=page.source_file,
                                        page_number=page.page_number,
                                        excerpt=item.excerpt,
                                    )
                                )
                                state.medications.append(
                                    MedicationRecord(
                                        name=item.name,
                                        phase=item.phase,
                                        dose=item.dose,
                                        route=item.route,
                                        frequency=item.frequency,
                                        duration=item.duration,
                                        reason=item.reason,
                                        evidence_ids=[evidence_id],
                                    )
                                )
                        except Exception as exc:
                            page.readable = False
                            unreadable += 1
                            error_text = str(exc)
                            is_quota_error = "RESOURCE_EXHAUSTED" in error_text or "Quota exceeded" in error_text
                            state.review_flags.append(
                                ReviewFlag(
                                    flag_id=new_id("flag"),
                                    category="provider_quota_exhausted" if is_quota_error else "document_read_failure",
                                    message=f"{page.source_file} page {page.page_number} could not be extracted: {exc}",
                                    severity="critical" if is_quota_error else "warning",
                                )
                            )
                            if is_quota_error:
                                quota_exhausted = True
                                for remaining_page in state.pages[page_index + 1 :]:
                                    remaining_page.readable = False
                                unreadable += len(state.pages) - page_index - 1
                                break
                    extracted = True
                    failure_ratio = unreadable / len(state.pages) if state.pages else 1.0
                    if failure_ratio > 0.20:
                        state.review_flags.append(
                            ReviewFlag(
                                flag_id=new_id("flag"),
                                category="insufficient_document_coverage",
                                message=(
                                    f"Only {len(state.pages) - unreadable} of {len(state.pages)} pages were "
                                    "successfully extracted. The draft is incomplete and requires clinician review."
                                ),
                                severity="critical",
                            )
                        )
                    step.result = f"Extracted {len(state.evidence)} evidence facts; {unreadable} pages unreadable."
                    step.next_decision = (
                        "Escalate incomplete extraction after conflict checks."
                        if quota_exhausted
                        else "Check for conflicts and missing information."
                    )

                elif action == "detect_conflicts":
                    state.conflicts = detect_conflicts(state.evidence)
                    critical_evidence_ids = {
                        evidence_id
                        for conflict in state.conflicts
                        if conflict.severity == "critical"
                        for evidence_id in conflict.evidence_ids
                    }
                    for fact in state.evidence:
                        if fact.evidence_id in critical_evidence_ids:
                            fact.identity_compatible = False
                    incompatible_sources = {
                        (fact.source_file, fact.page_number)
                        for fact in state.evidence
                        if not fact.identity_compatible
                    }
                    for fact in state.evidence:
                        if (fact.source_file, fact.page_number) in incompatible_sources:
                            fact.identity_compatible = False
                    for page in state.pages:
                        if (page.source_file, page.page_number) in incompatible_sources:
                            page.quarantined = True
                            page.quarantine_reason = "Critical identity or clinical-context conflict."
                    for conflict in state.conflicts:
                        state.review_flags.append(
                            ReviewFlag(
                                flag_id=new_id("flag"),
                                category=conflict.category,
                                message=conflict.description,
                                severity=conflict.severity,
                                evidence_ids=conflict.evidence_ids,
                            )
                        )
                    conflicts_checked = True
                    step.result = f"Found {len(state.conflicts)} conflicts; quarantined {len(incompatible_sources)} pages."
                    step.next_decision = "Assess required sections against compatible evidence."

                elif action == "find_missing_sections":
                    missing = find_missing_sections(
                        [fact for fact in state.evidence if fact.identity_compatible]
                    )
                    for section in missing:
                        state.review_flags.append(
                            ReviewFlag(
                                flag_id=new_id("flag"),
                                category="missing_required_section",
                                message=f"{section.replace('_', ' ').title()} is not documented.",
                                severity="warning",
                            )
                        )
                    missing_checked = True
                    step.result = f"Found {len(missing)} missing required sections."
                    step.next_decision = "Escalate critical concerns or reconcile medications."

                elif action == "flag_for_clinician_review":
                    critical = [flag for flag in state.review_flags if flag.severity == "critical"]
                    state.review_flags.append(
                        ReviewFlag(
                            flag_id=new_id("flag"),
                            category="clinical_safety_escalation",
                            message=f"Agent escalated {len(critical)} critical safety concern(s) for clinician review.",
                            severity="critical",
                            evidence_ids=sorted(
                                {evidence_id for flag in critical for evidence_id in flag.evidence_ids}
                            ),
                        )
                    )
                    escalation_done = True
                    step.result = f"Escalated {len(critical)} critical safety concern(s)."
                    step.next_decision = "Continue only with non-quarantined evidence."

                elif action == "reconcile_medications":
                    state.medication_changes = reconcile_medications(state.medications)
                    for change in state.medication_changes:
                        if change.requires_review:
                            state.review_flags.append(
                                ReviewFlag(
                                    flag_id=new_id("flag"),
                                    category="medication_reconciliation",
                                    message=f"{change.name}: {change.change_type} with no documented reason.",
                                    severity="warning",
                                )
                            )
                    medications_checked = True
                    step.result = f"Produced {len(state.medication_changes)} medication reconciliation entries."
                    if any(m.phase == "discharge" for m in state.medications):
                        step.next_decision = "Check discharge medication interactions."
                    else:
                        interactions_checked = True
                        step.next_decision = "No discharge medications were sourced; proceed to draft with a missing marker."

                elif action == "check_drug_interactions":
                    flags = mock_drug_interaction_check(state.medications)
                    state.review_flags.extend(flags)
                    interactions_checked = True
                    step.result = f"Mock interaction lookup returned {len(flags)} flags."
                    step.next_decision = "Draft only from compatible evidence."

                elif action == "draft_summary":
                    state.summary = build_summary(state.evidence, state.medication_changes, state.conflicts)
                    drafted = True
                    step.result = "Created a clinician-review draft with evidence citations."
                    step.next_decision = "Validate grounding."

                elif action == "refine_draft_with_memory":
                    # Part 2: apply correction-memory few-shot refinement to the raw draft.
                    # Safety guarantees preserved:
                    #   - Only non-missing, non-conflicting sections are touched.
                    #   - evidence_ids are never modified.
                    #   - The grounding validator runs immediately after this step.
                    #   - Provider falls back to original on any Gemini error.
                    assert state.summary is not None
                    assert _correction_memory is not None
                    sections_refined = 0
                    for field_name in SUMMARY_FIELD_NAMES:
                        field = getattr(state.summary, field_name)
                        if field.status in (FactStatus.missing, FactStatus.conflicting):
                            continue
                        examples = _correction_memory.get_examples(field_name)
                        if not examples:
                            continue
                        try:
                            refined_value = self.provider.refine_section_with_memory(
                                field_name, field.value, examples
                            )
                        except Exception:
                            # Never fail the whole run over a refinement error
                            refined_value = field.value
                        if refined_value and refined_value != field.value:
                            # Rebuild field preserving status and evidence_ids; only value changes
                            updated_field = field.model_copy(update={"value": refined_value})
                            setattr(state.summary, field_name, updated_field)
                            sections_refined += 1
                    refinement_done = True
                    step.result = (
                        f"Memory refinement applied to {sections_refined} section(s). "
                        "Grounding validator will run next."
                    )
                    step.next_decision = "Validate grounding on refined draft."

                elif action == "validate_grounding":
                    assert state.summary is not None
                    flags = validate_grounding(state.summary, state.evidence)
                    state.review_flags.extend(flags)
                    validated = True
                    step.result = f"Grounding validation returned {len(flags)} flags."
                    step.next_decision = "Finish with unresolved flags visible."

                else:
                    blocking_categories = {
                        "provider_quota_exhausted",
                        "insufficient_document_coverage",
                        "agent_tool_failure",
                        "iteration_cap",
                    }
                    has_blocking_flag = any(
                        flag.category in blocking_categories and flag.severity == "critical"
                        for flag in state.review_flags
                    )
                    state.completed = not has_blocking_flag
                    state.stopped_reason = (
                        "completed" if state.completed else "incomplete due to critical safety concerns"
                    )
                    step.result = (
                        "Agent run completed."
                        if state.completed
                        else "Agent run ended with an incomplete draft and critical safety escalation."
                    )
                    step.next_decision = "Return draft for clinician review."

            except Exception as exc:
                step.error = str(exc)
                step.result = "Tool failed; failure was recorded."
                step.next_decision = "Return an incomplete draft and escalate."
                state.review_flags.append(
                    ReviewFlag(
                        flag_id=new_id("flag"),
                        category="agent_tool_failure",
                        message=f"{action} failed: {exc}",
                        severity="critical",
                    )
                )
                state.stopped_reason = f"{action} failed"
                state.trace.append(step)
                break

            state.trace.append(step)
            if action == "finish":
                break
        else:
            state.stopped_reason = "maximum iteration cap reached"
            state.review_flags.append(
                ReviewFlag(
                    flag_id=new_id("flag"),
                    category="iteration_cap",
                    message="Agent stopped at the maximum iteration cap.",
                    severity="critical",
                )
            )

        return state
