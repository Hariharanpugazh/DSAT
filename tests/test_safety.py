from __future__ import annotations

from pathlib import Path

import fitz

from discharge_agent.agent import DischargeSummaryAgent
from discharge_agent.models import (
    DischargeSummaryDraft,
    EvidenceFact,
    ExtractedFact,
    FactStatus,
    MedicationRecord,
    PageExtraction,
    SummaryField,
)
from discharge_agent.tools import build_summary, detect_conflicts, reconcile_medications, validate_grounding


def fact(category: str, value: str, evidence_id: str) -> EvidenceFact:
    return EvidenceFact(
        evidence_id=evidence_id,
        category=category,
        value=value,
        source_file="notes.pdf",
        page_number=1,
        excerpt=value,
    )


def test_missing_fields_remain_missing() -> None:
    summary = build_summary([], [], [])
    assert summary.principal_diagnosis.status == FactStatus.missing
    assert "Missing" in summary.principal_diagnosis.value


def test_pending_result_remains_pending() -> None:
    pending = fact("pending_result", "Urine culture pending", "ev-1")
    pending.status = FactStatus.pending
    summary = build_summary([pending], [], [])
    assert summary.pending_results.status == FactStatus.pending


def test_not_known_allergy_is_preserved() -> None:
    summary = build_summary([fact("allergy", "Not known", "ev-1")], [], [])
    assert summary.allergies.value == "Not known"


def test_conflicting_sex_is_critical() -> None:
    conflicts = detect_conflicts([fact("sex", "Female", "ev-1"), fact("sex", "Male", "ev-2")])
    assert conflicts[0].severity == "critical"


def test_conflicting_principal_diagnoses_are_not_merged() -> None:
    evidence = [
        fact("principal_diagnosis", "Acute gastroenteritis", "ev-1"),
        fact("principal_diagnosis", "DKA", "ev-2"),
    ]
    conflicts = detect_conflicts(evidence)
    summary = build_summary(evidence, [], conflicts)
    assert any(c.category == "principal_diagnosis" for c in conflicts)
    assert summary.principal_diagnosis.status == FactStatus.conflicting


def test_cross_patient_context_is_flagged() -> None:
    conflicts = detect_conflicts(
        [fact("clinical_context", "Female patient", "ev-1"), fact("clinical_context", "Prostate noted", "ev-2")]
    )
    assert any(c.category == "possible_cross_patient_contamination" for c in conflicts)


def test_medication_reconciliation_flags_undocumented_change() -> None:
    changes = reconcile_medications(
        [
            MedicationRecord(name="Aspirin", phase="admission", dose="75 mg"),
            MedicationRecord(name="Aspirin", phase="discharge", dose="150 mg"),
            MedicationRecord(name="Atorvastatin", phase="discharge", dose="20 mg"),
        ]
    )
    assert {c.change_type for c in changes} == {"changed", "added"}
    assert all(c.requires_review for c in changes)


def test_unsupported_verified_summary_fact_is_rejected() -> None:
    missing = SummaryField(value="Missing", status=FactStatus.missing)
    summary = DischargeSummaryDraft(
        patient_demographics=SummaryField(value="Invented patient", status=FactStatus.verified),
        admission_date=missing,
        discharge_date=missing,
        principal_diagnosis=missing,
        secondary_diagnoses=missing,
        hospital_course=missing,
        procedures=missing,
        discharge_medications=missing,
        allergies=missing,
        follow_up_instructions=missing,
        pending_results=missing,
        discharge_condition=missing,
    )
    flags = validate_grounding(summary, [])
    assert any(f.category == "unsupported_summary_fact" for f in flags)


class EmptyProvider:
    def extract_page(self, page, source_path):
        return PageExtraction()


class FailingProvider:
    def extract_page(self, page, source_path):
        raise TimeoutError("simulated timeout")


class QuotaProvider:
    def __init__(self):
        self.calls = 0

    def extract_page(self, page, source_path):
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded")


def make_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Synthetic note")
    document.save(path)
    document.close()


def make_two_page_pdf(path: Path) -> None:
    document = fitz.open()
    for text in ("Female patient note", "Prostate imaging note"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class ContaminatedProvider:
    def extract_page(self, page, source_path):
        value = "Female patient" if page.page_number == 1 else "Prostate noted"
        return PageExtraction(facts=[ExtractedFact(category="clinical_context", value=value, excerpt=value)])


def test_agent_records_page_extraction_failure(tmp_path: Path) -> None:
    pdf = tmp_path / "patient.pdf"
    make_pdf(pdf)
    result = DischargeSummaryAgent(FailingProvider()).run("patient", [pdf])
    assert any(flag.category == "document_read_failure" for flag in result.review_flags)
    assert result.summary is not None


def test_agent_quarantines_cross_patient_contamination(tmp_path: Path) -> None:
    pdf = tmp_path / "patient-2.pdf"
    make_two_page_pdf(pdf)
    result = DischargeSummaryAgent(ContaminatedProvider()).run("patient-2", [pdf])
    assert all(page.quarantined for page in result.pages)
    assert all(not evidence.identity_compatible for evidence in result.evidence)
    assert any(flag.category == "clinical_safety_escalation" for flag in result.review_flags)


def test_agent_stops_page_calls_after_quota_exhaustion(tmp_path: Path) -> None:
    pdf = tmp_path / "patient.pdf"
    make_two_page_pdf(pdf)
    provider = QuotaProvider()
    result = DischargeSummaryAgent(provider).run("patient", [pdf])
    assert provider.calls == 1
    assert any(flag.category == "provider_quota_exhausted" for flag in result.review_flags)
    assert any(flag.category == "insufficient_document_coverage" for flag in result.review_flags)
    assert result.completed is False


def test_iteration_cap_stops_agent(tmp_path: Path) -> None:
    pdf = tmp_path / "patient.pdf"
    make_pdf(pdf)
    result = DischargeSummaryAgent(EmptyProvider(), max_steps=1).run("patient", [pdf])
    assert result.stopped_reason == "maximum iteration cap reached"
    assert any(flag.category == "iteration_cap" for flag in result.review_flags)
