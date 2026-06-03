from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FactStatus(str, Enum):
    verified = "verified"
    pending = "pending"
    missing = "missing"
    conflicting = "conflicting"
    unreadable = "unreadable"


class DocumentPage(BaseModel):
    document_id: str
    source_file: str
    page_number: int
    text: str = ""
    image_bytes: bytes | None = Field(default=None, exclude=True)
    extraction_method: Literal["embedded_text", "vision", "none"] = "none"
    readable: bool = True
    quarantined: bool = False
    quarantine_reason: str | None = None
    document_type: str = "unknown"


class PatientIdentity(BaseModel):
    patient_id: str | None = None
    name: str | None = None
    age: str | None = None
    sex: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceFact(BaseModel):
    evidence_id: str
    category: str
    value: str
    source_file: str
    page_number: int
    excerpt: str
    status: FactStatus = FactStatus.verified
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    readable: bool = True
    identity_compatible: bool = True
    conflict_ids: list[str] = Field(default_factory=list)


class MedicationRecord(BaseModel):
    name: str
    phase: Literal["admission", "discharge", "inpatient", "unknown"] = "unknown"
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    reason: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class MedicationChange(BaseModel):
    name: str
    change_type: Literal["added", "stopped", "changed", "unchanged", "uncertain"]
    admission: MedicationRecord | None = None
    discharge: MedicationRecord | None = None
    documented_reason: str | None = None
    requires_review: bool = False


class Conflict(BaseModel):
    conflict_id: str
    category: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    severity: Literal["warning", "critical"] = "warning"


class ReviewFlag(BaseModel):
    flag_id: str
    category: str
    message: str
    severity: Literal["info", "warning", "critical"] = "warning"
    evidence_ids: list[str] = Field(default_factory=list)
    resolved: bool = False


class SummaryField(BaseModel):
    value: str
    status: FactStatus = FactStatus.verified
    evidence_ids: list[str] = Field(default_factory=list)


class DischargeSummaryDraft(BaseModel):
    disclaimer: str = "DRAFT FOR CLINICIAN REVIEW - NOT A FINAL CLINICAL DOCUMENT"
    patient_demographics: SummaryField
    admission_date: SummaryField
    discharge_date: SummaryField
    principal_diagnosis: SummaryField
    secondary_diagnoses: SummaryField
    hospital_course: SummaryField
    procedures: SummaryField
    discharge_medications: SummaryField
    allergies: SummaryField
    follow_up_instructions: SummaryField
    pending_results: SummaryField
    discharge_condition: SummaryField


class AgentStep(BaseModel):
    step_number: int
    rationale: str
    action: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    next_decision: str = ""
    error: str | None = None


class AgentRunResult(BaseModel):
    patient_label: str
    summary: DischargeSummaryDraft | None = None
    evidence: list[EvidenceFact] = Field(default_factory=list)
    medications: list[MedicationRecord] = Field(default_factory=list)
    medication_changes: list[MedicationChange] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)
    pages: list[DocumentPage] = Field(default_factory=list)
    completed: bool = False
    stopped_reason: str | None = None


class ExtractedFact(BaseModel):
    category: str
    value: str
    excerpt: str
    status: FactStatus = FactStatus.verified
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractedMedication(BaseModel):
    name: str
    phase: Literal["admission", "discharge", "inpatient", "unknown"] = "unknown"
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    reason: str | None = None
    excerpt: str


class PageExtraction(BaseModel):
    document_type: str = "unknown"
    readable: bool = True
    identity: PatientIdentity = Field(default_factory=PatientIdentity)
    facts: list[ExtractedFact] = Field(default_factory=list)
    medications: list[ExtractedMedication] = Field(default_factory=list)

