from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from .models import (
    Conflict,
    DischargeSummaryDraft,
    EvidenceFact,
    FactStatus,
    MedicationChange,
    MedicationRecord,
    ReviewFlag,
    SummaryField,
)


REQUIRED_CATEGORIES = {
    "patient_demographics": ("patient_name", "age", "sex", "patient_id"),
    "admission_date": ("admission_date",),
    "discharge_date": ("discharge_date",),
    "principal_diagnosis": ("principal_diagnosis",),
    "secondary_diagnoses": ("secondary_diagnosis",),
    "hospital_course": ("hospital_course",),
    "procedures": ("procedure",),
    "allergies": ("allergy",),
    "follow_up_instructions": ("follow_up",),
    "pending_results": ("pending_result",),
    "discharge_condition": ("discharge_condition",),
}

SUMMARY_FIELD_NAMES = (
    "patient_demographics",
    "admission_date",
    "discharge_date",
    "principal_diagnosis",
    "secondary_diagnoses",
    "hospital_course",
    "procedures",
    "discharge_medications",
    "allergies",
    "follow_up_instructions",
    "pending_results",
    "discharge_condition",
)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def find_missing_sections(evidence: list[EvidenceFact]) -> list[str]:
    categories = {fact.category for fact in evidence if fact.identity_compatible and fact.readable}
    return [
        section
        for section, candidates in REQUIRED_CATEGORIES.items()
        if not any(category in categories for category in candidates)
    ]


def detect_conflicts(evidence: list[EvidenceFact]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    grouped: dict[str, list[EvidenceFact]] = defaultdict(list)
    for fact in evidence:
        if fact.readable:
            grouped[fact.category].append(fact)

    for category in ("patient_id", "sex", "age", "admission_date", "discharge_date", "principal_diagnosis"):
        facts = grouped.get(category, [])
        values = {fact.value.strip().lower() for fact in facts}
        if len(values) > 1:
            conflict = Conflict(
                conflict_id=new_id("conflict"),
                category=category,
                description=f"Conflicting {category.replace('_', ' ')} values: {', '.join(sorted(values))}",
                evidence_ids=[fact.evidence_id for fact in facts],
                severity="critical" if category in {"patient_id", "sex"} else "warning",
            )
            conflicts.append(conflict)
            for fact in facts:
                fact.status = FactStatus.conflicting
                fact.conflict_ids.append(conflict.conflict_id)

    contexts = grouped.get("clinical_context", [])
    identity_context = contexts + grouped.get("sex", []) + grouped.get("principal_diagnosis", [])
    context_text = " ".join(f.value.lower() for f in identity_context)
    incompatible_pairs = [
        ("prostate", "pregnan"),
        ("prostate", "female"),
        ("obstetric", "male"),
    ]
    for left, right in incompatible_pairs:
        if left in context_text and right in context_text:
            affected = [f for f in identity_context if left in f.value.lower() or right in f.value.lower()]
            conflicts.append(
                Conflict(
                    conflict_id=new_id("conflict"),
                    category="possible_cross_patient_contamination",
                    description=f"Incompatible clinical contexts detected: {left} and {right}.",
                    evidence_ids=[f.evidence_id for f in affected],
                    severity="critical",
                )
            )
            for fact in affected:
                fact.status = FactStatus.conflicting
    return conflicts


def reconcile_medications(medications: list[MedicationRecord]) -> list[MedicationChange]:
    admission = {m.name.lower(): m for m in medications if m.phase == "admission"}
    discharge = {m.name.lower(): m for m in medications if m.phase == "discharge"}
    changes: list[MedicationChange] = []
    for key in sorted(set(admission) | set(discharge)):
        before, after = admission.get(key), discharge.get(key)
        if before is None:
            kind = "added"
        elif after is None:
            kind = "stopped"
        elif (before.dose, before.route, before.frequency) != (after.dose, after.route, after.frequency):
            kind = "changed"
        else:
            kind = "unchanged"
        reason = (after.reason if after else None) or (before.reason if before else None)
        changes.append(
            MedicationChange(
                name=(after or before).name,
                change_type=kind,
                admission=before,
                discharge=after,
                documented_reason=reason,
                requires_review=kind in {"added", "stopped", "changed"} and not reason,
            )
        )
    return changes


def mock_drug_interaction_check(medications: list[MedicationRecord]) -> list[ReviewFlag]:
    # Deliberately transparent mock: replace with a validated medication knowledge base in production.
    names = {m.name.lower() for m in medications if m.phase == "discharge"}
    flags: list[ReviewFlag] = []
    known_pairs = {frozenset({"warfarin", "aspirin"}), frozenset({"clarithromycin", "simvastatin"})}
    for pair in known_pairs:
        if pair.issubset(names):
            flags.append(
                ReviewFlag(
                    flag_id=new_id("flag"),
                    category="drug_interaction",
                    message=f"Mock interaction lookup flagged: {' + '.join(sorted(pair))}.",
                    severity="critical",
                )
            )
    return flags


def build_summary(
    evidence: list[EvidenceFact],
    medication_changes: list[MedicationChange],
    conflicts: list[Conflict],
) -> DischargeSummaryDraft:
    conflict_categories = {c.category for c in conflicts}
    usable = [f for f in evidence if f.identity_compatible and f.readable and f.status != FactStatus.conflicting]

    def field(categories: tuple[str, ...], missing: str = "Missing - clinician review required") -> SummaryField:
        facts = [f for f in usable if f.category in categories]
        if not facts:
            if any(c in conflict_categories for c in categories):
                return SummaryField(value="Conflicting - clinician review required", status=FactStatus.conflicting)
            return SummaryField(value=missing, status=FactStatus.missing)
        status = FactStatus.pending if any(f.status == FactStatus.pending for f in facts) else FactStatus.verified
        return SummaryField(
            value="; ".join(dict.fromkeys(f.value for f in facts)),
            status=status,
            evidence_ids=[f.evidence_id for f in facts],
        )

    med_lines = []
    med_evidence: list[str] = []
    for change in medication_changes:
        med = change.discharge
        if not med:
            continue
        detail = " ".join(filter(None, [med.name, med.dose, med.route, med.frequency, med.duration]))
        suffix = f" [{change.change_type}]"
        if change.requires_review:
            suffix += " - reason not documented; reconcile"
        med_lines.append(detail + suffix)
        med_evidence.extend(med.evidence_ids)

    return DischargeSummaryDraft(
        patient_demographics=field(("patient_name", "age", "sex", "patient_id")),
        admission_date=field(("admission_date",)),
        discharge_date=field(("discharge_date",)),
        principal_diagnosis=field(("principal_diagnosis",)),
        secondary_diagnoses=field(("secondary_diagnosis",)),
        hospital_course=field(("hospital_course",)),
        procedures=field(("procedure",)),
        discharge_medications=SummaryField(
            value="; ".join(med_lines) if med_lines else "Missing - clinician review required",
            status=FactStatus.verified if med_lines else FactStatus.missing,
            evidence_ids=med_evidence,
        ),
        allergies=field(("allergy",)),
        follow_up_instructions=field(("follow_up",)),
        pending_results=field(("pending_result",), "None documented - clinician review required"),
        discharge_condition=field(("discharge_condition",)),
    )


def validate_grounding(summary: DischargeSummaryDraft, evidence: list[EvidenceFact]) -> list[ReviewFlag]:
    valid_ids = {f.evidence_id for f in evidence if f.readable and f.identity_compatible}
    flags: list[ReviewFlag] = []
    for field_name in SUMMARY_FIELD_NAMES:
        summary_field = getattr(summary, field_name)
        if summary_field.status == FactStatus.verified and not summary_field.evidence_ids:
            flags.append(
                ReviewFlag(
                    flag_id=new_id("flag"),
                    category="unsupported_summary_fact",
                    message=f"{field_name} is marked verified but has no evidence citation.",
                    severity="critical",
                )
            )
        unknown = set(summary_field.evidence_ids) - valid_ids
        if unknown:
            flags.append(
                ReviewFlag(
                    flag_id=new_id("flag"),
                    category="invalid_evidence_reference",
                    message=f"{field_name} references unavailable evidence: {', '.join(sorted(unknown))}.",
                    severity="critical",
                )
            )
    return flags


def source_path_map(paths: list[Path]) -> dict[str, Path]:
    return {path.name: path for path in paths}
