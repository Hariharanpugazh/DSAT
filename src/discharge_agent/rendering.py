from __future__ import annotations

import json
from pathlib import Path

from .models import AgentRunResult, DischargeSummaryDraft


FIELD_LABELS = {
    "patient_demographics": "Patient Demographics",
    "admission_date": "Admission Date",
    "discharge_date": "Discharge Date",
    "principal_diagnosis": "Principal Diagnosis",
    "secondary_diagnoses": "Secondary Diagnoses",
    "hospital_course": "Hospital Course",
    "procedures": "Procedures",
    "discharge_medications": "Discharge Medications",
    "allergies": "Allergies",
    "follow_up_instructions": "Follow-up Instructions",
    "pending_results": "Pending Results",
    "discharge_condition": "Discharge Condition",
}


def summary_to_markdown(summary: DischargeSummaryDraft) -> str:
    lines = [f"# {summary.disclaimer}", ""]
    for name in FIELD_LABELS:
        field = getattr(summary, name)
        citations = f" ({', '.join(field.evidence_ids)})" if field.evidence_ids else ""
        lines.extend([f"## {FIELD_LABELS[name]}", f"{field.value}{citations}", ""])
    return "\n".join(lines)


def write_outputs(result: AgentRunResult, output_root: Path) -> Path:
    output_dir = output_root / result.patient_label
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.summary:
        (output_dir / "discharge_summary.md").write_text(
            summary_to_markdown(result.summary), encoding="utf-8"
        )
        (output_dir / "discharge_summary.json").write_text(
            result.summary.model_dump_json(indent=2), encoding="utf-8"
        )
    (output_dir / "trace.json").write_text(
        json.dumps([step.model_dump(mode="json") for step in result.trace], indent=2),
        encoding="utf-8",
    )
    (output_dir / "review_flags.json").write_text(
        json.dumps([flag.model_dump(mode="json") for flag in result.review_flags], indent=2),
        encoding="utf-8",
    )
    (output_dir / "evidence.json").write_text(
        json.dumps([fact.model_dump(mode="json") for fact in result.evidence], indent=2),
        encoding="utf-8",
    )
    return output_dir
