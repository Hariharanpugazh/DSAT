from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import fitz

from discharge_agent.agent import DischargeSummaryAgent
from discharge_agent.models import (
    DischargeSummaryDraft,
    ExtractedFact,
    FactStatus,
    PageExtraction,
    SummaryField,
)
from discharge_agent.part2.correction_memory import CorrectionMemory
from discharge_agent.part2.refinement import refine_summary_with_memory, refine_text_with_examples
from discharge_agent.part2.simulated_reviewer import SimulatedReviewer


def summary_with(**overrides: SummaryField) -> DischargeSummaryDraft:
    missing = SummaryField(value="Missing - clinician review required", status=FactStatus.missing)
    values = {
        "patient_demographics": missing,
        "admission_date": missing,
        "discharge_date": missing,
        "principal_diagnosis": missing,
        "secondary_diagnoses": missing,
        "hospital_course": missing,
        "procedures": missing,
        "discharge_medications": missing,
        "allergies": missing,
        "follow_up_instructions": missing,
        "pending_results": SummaryField(
            value="None documented - clinician review required",
            status=FactStatus.missing,
        ),
        "discharge_condition": missing,
    }
    values.update(overrides)
    return DischargeSummaryDraft(**values)


def test_correction_memory_skips_identical_and_duplicate_examples(tmp_path: Path) -> None:
    memory = CorrectionMemory(tmp_path / "memory.json")
    memory.add("procedures", "CBC", "CBC")
    memory.add("procedures", "CBC; GRBS", "- CBC\n- GRBS")
    memory.add("procedures", "CBC; GRBS", "- CBC\n- GRBS")
    assert memory.total_examples() == 1


def test_simulated_reviewer_preserves_safety_placeholders_and_not_known() -> None:
    draft = summary_with(
        principal_diagnosis=SummaryField(
            value="Conflicting - clinician review required",
            status=FactStatus.conflicting,
        ),
        allergies=SummaryField(value="Not known", status=FactStatus.verified, evidence_ids=["ev-1"]),
    )
    review = SimulatedReviewer().review(draft)
    assert review.corrected_sections["principal_diagnosis"] == "Conflicting - clinician review required"
    assert review.corrected_sections["allergies"] == "Not known"


def test_deterministic_refiner_learns_formatting_without_exact_replay(tmp_path: Path) -> None:
    memory = CorrectionMemory(tmp_path / "memory.json")
    memory.add("procedures", "CBC; GRBS", "- CBC\n- GRBS")
    examples = memory.get_examples("procedures")

    refined = refine_text_with_examples(
        "procedures",
        "IV cannula; ECG; CBC",
        examples,
        allow_exact_replay=False,
    )

    assert refined == "- IV cannula\n- ECG\n- CBC"


def test_refiner_preserves_field_status_and_evidence_ids(tmp_path: Path) -> None:
    memory = CorrectionMemory(tmp_path / "memory.json")
    memory.add("hospital_course", "A; B", "A. B.")
    draft = summary_with(
        hospital_course=SummaryField(
            value="IV fluids; improved",
            status=FactStatus.verified,
            evidence_ids=["ev-1"],
        )
    )

    report = refine_summary_with_memory(draft, memory)
    field = report.draft.hospital_course

    assert field.value == "IV fluids. improved."
    assert field.status == FactStatus.verified
    assert field.evidence_ids == ["ev-1"]


class Part2Provider:
    def extract_page(self, page, source_path):
        return PageExtraction(
            facts=[
                ExtractedFact(
                    category="hospital_course",
                    value="IV fluids; improved",
                    excerpt="IV fluids; improved",
                )
            ]
        )


def make_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Synthetic note")
    document.save(path)
    document.close()


def test_agent_refines_with_memory_before_grounding_validation(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.json"
    memory = CorrectionMemory(memory_path)
    memory.add("hospital_course", "A; B", "A. B.")
    pdf = tmp_path / "patient.pdf"
    make_pdf(pdf)

    result = DischargeSummaryAgent(
        Part2Provider(),
        max_steps=20,
        memory_path=memory_path,
    ).run("patient", [pdf])

    actions = [step.action for step in result.trace]
    assert actions.index("refine_draft_with_memory") < actions.index("validate_grounding")
    assert result.summary is not None
    assert result.summary.hospital_course.value == "IV fluids. improved."
    assert result.summary.hospital_course.evidence_ids


def test_part2_eval_runs_deterministically_without_console_unicode_crash(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    patient_dir = output_dir / "patient"
    patient_dir.mkdir(parents=True)
    draft = summary_with(
        procedures=SummaryField(value="CBC; GRBS", status=FactStatus.verified, evidence_ids=["ev-1"])
    )
    (patient_dir / "discharge_summary.json").write_text(
        draft.model_dump_json(indent=2),
        encoding="utf-8",
    )

    results_path = tmp_path / "part2_results.json"
    proc = subprocess.run(
        [
            sys.executable,
            "part2_eval.py",
            "--patients",
            "patient",
            "--output-dir",
            str(output_dir),
            "--memory-path",
            str(tmp_path / "memory.json"),
            "--results-path",
            str(results_path),
            "--clear-memory",
            "--deterministic",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert results_path.exists()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["refinement_mode"] == "deterministic"
