from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from discharge_agent.agent import DischargeSummaryAgent
from discharge_agent.provider import GeminiExtractionProvider
from discharge_agent.rendering import summary_to_markdown, write_outputs
from discharge_agent.part2.simulated_reviewer import SimulatedReviewer
from discharge_agent.part2.correction_memory import CorrectionMemory, DEFAULT_MEMORY_PATH


st.set_page_config(page_title="Discharge Summary Agent", layout="wide")
st.title("Discharge Summary Agent")
st.error("DRAFT FOR CLINICIAN REVIEW - NOT A FINAL CLINICAL DOCUMENT")
st.caption("Every clinical fact must be grounded in source evidence. Unknowns remain unknown.")

tab_agent, tab_part2 = st.tabs(["Part 1 — Agent", "Part 2 — Learning from Edits"])

# ---------------------------------------------------------------------------
# Part 1 tab
# ---------------------------------------------------------------------------
with tab_agent:
    patient_label = st.text_input("Patient label", value="patient")
    uploads = st.file_uploader("Upload synthetic source-note PDFs", type="pdf", accept_multiple_files=True)
    max_steps = st.number_input("Maximum agent steps", min_value=1, max_value=50, value=20)
    enable_memory = st.checkbox(
        "Enable correction-memory refinement (Part 2)",
        value=DEFAULT_MEMORY_PATH.exists(),
        help=(
            "When checked, the agent will apply a style-refinement step using "
            "past reviewer corrections stored in outputs/correction_memory.json. "
            "The grounding validator still runs after refinement."
        ),
    )

    if st.button("Run agent", type="primary", disabled=not uploads):
        try:
            with tempfile.TemporaryDirectory(prefix="discharge-agent-") as temp_dir:
                paths = []
                for upload in uploads:
                    path = Path(temp_dir) / upload.name
                    path.write_bytes(upload.getvalue())
                    paths.append(path)
                with st.status("Running agent...", expanded=True) as status:
                    provider = GeminiExtractionProvider()
                    memory_path = DEFAULT_MEMORY_PATH if enable_memory else None
                    result = DischargeSummaryAgent(
                        provider, int(max_steps), memory_path=memory_path
                    ).run(patient_label, paths)
                    output_dir = write_outputs(result, Path("outputs"))
                    status.update(label="Agent run complete", state="complete")
        except Exception as exc:
            st.exception(exc)
        else:
            left, right = st.columns([2, 1])
            with left:
                st.subheader("Draft")
                if result.summary:
                    st.markdown(summary_to_markdown(result.summary))
                else:
                    st.warning("No draft was produced.")
            with right:
                st.subheader("Review Flags")
                if not result.review_flags:
                    st.success("No review flags were generated.")
                for flag in result.review_flags:
                    st.warning(f"**{flag.severity.upper()} - {flag.category}**\n\n{flag.message}")
                st.caption(f"Artifacts written to `{output_dir}`")

            st.subheader("Medication Reconciliation")
            st.dataframe(
                [change.model_dump(mode="json") for change in result.medication_changes],
                use_container_width=True,
            )

            st.subheader("Evidence Ledger")
            st.dataframe(
                [fact.model_dump(mode="json") for fact in result.evidence],
                use_container_width=True,
            )

            st.subheader("Agent Trace")
            for step in result.trace:
                with st.expander(f"Step {step.step_number}: {step.action}", expanded=False):
                    st.write(f"**Decision rationale:** {step.rationale}")
                    st.write(f"**Inputs:** {step.inputs}")
                    st.write(f"**Result:** {step.result}")
                    st.write(f"**Next decision:** {step.next_decision}")
                    if step.error:
                        st.error(step.error)

# ---------------------------------------------------------------------------
# Part 2 tab
# ---------------------------------------------------------------------------
with tab_part2:
    st.subheader("Part 2 — Learning from Doctor Edits")
    st.markdown(
        """
The simulated reviewer applies a **consistent hidden editing policy** to the agent's
draft, producing `(draft, corrected)` pairs. Those pairs are stored in a correction
memory and injected as few-shot examples on subsequent agent runs.

**Reward signal:** normalised SequenceMatcher similarity per section (1 = no edits needed).

**Safety guarantee:** the grounding validator (Part 1) still runs *after* any
memory-based refinement. Missing and Conflicting sections are never touched.
        """
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Step 1 — Review a completed draft")
        p2_patient_label = st.selectbox(
            "Choose a patient whose draft has already been generated",
            options=sorted(
                [d.name for d in Path("outputs").iterdir() if d.is_dir() and (d / "discharge_summary.json").exists()]
            ) if Path("outputs").exists() else [],
            index=0,
        )

        if st.button("Apply simulated reviewer & store corrections", type="primary"):
            draft_path = Path("outputs") / p2_patient_label / "discharge_summary.json"
            if not draft_path.exists():
                st.error(f"No discharge_summary.json found for '{p2_patient_label}'.")
            else:
                from discharge_agent.models import DischargeSummaryDraft
                draft = DischargeSummaryDraft.model_validate(
                    json.loads(draft_path.read_text(encoding="utf-8"))
                )
                reviewer = SimulatedReviewer()
                review_result = reviewer.review(draft)

                memory = CorrectionMemory(DEFAULT_MEMORY_PATH)
                draft_values = {name: getattr(draft, name).value for name in [s.section for s in review_result.scores]}
                added = memory.add_from_review_result(review_result.corrected_sections, draft_values)

                st.success(
                    f"Reviewer applied. Overall similarity: **{review_result.overall_similarity:.3f}**  "
                    f"({added} new correction(s) stored; {memory.total_examples()} total in memory)"
                )

                st.markdown("#### Section Scores (similarity before learning)")
                score_rows = [
                    {"section": s.section, "similarity": round(s.similarity, 3), "draft_snippet": s.draft[:80], "corrected_snippet": s.corrected[:80]}
                    for s in review_result.scores
                ]
                st.dataframe(score_rows, use_container_width=True)

    with col_right:
        st.markdown("### Step 2 — Correction Memory")
        memory = CorrectionMemory(DEFAULT_MEMORY_PATH)
        total = memory.total_examples()
        sections = memory.sections_with_examples()
        if total == 0:
            st.info("No corrections stored yet. Run Step 1 first.")
        else:
            st.metric("Total stored corrections", total)
            st.metric("Sections with examples", len(sections))
            for sec in sections:
                examples = memory.get_examples(sec)
                with st.expander(f"{sec} ({len(examples)} example(s))"):
                    for i, ex in enumerate(examples, 1):
                        st.markdown(f"**Example {i}**")
                        st.text(f"BEFORE: {ex.draft[:200]}")
                        st.text(f"AFTER:  {ex.corrected[:200]}")

    st.markdown("---")
    st.markdown("### Step 3 — Before / After Results")
    results_path = Path("outputs") / "part2_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        st.metric("Memory examples total", results["memory_examples_total"])
        for patient_result in results["per_patient"]:
            with st.expander(f"Patient: {patient_result['patient']}"):
                st.metric(
                    "Baseline overall similarity",
                    f"{patient_result['baseline_overall']:.3f}",
                )
                st.metric(
                    "Post-refinement overall similarity",
                    f"{patient_result['refined_overall']:.3f}",
                    delta=f"{patient_result['refined_overall'] - patient_result['baseline_overall']:+.3f}",
                )
                rows = []
                for sec in patient_result["baseline"]:
                    rows.append({
                        "section": sec,
                        "baseline": patient_result["baseline"].get(sec, "-"),
                        "refined": patient_result["refined"].get(sec, "-"),
                    })
                st.dataframe(rows, use_container_width=True)

        st.markdown("#### Improvement Curve")
        curve_data = [
            {"patient": r["patient"], "baseline": r["baseline_overall"], "refined": r["refined_overall"]}
            for r in results["iteration_scores"]
        ]
        st.line_chart(
            {
                "Baseline": [r["baseline"] for r in curve_data],
                "Refined": [r["refined"] for r in curve_data],
            }
        )

        st.markdown("#### Limitations")
        for k, v in results.get("limitations", {}).items():
            with st.expander(k.replace("_", " ").title()):
                st.write(v)

        if st.button("Re-run part2_eval.py (baseline scores only)"):
            import subprocess, sys
            proc = subprocess.run(
                [sys.executable, "part2_eval.py", "--skip-refinement"],
                capture_output=True,
                text=True,
            )
            st.code(proc.stdout or proc.stderr)
    else:
        st.info(
            "No `outputs/part2_results.json` found yet. "
            "Run `python part2_eval.py` from the terminal, then refresh."
        )


