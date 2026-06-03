# Discharge Summary Agent

A clinically safe, auditable agent that reads synthetic source-note PDFs and produces a structured discharge-summary draft for clinician review.

> **DRAFT FOR CLINICIAN REVIEW - NOT A FINAL CLINICAL DOCUMENT**

## Why This Design

The system uses an explicit agent loop instead of a single summarization prompt. At every step it inspects its current state, chooses one tool, records a readable decision trace, and re-plans from the tool result. Clinical content is stored in an evidence ledger before it can appear in the draft.

The core safety rule is simple: a verified summary field must cite one or more compatible source evidence IDs. Missing, pending, conflicting, unreadable, or unsupported information is not guessed.

## Stack

- Python 3.10+; Python 3.11 recommended
- Streamlit
- Google Gen AI SDK with a billing-enabled Gemini API project
- PyMuPDF
- Pydantic v2
- Pytest

Do not use unpaid Gemini quota for this assignment. Use a provider configuration with appropriate data controls, keep all records synthetic, and never commit API keys or patient PDFs.

The agent treats provider quota exhaustion and insufficient page coverage as critical safety
failures. It stops further page calls, escalates for clinician review, emits an incomplete draft,
and reports `Completed: False` instead of pretending partial extraction succeeded.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:GEMINI_API_KEY="your-key"
```

Alternatively, copy `.env.example` to `.env` and set `GEMINI_API_KEY` there. The provider loads
the local `.env` file automatically, and `.env` is excluded from Git. If you want a separate
key for optional Part 2 refinement experiments, set `GEMINI_API_KEY_02`.

Run the Streamlit demo:

```powershell
streamlit run app.py
```

Run the CLI:

```powershell
discharge-agent "C:\path\to\patient.pdf" --patient-label patient-2
```

To apply Part 2 correction-memory refinement in a CLI run:

```powershell
discharge-agent "C:\path\to\patient.pdf" --patient-label patient-2 --enable-memory-refinement
```

Artifacts are written to:

```text
outputs/<patient>/
  discharge_summary.md
  discharge_summary.json
  evidence.json
  review_flags.json
  trace.json
```

## Agent Loop

The state-aware planner chooses from these actions:

1. Inspect PDFs and build a page inventory.
2. Extract page-level evidence using embedded text or Gemini vision.
3. Detect identity and clinical-context conflicts.
4. Find required sections that remain unsupported.
5. Escalate critical concerns when present.
6. Reconcile admission and discharge medications.
7. Run a transparent mock drug-interaction lookup only when discharge medications exist.
8. Draft only from compatible evidence.
9. Optionally apply Part 2 correction-memory style refinement.
10. Validate every verified draft field against the evidence ledger.
11. Finish with all unresolved review flags visible.

The loop has a hard iteration cap. Tool failures, empty reads, and unreadable pages are recorded and surfaced instead of being treated as successful extraction.

## No-Fabrication Guardrail

Each evidence fact stores its source file, page number, excerpt, status, confidence, identity compatibility, and conflict links. The draft composer only consumes readable, compatible evidence. The grounding validator rejects verified fields without valid citations.

Important distinctions are preserved:

- `Not known` is not rewritten as `No known allergies`.
- Pending results remain pending.
- Unexplained medication changes remain reconciliation flags.
- Conflicting identity or clinical context is escalated rather than resolved by model confidence.

## Patient 2 Safety Case

The provided Patient 2 PDF is a 71-page image-only scan. It appears to contain incompatible contexts, including female clinical notes, DKA, prostate imaging, and an obstetric consultation. A safe run should flag possible cross-patient contamination and avoid producing a blended confident summary.

## Part 2 Learning From Edits

Part 2 is implemented as a safe simulated-review learning demo, not as clinical model training. A simulated reviewer applies consistent formatting edits to a draft, the app stores `(draft, corrected)` pairs in `outputs/correction_memory.json`, and later drafts can be refined from that correction memory.

**Measured improvement (both patients, `--clear-memory --deterministic`):**

| Patient | Baseline | After refinement | Oracle ceiling | Improvement |
|---|---|---|---|---|
| Patient 1 | 0.871 | 0.994 | 0.994 | +0.123 |
| Patient 2 | 0.821 | 1.000 | 1.000 | +0.179 |

The memory-based refiner matches the oracle ceiling for both patients, confirming the learning mechanism works end-to-end. Run it yourself:

```powershell
python part2_eval.py --clear-memory --deterministic
```

The reward signal is section-level `SequenceMatcher` similarity against the simulated reviewer's corrected text (1 = no edits needed, 0 = fully rewritten). The oracle score is the theoretical upper bound where reviewer corrections are applied directly. Part 1 grounding remains the safety gate: refinement never changes evidence IDs or field status, and validation runs immediately afterward.

## Limitations

- Gemini extraction can still misread handwriting; unreadable content must be reviewed manually.
- The drug-interaction tool is intentionally a mock and is not suitable for clinical use.
- Identity matching currently uses explicit identifiers and conservative context checks, not a hospital master patient index.
- Part 2 uses a simulated reviewer and formatting-style feedback only. Real clinician edits may correct clinical facts or prioritization, which would require stronger governance and validation.

## Tests

```powershell
pytest
python -m compileall src
python part2_eval.py --deterministic
```

The deterministic tests cover missing and pending fields, allergy wording, conflicts, medication reconciliation, grounding rejection, extraction failure, the iteration cap, correction memory, Part 2 refinement safety, and evaluator execution.

## Demo Outline

1. Run a cleaner synthetic patient and show the grounded draft.
2. Run Patient 2 and show the conflict escalation.
3. Open the trace at the conflict-detection step.
4. Show medication reconciliation, review flags, and evidence citations.
5. Open the Part 2 tab and show stored corrections plus before/after similarity.

## With More Time

**Part 1 improvements:**
- Replace the mock drug-interaction lookup with a real clinical API (e.g., RxNorm / OpenFDA).
- Improve identity matching to use a hospital master patient index rather than conservative keyword heuristics; this would reduce false-positive conflict flags.
- Add structured section-level confidence scoring so clinicians can see which fields came from a single source vs. multiple concordant sources.
- Cache Gemini vision results per page so re-runs or partial re-extractions do not re-bill the API.
- Expand the test suite to cover multi-PDF patients and adversarial PDF content (corrupt pages, OCR noise, overlapping patient IDs).

**Part 2 improvements:**
- Accumulate enough real clinician-edited pairs (even 20–30 per section) to fine-tune a small language model (SFT or DPO) on the `(draft, edited)` pairs rather than relying on rule-based memory lookup.
- Add a held-out evaluation split so the reported improvement curve is measured on drafts the system has never seen, not on the same patient whose corrections it just stored.
- Implement a contextual bandit over multiple prompt templates per section; reward signal = SequenceMatcher improvement; this would let the system learn which phrasing style a given institution prefers without any label supervision beyond edit distance.
- Governance layer: before a learned style change is applied to a new draft, require a second reviewer confirmation if the change touches a clinical fact (e.g., a diagnosis label), not just formatting.

## Provider References

- [Gemini document processing](https://ai.google.dev/gemini-api/docs/document-processing)
- [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini API terms](https://ai.google.dev/gemini-api/terms)
