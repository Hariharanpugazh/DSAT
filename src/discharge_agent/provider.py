from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from google import genai
from google.genai import types
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .ingestion import render_page
from .models import DocumentPage, PageExtraction


EXTRACTION_PROMPT = """You extract only explicitly documented clinical evidence from one source-note page.
Do not infer, diagnose, normalize unknown allergies into no allergies, or resolve contradictions.
Return facts useful for a discharge summary. Use short exact excerpts as evidence.
For handwriting that is not reliably readable, omit the fact or mark the page unreadable.
Categories should use these names when applicable:
patient_name, patient_id, age, sex, admission_date, discharge_date, principal_diagnosis,
secondary_diagnosis, hospital_course, procedure, allergy, follow_up, pending_result,
discharge_condition, clinical_context.
Always include clinical_context facts for major diagnoses, sex-specific anatomy, pregnancy or
obstetric findings, and any clue that could indicate the page belongs to a different patient.
Medication phase must be admission, discharge, inpatient, or unknown.
"""


class ExtractionProvider(Protocol):
    def extract_page(self, page: DocumentPage, source_path: Path) -> PageExtraction: ...


class GeminiExtractionProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required for Gemini extraction.")
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def refine_section_with_memory(
        self,
        section_name: str,
        draft_value: str,
        examples: list,  # list[CorrectionExample] — kept as list to avoid circular import
    ) -> str:
        """Reformat *draft_value* using few-shot style examples from correction memory.

        SAFETY CONTRACT:
        - The prompt explicitly forbids adding any new clinical fact, medication,
          diagnosis, date, lab value, or number not already present in the input.
        - Missing/Conflicting sections are passed through unchanged.
        - The grounding validator (Part 1, unmodified) still runs after this step.
        - If Gemini returns empty or raises, the original draft_value is returned
          unchanged so the agent never silently loses content.
        """
        if not examples or "Missing" in draft_value or "Conflicting" in draft_value:
            return draft_value

        few_shot_lines = []
        for ex in examples:
            few_shot_lines.append(f"BEFORE: {ex.draft}")
            few_shot_lines.append(f"AFTER: {ex.corrected}")
            few_shot_lines.append("")
        few_shot_block = "\n".join(few_shot_lines).strip()

        prompt = (
            "You are a clinical document formatter. "
            "Reformat the INPUT text using the STYLE shown in the examples below.\n\n"
            "NON-NEGOTIABLE RULES:\n"
            "1. Do NOT add any new clinical fact, medication, diagnosis, date, lab value, "
            "or patient name that is not already present.\n"
            "2. Do NOT remove clinical information.\n"
            "3. Only change: formatting (bullets, sentence structure), "
            "capitalisation, punctuation, and removing inline citation tags like (ev-...).\n"
            "4. If the text contains the word 'Missing' or 'Conflicting', "
            "output it EXACTLY unchanged.\n"
            "5. Output ONLY the reformatted text. No preamble, no labels.\n\n"
            "STYLE EXAMPLES:\n"
        )
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n  Input:  {ex.draft}\n  Output: {ex.corrected}\n\n"
        prompt += f"Now reformat:\n  Input:  {draft_value}\n  Output:"

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0),
            )
            refined = (response.text or "").strip()
            # Safety gate: if Gemini hallucinated a placeholder, revert
            if not refined or len(refined) > len(draft_value) * 4:
                return draft_value
            return refined
        except Exception:
            # Never propagate refinement failures — fall back to original
            return draft_value

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def extract_page(self, page: DocumentPage, source_path: Path) -> PageExtraction:
        if page.text:
            content = [EXTRACTION_PROMPT, f"\nPAGE TEXT:\n{page.text}"]
        else:
            image = render_page(source_path, page.page_number)
            content = [
                EXTRACTION_PROMPT,
                types.Part.from_bytes(data=image, mime_type="image/png"),
            ]
        response = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PageExtraction,
                temperature=0,
            ),
        )
        if not response.text:
            raise ValueError("Gemini returned an empty extraction.")
        return PageExtraction.model_validate(json.loads(response.text))
