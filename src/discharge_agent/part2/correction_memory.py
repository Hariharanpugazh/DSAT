"""Correction Memory — Part 2 learning mechanism.

Stores accumulated (draft_text, corrected_text) pairs per discharge-summary
section in a JSON file.  During drafting, stored examples are injected as
few-shot context into the Gemini refinement prompt so the agent can generalise
the reviewer's style without ever seeing the reviewer's source code.

Schema of the backing JSON file:
  {
    "<section_name>": [
      {"draft": "<raw agent draft>", "corrected": "<reviewer corrected>"},
      ...
    ],
    ...
  }

Safety note: this module only stores and retrieves text.  It never modifies
evidence IDs, evidence facts, or any field that the grounding validator checks.
The grounding validator (Part 1, unchanged) runs after the memory-based
refinement step, so any hallucinated content that somehow acquired a verified
status would still be caught and flagged as critical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


DEFAULT_MEMORY_PATH = Path("outputs") / "correction_memory.json"


class CorrectionExample(NamedTuple):
    draft: str
    corrected: str


class CorrectionMemory:
    """Persistent store of (draft, corrected) pairs per section.

    Args:
        path: Path to the JSON backing file.  Created automatically on first
              ``add`` call if it does not exist.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_MEMORY_PATH
        self._data: dict[str, list[dict[str, str]]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, section: str, draft: str, corrected: str) -> None:
        """Store a correction pair.  Skips identical pairs and exact duplicates."""
        if draft.strip() == corrected.strip():
            return  # reviewer made no change — nothing to learn
        if section not in self._data:
            self._data[section] = []
        # Avoid exact duplicate (draft, corrected) pairs
        already_stored = any(
            e["draft"] == draft and e["corrected"] == corrected
            for e in self._data[section]
        )
        if not already_stored:
            self._data[section].append({"draft": draft, "corrected": corrected})
            self._persist()

    def add_from_review_result(
        self,
        corrected_sections: dict[str, str],
        draft_sections: dict[str, str],
    ) -> int:
        """Bulk-add corrections from a reviewer pass.  Returns number of new entries."""
        added = 0
        for section, corrected_text in corrected_sections.items():
            draft_text = draft_sections.get(section, "")
            if draft_text and corrected_text != draft_text:
                prev_len = len(self._data.get(section, []))
                self.add(section, draft_text, corrected_text)
                if len(self._data.get(section, [])) > prev_len:
                    added += 1
        return added

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_examples(self, section: str, n: int = 3) -> list[CorrectionExample]:
        """Return up to *n* most-recently added examples for a section."""
        entries = self._data.get(section, [])
        return [CorrectionExample(**e) for e in entries[-n:]]

    def has_examples(self, section: str) -> bool:
        return bool(self._data.get(section))

    def sections_with_examples(self) -> list[str]:
        return [s for s, examples in self._data.items() if examples]

    def total_examples(self) -> int:
        return sum(len(v) for v in self._data.values())

    def as_dict(self) -> dict[str, list[dict[str, str]]]:
        """Return a copy of the backing data (for serialisation/display)."""
        return {k: list(v) for k, v in self._data.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
