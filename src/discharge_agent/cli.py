from __future__ import annotations

import argparse
from pathlib import Path

from .agent import DischargeSummaryAgent
from .part2.correction_memory import DEFAULT_MEMORY_PATH
from .provider import GeminiExtractionProvider
from .rendering import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the discharge-summary drafting agent.")
    parser.add_argument("pdfs", nargs="+", type=Path, help="Source-note PDF files")
    parser.add_argument("--patient-label", default="patient", help="Output folder label")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default=None, help="Gemini model override")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--enable-memory-refinement",
        action="store_true",
        help="Apply Part 2 correction-memory style refinement before grounding validation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    missing = [str(path) for path in args.pdfs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing PDF files: {', '.join(missing)}")
    provider = GeminiExtractionProvider(model=args.model)
    memory_path = DEFAULT_MEMORY_PATH if args.enable_memory_refinement else None
    agent = DischargeSummaryAgent(provider=provider, max_steps=args.max_steps, memory_path=memory_path)
    result = agent.run(args.patient_label, args.pdfs)
    output_dir = write_outputs(result, args.output_dir)
    print(f"Completed: {result.completed}")
    print(f"Review flags: {len(result.review_flags)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
