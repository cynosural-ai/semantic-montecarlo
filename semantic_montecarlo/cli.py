"""Command-line entry point for the semantic Monte Carlo pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.observability import setup_logging
from semantic_montecarlo.pipeline import run
from semantic_montecarlo.schemas.run_result import RunResult

_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "benchmark" / "test.csv"
)
_DEFAULT_OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Run the pipeline, print its distribution as JSON, and save run artifacts."""
    parser = argparse.ArgumentParser(
        description="Estimate a numeric distribution from a factual question.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to estimate; omit to sample one from the benchmark CSV.",
    )
    parser.add_argument("--unit", help="Required unit for the numeric answers.")
    parser.add_argument("--paraphrases", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR); default INFO.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to save run artifacts (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    # First call: console logging only, so arg-resolution and errors are visible
    # before the run-specific output directory exists.
    setup_logging(args.log_level)

    question, unit = _resolve_question(args.question, args.unit, args.seed)
    print(f"Question: {question}\nUnit: {unit or 'not specified'}", file=sys.stderr)

    # Resolve the run directory up front so the file handler captures the whole
    # run, then reconfigure logging to also write to run.log.
    run_dir = _run_dir(args.output_dir)
    setup_logging(args.log_level, log_file=run_dir / "run.log")

    result = run(
        question,
        client=LLMClient(),
        unit=unit,
        n_paraphrases=args.paraphrases,
        n_resamples=args.resamples,
        seed=args.seed,
    )

    _save_run(
        result,
        run_dir,
        parameters={
            "n_paraphrases": args.paraphrases,
            "n_resamples": args.resamples,
            "seed": args.seed,
        },
    )
    # Headline output to stdout: just the distribution.
    print(result.distribution.model_dump_json(indent=2))


def _run_dir(output_dir: Path) -> Path:
    """Create and return a timestamped subdirectory under ``output_dir``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run(
    result: RunResult,
    run_dir: Path,
    parameters: dict[str, object],
) -> None:
    """
    Persist run artifacts: ``result.json`` (summary) and ``searches.json``.

    Args:
        result: The :class:`RunResult` from the pipeline.
        run_dir: Timestamped directory to write into.
        parameters: CLI invocation parameters, recorded in ``result.json``.
    """
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": result.model,
        "elapsed_seconds": result.elapsed_seconds,
        "question": result.question,
        "unit": result.unit,
        "parameters": parameters,
        "distribution": [
            {"value": value, "probability": probability}
            for value, probability in result.distribution.data
        ],
    }
    (run_dir / "result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run_dir / "searches.json").write_text(
        json.dumps([a.model_dump() for a in result.answers], indent=2),
        encoding="utf-8",
    )


def _resolve_question(
    question: str | None,
    unit: str | None,
    seed: int | None,
) -> tuple[str, str | None]:
    rows = _load_benchmark()
    if question is None:
        row = random.Random(seed).choice(rows)
        return row["question"], row["answer_unit"]

    if unit is not None:
        return question, unit

    normalized_question = _normalize(question)
    matching_row = next(
        (row for row in rows if _normalize(row["question"]) == normalized_question),
        None,
    )
    return (
        (question, matching_row["answer_unit"])
        if matching_row is not None
        else (question, None)
    )


def _load_benchmark() -> list[dict[str, str]]:
    with _BENCHMARK_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Benchmark dataset is empty: {_BENCHMARK_PATH}")
    return rows


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
