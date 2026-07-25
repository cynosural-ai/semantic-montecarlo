"""Command-line entry point for the semantic Monte Carlo pipeline."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.observability import setup_logging
from semantic_montecarlo.pipeline import run

_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "benchmark" / "test.csv"
)


def main() -> None:
    """Run the pipeline and print its estimated distribution as JSON."""
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
    args = parser.parse_args()

    setup_logging(args.log_level)

    question, unit = _resolve_question(args.question, args.unit, args.seed)
    print(f"Question: {question}\nUnit: {unit or 'not specified'}", file=sys.stderr)

    result = run(
        question,
        client=LLMClient(),
        unit=unit,
        n_paraphrases=args.paraphrases,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    print(result.model_dump_json(indent=2))


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
