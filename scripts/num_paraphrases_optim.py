"""Select the paraphrase count that minimizes mean squared confidence error."""

from collections.abc import Callable, Sequence
from functools import partial
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semantic_montecarlo.llm.client import LLMClient
from semantic_montecarlo.observability import get_logger, setup_logging
from semantic_montecarlo.pipeline.run import run
from semantic_montecarlo.schemas.run_result import RunResult
from semantic_montecarlo.stats import norm_var_comp

_logger = get_logger(__name__)


EstimateFunction = Callable[..., RunResult]


def save_results_atomic(
    results_df: pd.DataFrame,
    results_path: Path,
) -> None:
    """Save results without leaving a partially written cache file."""
    temporary_path = results_path.with_suffix(".tmp")
    results_df.to_csv(temporary_path, index=False)
    temporary_path.replace(results_path)


def benchmark_paraphrases(
    estimate: EstimateFunction,
    paraphrase_values: Sequence[int] = (0, 2, 5, 10),
) -> pd.DataFrame:
    """
    Evaluate estimate() for several n_paraphrases values.

    Existing results are loaded from num_paraphrases.csv and reused.
    """
    paraphrase_values = tuple(paraphrase_values)

    if not paraphrase_values:
        raise ValueError("paraphrase_values must contain at least one value")

    benchmark_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "benchmark"
    )

    source_path = benchmark_dir / "eval.csv"
    results_path = benchmark_dir / "results" / "num_paraphrases.csv"

    source_df = pd.read_csv(source_path)

    if results_path.exists():
        cached_df = pd.read_csv(results_path)

        # Normalize key types before constructing the cache lookup.
        cached_df["paraphrases"] = cached_df["paraphrases"].astype(int)

        _logger.info(
            "Loaded %d cached benchmark results from %s",
            len(cached_df),
            results_path,
        )
    else:
        cached_df = pd.DataFrame()
        _logger.info("No existing cache found at %s", results_path)

    cached_keys: set[tuple[Any, int]] = set()

    if not cached_df.empty:
        cached_keys = {
            (row.id, int(row.paraphrases))
            for row in cached_df[
                ["id", "paraphrases"]
            ].itertuples(index=False)
        }

    new_records: list[dict[str, Any]] = []

    for row_data in source_df.to_dict(orient="records"):
        question = (
            f"{row_data['question']}\n"
            f"Answer in the following unit: {row_data['answer_unit']}"
        )

        expected_confidence = (
            float(row_data["confidence_mean"]) / 100.0
        )

        for paraphrases in paraphrase_values:
            cache_key = (
                row_data["id"],
                paraphrases,
            )

            if cache_key in cached_keys:
                _logger.info(
                    "Cache hit: id=%s, paraphrases=%d",
                    row_data["id"],
                    paraphrases,
                )
                continue

            _logger.info(
                "Computing: id=%s, paraphrases=%d",
                row_data["id"],
                paraphrases,
            )

            run_result = estimate(
                question,
                n_paraphrases=paraphrases,
            )

            bootstrap_mean = run_result.bootstrap_mean

            estimated_confidence = norm_var_comp(
                bootstrap_mean
            )

            difference = estimated_confidence - expected_confidence

            bootstrap_mean_data = [
                (float(x), float(probability))
                for x, probability in bootstrap_mean.data
            ]

            new_records.append(
                {
                    **row_data,
                    "benchmark_question": question,
                    "paraphrases": paraphrases,
                    # JSON is stable and can be loaded again later.
                    "bootstrap_mean_data": json.dumps(
                        bootstrap_mean_data
                    ),
                    "no_answer_probability": float(
                        bootstrap_mean.no_answer_probability
                    ),
                    "expected_confidence": expected_confidence,
                    "estimated_confidence": estimated_confidence,
                    "difference": difference,
                    "absolute_error": abs(difference),
                    "squared_error": difference**2,
                }
            )

            new_results_df = pd.DataFrame.from_records(new_records)

            if cached_df.empty:
                checkpoint_df = new_results_df
            else:
                checkpoint_df = pd.concat(
                    [cached_df, new_results_df],
                    ignore_index=True,
                    sort=False,
                )

            checkpoint_df = (
                checkpoint_df.drop_duplicates(
                    subset=["id", "paraphrases"],
                    keep="last",
                )
                .sort_values(
                    ["id", "paraphrases"],
                    kind="stable",
                )
                .reset_index(drop=True)
            )

            save_results_atomic(
                checkpoint_df,
                results_path,
            )

            _logger.info(
                "Saved checkpoint: id=%s, paraphrases=%d",
                row_data["id"],
                paraphrases,
            )

    new_results_df = pd.DataFrame.from_records(new_records)

    if cached_df.empty:
        results_df = new_results_df
    elif new_results_df.empty:
        results_df = cached_df.copy()
    else:
        results_df = pd.concat(
            [cached_df, new_results_df],
            ignore_index=True,
            sort=False,
        )

    if results_df.empty:
        raise ValueError("No benchmark results were available or computed")

    # Remove accidental duplicate cache rows. Keep the most recent row.
    results_df = (
        results_df.drop_duplicates(
            subset=["id", "paraphrases"],
            keep="last",
        )
        .sort_values(
            ["id", "paraphrases"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    _logger.info(
        "Computed %d new results; reused %d cached results",
        len(new_results_df),
        len(results_df) - len(new_results_df),
    )

    _logger.info(
        "Row-level results:\n%s",
        results_df[
            [
                "id",
                "paraphrases",
                "expected_confidence",
                "estimated_confidence",
                "difference",
                "absolute_error",
                "squared_error",
                "bootstrap_mean_data",
                "no_answer_probability",
            ]
        ].to_string(index=False),
    )

    save_results_atomic(results_df, results_path)

    return results_df


if __name__ == "__main__":
    setup_logging("DEBUG")
    client = LLMClient()

    estimate: EstimateFunction = partial(
        run,
        client=client,
        n_resamples=10_000,
        seed=None,
    )

    benchmark_paraphrases(
        estimate=estimate,
    )