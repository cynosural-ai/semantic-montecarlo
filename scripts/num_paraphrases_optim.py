from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semantic_montecarlo.llm.client import LLMClient
from semantic_montecarlo.observability import get_logger, setup_logging
from semantic_montecarlo.pipeline.run import run
from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.stats import norm_var_comp

_logger = get_logger(__name__)


EstimateFunction = Callable[..., Distribution]


def benchmark_paraphrases(
    estimate: EstimateFunction,
    paraphrase_values: Sequence[int] = (0, 2, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate estimate() for several paraphrases parameter values.

    Args:
        estimate:
            Function with a compatible signature such as:

                estimate(question, paraphrases=number)

        paraphrase_values:
            Values to test for the paraphrases parameter.

    Returns:
        results_df:
        One row per benchmark test case and paraphrases value.

        summary_df:
        Aggregate metrics for each paraphrases value.
    """

    source_df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "benchmark" / "eval.csv"
    )

    records: list[dict[str, Any]] = []

    for row in source_df.itertuples(index=False):
        row_data = row._asdict()

        question = (
            f"{pd.Series(row_data)['question']}\n"
            f"Answer in the following unit: {pd.Series(row_data)['answer_unit']}"
        )

        expected_confidence = (
            float(row_data["confidence_mean"]) / 100.0
        )

        for paraphrases in paraphrase_values:
            distribution = estimate(
                question,
                paraphrases=int(paraphrases),
            )

            estimated_confidence = norm_var_comp(
                distribution
            )

            difference = (
                estimated_confidence
                - expected_confidence
            )

            record = {
                **row_data,
                "benchmark_question": question,
                "paraphrases": int(paraphrases),

                # Keep both forms when working in memory.
                "distribution": distribution,
                "distribution_data": [
                    (float(x), float(probability))
                    for x, probability in distribution.data
                ],

                "expected_confidence": expected_confidence,
                "estimated_confidence": estimated_confidence,

                # Signed error:
                # positive means estimated confidence is too high.
                "difference": difference,

                "absolute_error": abs(difference),
                "squared_error": difference**2,
            }

            records.append(record)

    results_df = pd.DataFrame.from_records(records)

    summary_df = (
        results_df.groupby(
            "paraphrases",
            as_index=False,
        )
        .agg(
            test_rows=("test_row", "count"),
            mean_expected_confidence=(
                "expected_confidence",
                "mean",
            ),
            mean_estimated_confidence=(
                "estimated_confidence",
                "mean",
            ),
            mean_difference=("difference", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
        )
    )

    summary_df["rmse"] = np.sqrt(
        summary_df["mse"]
    )

    summary_df = summary_df.sort_values(
        ["mse", "paraphrases"],
        ascending=[True, True],
    ).reset_index(drop=True)

    _logger.info(
        "Row-level results:\n%s",
        results_df[
            [
                "test_row",
                "paraphrases",
                "expected_confidence",
                "estimated_confidence",
                "difference",
                "absolute_error",
                "squared_error",
                "distribution_data",
            ]
        ].to_string(index=False),
    )

    _logger.info("Parameter summary:\n%s", summary_df.to_string(index=False))

    best_row = summary_df.iloc[0]

    _logger.info(
        "Best paraphrases value by MSE: %d",
        int(best_row["paraphrases"]),
    )
    _logger.info("MSE: %.6f", best_row["mse"])
    _logger.info("RMSE: %.6f", best_row["rmse"])

    results_df.to_csv(
        Path(__file__).resolve().parent.parent / "data" / "benchmark" / "num_paraphrases.csv",
        index=False
    )
    summary_df.to_csv(
        Path(__file__).resolve().parent.parent / "data" / "benchmark" / "num_paraphrases_summary.csv",
        index=False
    )


if __name__ == "__main__":
    setup_logging("INFO")
    client = LLMClient()

    estimate: Callable[[str], Distribution] = partial(
        run,
        client=client,
        n_paraphrases=5,
        n_resamples=10_000,
        seed=None,
    )

    results_df, summary_df = benchmark_paraphrases(
        estimate=estimate,
        paraphrase_values=[0, 2, 5, 10],
    )
