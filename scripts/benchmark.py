"""Run the pipeline against the held-out benchmark dataset."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_montecarlo.llm.client import LLMClient
from semantic_montecarlo.observability import get_logger, setup_logging
from semantic_montecarlo.pipeline.run import run
from semantic_montecarlo.schemas.run_result import RunResult
from semantic_montecarlo.stats import norm_var_comp

_logger = get_logger(__name__)


def benchmark(
    estimate: Callable[[str], RunResult],
) -> pd.DataFrame:
    """Evaluate an estimator and persist row-level benchmark results."""
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "benchmark" / "test.csv",
    )

    df["solution"] = df.apply(
        lambda row: (
            estimate(
                f"{row['question']}\nAnswer in the following unit: {row['answer_unit']}"
            ).distribution
        ),
        axis=1,
    )

    # norm_var_comp already returns the complement, so do not invert it.
    df["estimated_confidence"] = df["solution"].map(norm_var_comp)

    df["expected_confidence"] = df["confidence_mean"].astype(float) / 100.0

    df["squared_error"] = np.square(
        df["expected_confidence"] - df["estimated_confidence"]
    )

    mse = float(df["squared_error"].mean())

    _logger.info("Benchmark results:\n%s", df.to_string(index=False))
    _logger.info("MSE: %.6f", mse)

    df.to_csv(
        Path(__file__).resolve().parent.parent
        / "data"
        / "benchmark"
        / "num_paraphrases.csv",
        index=False,
    )
    return df


if __name__ == "__main__":
    setup_logging("INFO")
    client = LLMClient()

    estimate: Callable[[str], RunResult] = partial(
        run,
        client=client,
        n_paraphrases=5,
        n_resamples=10_000,
        seed=None,
    )

    benchmark(estimate)
