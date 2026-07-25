from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_montecarlo.llm.client import LLMClient
from semantic_montecarlo.observability import get_logger, setup_logging
from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.stats import norm_var_comp
from semantic_montecarlo.pipeline.run import run


_logger = get_logger(__name__)


def benchmark(
    estimate: Callable[[str], Distribution],
) -> pd.DataFrame:
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "benchmark" / "test.csv",
    )

    df["solution"] = df.apply(
        lambda row: estimate((
            f"{row['question']}\n"
            f"Answer in the following unit: {row['answer_unit']}"
        )),
        axis=1,
    )

    # norm_var_comp already returns the complement, so do not invert it.
    df["estimated_confidence"] = df["solution"].map(
        norm_var_comp
    )

    df["expected_confidence"] = (
        df["confidence_mean"].astype(float) / 100.0
    )

    df["squared_error"] = np.square(
        df["expected_confidence"]
        - df["estimated_confidence"]
    )

    mse = float(df["squared_error"].mean())

    _logger.info("Benchmark results:\n%s", df.to_string(index=False))
    _logger.info("MSE: %.6f", mse)

    return df


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

    benchmark(estimate)
