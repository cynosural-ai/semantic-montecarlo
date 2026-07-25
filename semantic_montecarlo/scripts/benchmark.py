from collections.abc import Callable

import numpy as np
import pandas as pd

from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.utils import src_path


def norm_var_comp(
    distribution: Distribution,
) -> float:
    """
    Compute the uniform-normalized variance complement.

    Returns:
        1.0 when all probability is concentrated at one x-value.
        0.0 when the variance equals or exceeds the variance of a
        uniform distribution over the supplied x-values.

    Notes:
        - Distances between x-values affect the result.
        - Duplicate x-values are combined.
        - This function does not mutate or normalize the distribution.
    """
    data = np.asarray(
        distribution.data,
        dtype=np.float64,
    )

    x = data[:, 0]
    probabilities = data[:, 1]

    # Combine probability masses assigned to duplicate x-values.
    unique_x, inverse_indices = np.unique(
        x,
        return_inverse=True,
    )

    combined_probabilities = np.zeros(
        unique_x.size,
        dtype=np.float64,
    )

    np.add.at(
        combined_probabilities,
        inverse_indices,
        probabilities,
    )

    if unique_x.size == 1:
        return 1.0

    weighted_mean = np.sum(
        unique_x * combined_probabilities
    )

    weighted_variance = np.sum(
        combined_probabilities
        * np.square(unique_x - weighted_mean)
    )

    uniform_variance = np.var(unique_x)

    if np.isclose(uniform_variance, 0.0):
        return 1.0

    variance_ratio = weighted_variance / uniform_variance
    complement = 1.0 - variance_ratio

    return float(np.clip(complement, 0.0, 1.0))


def build_question(row: pd.Series) -> str:
    return (
        f"{row['question']}\n"
        f"Answer in the following unit: {row['answer_unit']}"
    )


def benchmark(
    estimate: Callable[[str], Distribution],
) -> pd.DataFrame:
    df = pd.read_csv(
        src_path / "data" / "benchmark" / "benchmark.csv",
    )

    df["solution"] = df.apply(
        lambda row: estimate(build_question(row)),
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

    print(df)
    print(f"MSE: {mse:.6f}")

    return df


if __name__ == "__main__":

    def dummy_estimate(question: str) -> Distribution:
        """
        Return a symmetric distribution centered on the question length.

        The probabilities sum to exactly 1:
            0.25 + 0.50 + 0.25 = 1.00
        """
        length = float(len(question))

        return Distribution(
            data=[
                (length - 10.0, 0.25),
                (length, 0.50),
                (length + 10.0, 0.25),
            ]
        )

    benchmark(dummy_estimate)