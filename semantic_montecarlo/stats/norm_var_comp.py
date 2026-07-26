"""Confidence metric for an estimated numeric distribution."""

import numpy as np

from semantic_montecarlo.schemas.models import Distribution


def norm_var_comp(
    distribution: Distribution,
) -> float:
    """
    Compute the uniform-normalized variance complement.

    Returns:
        Numeric concentration discounted by the probability of no answer.
        0.0 when the variance equals or exceeds the variance of a
        uniform distribution, or when no numeric answer is available.

    Notes:
        - Distances between x-values affect the result.
        - Duplicate x-values are combined.
        - This function does not mutate or normalize the distribution.
    """
    if not distribution.data:
        return 0.0

    answer_probability = 1.0 - distribution.no_answer_probability
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
        return answer_probability

    weighted_mean = np.sum(unique_x * combined_probabilities)

    weighted_variance = np.sum(
        combined_probabilities * np.square(unique_x - weighted_mean)
    )

    uniform_variance = np.var(unique_x)

    if np.isclose(uniform_variance, 0.0):
        return answer_probability

    variance_ratio = weighted_variance / uniform_variance
    complement = 1.0 - variance_ratio

    return answer_probability * float(np.clip(complement, 0.0, 1.0))
