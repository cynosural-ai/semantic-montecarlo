import math

import numpy as np
from pydantic import BaseModel, field_validator


class Distribution(BaseModel):
    data: list[tuple[float, float]]

    @field_validator("data")
    @classmethod
    def validate_data(
        cls,
        data: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        x_values = np.asarray(
            [x for x, _ in data],
            dtype=np.float64,
        )
        probabilities = np.asarray(
            [probability for _, probability in data],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(x_values)):
            raise ValueError("All x-values must be finite.")

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("All probabilities must be finite.")

        if np.any(probabilities < 0.0):
            raise ValueError("Probabilities cannot be negative.")

        total_probability = math.fsum(
            float(probability)
            for probability in probabilities
        )

        if not math.isclose(
            total_probability,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Probabilities must sum to 1.0; "
                f"got {total_probability}."
            )

        return data
