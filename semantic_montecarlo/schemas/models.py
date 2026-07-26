"""Schemas shared by the pipeline and benchmark."""

import math
from typing import Self

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class Distribution(BaseModel):
    """Conditional numeric outcomes and the probability of no answer."""

    data: list[tuple[float, float]]
    no_answer_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("data")
    @classmethod
    def validate_data(
        cls,
        data: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Validate numeric values and their conditional probabilities."""
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
            float(probability) for probability in probabilities
        )

        if data and not math.isclose(
            total_probability,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Probabilities must sum to 1.0; got {total_probability}.")

        return data

    @model_validator(mode="after")
    def validate_answerability(self) -> Self:
        """Ensure numeric outcomes agree with the no-answer probability."""
        if not self.data and self.no_answer_probability != 1.0:
            raise ValueError(
                "An empty numeric distribution requires "
                "no_answer_probability to be 1.0."
            )
        if self.data and self.no_answer_probability == 1.0:
            raise ValueError(
                "A numeric distribution cannot have no_answer_probability equal to 1.0."
            )
        return self
