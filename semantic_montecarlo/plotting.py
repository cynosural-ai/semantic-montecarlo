"""Static visualization of one semantic Monte Carlo result."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
from numpy.typing import NDArray

from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.schemas.run_result import RunResult

_BACKGROUND = "#FAFAF8"
_INK = "#17202A"
_MUTED = "#667085"
_DENSITY = "#526DFF"
_DENSITY_FILL = "#AEBBFF"
_EMPIRICAL = "#17202A"
_NO_ANSWER = "#D97757"
_GRID_POINTS = 512
_KDE_EXTENT = 3.0


def plot_result(result: RunResult) -> Figure:
    """Create the empirical, bootstrap, and no-answer visualization."""
    no_answer_probability = result.bootstrap_mean.no_answer_probability
    figure = Figure(
        figsize=(9.0, 5.5),
        layout="constrained",
        facecolor=_BACKGROUND,
    )

    if no_answer_probability > 0.0:
        grid = figure.add_gridspec(2, 1, height_ratios=(0.55, 4.0))
        answerability_axes = figure.add_subplot(grid[0])
        distribution_axes = figure.add_subplot(grid[1])
        _plot_no_answer(answerability_axes, no_answer_probability)
    else:
        distribution_axes = figure.subplots()

    figure.suptitle(
        textwrap.fill(result.question, width=55),
        x=0.06,
        horizontalalignment="left",
        color=_INK,
        fontsize=16,
        fontweight="bold",
    )
    _plot_numeric_distribution(
        distribution_axes,
        empirical=result.distribution,
        bootstrap_mean=result.bootstrap_mean,
        unit=result.unit,
    )
    return figure


def save_result_plot(result: RunResult, path: Path) -> None:
    """Render ``result`` to a headless PNG file."""
    figure = plot_result(result)
    figure.savefig(
        path,
        dpi=180,
        facecolor=figure.get_facecolor(),
    )


def _plot_no_answer(axes: Axes, probability: float) -> None:
    axes.hlines(0.0, 0.0, 1.0, color="#E7E7E2", linewidth=8, zorder=1)
    axes.hlines(0.0, 0.0, probability, color=_NO_ANSWER, linewidth=8, zorder=2)
    axes.text(
        0.0,
        0.38,
        f"No answer  {probability:.1%}",
        color=_NO_ANSWER,
        fontsize=10,
        fontweight="bold",
    )
    axes.set(xlim=(0.0, 1.0), ylim=(-0.35, 0.75))
    axes.set_axis_off()


def _plot_numeric_distribution(
    axes: Axes,
    *,
    empirical: Distribution,
    bootstrap_mean: Distribution,
    unit: str | None,
) -> None:
    bootstrap_values, bootstrap_weights = _positive_mass(bootstrap_mean)
    _style_distribution_axes(axes, unit)

    if not bootstrap_values.size:
        axes.text(
            0.5,
            0.5,
            "No numeric estimate available",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            color=_MUTED,
            fontsize=13,
        )
        axes.set(xticks=[], yticks=[])
        return

    estimate = float(np.sum(bootstrap_values * bootstrap_weights))

    if bootstrap_values.size == 1:
        _plot_point_mass(axes, bootstrap_values[0])
        bootstrap_handle: Patch | Line2D = Line2D(
            [],
            [],
            color=_DENSITY,
            linewidth=2.5,
            label="Bootstrap point mass",
        )
    else:
        grid, density = _weighted_kde(bootstrap_values, bootstrap_weights)
        axes.fill_between(
            grid,
            density,
            color=_DENSITY_FILL,
            alpha=0.65,
            linewidth=0,
        )
        axes.plot(grid, density, color=_DENSITY, linewidth=2.25)
        axes.set_ylim(bottom=0.0)
        _plot_interval(axes, bootstrap_values, bootstrap_weights)
        bootstrap_handle = Patch(
            facecolor=_DENSITY_FILL,
            edgecolor=_DENSITY,
            label="Bootstrap mean",
        )

    axes.axvline(
        estimate,
        color=_DENSITY,
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        alpha=0.9,
    )
    _plot_empirical_rug(axes, empirical)
    _add_legend(axes, bootstrap_handle, estimate)


def _style_distribution_axes(axes: Axes, unit: str | None) -> None:
    axes.set_facecolor(_BACKGROUND)
    axes.set_title(
        "Estimated numeric distribution",
        loc="left",
        color=_INK,
        fontsize=12,
        pad=16,
    )
    axes.set_xlabel(unit or "Value", color=_MUTED, labelpad=10)
    axes.set_yticks([])
    axes.xaxis.set_major_formatter(StrMethodFormatter("{x:,.6g}"))
    axes.tick_params(axis="x", colors=_MUTED, length=0, pad=8)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color("#D7D9DF")


def _plot_point_mass(axes: Axes, value: float) -> None:
    padding = max(abs(value) * 0.08, 1.0)
    axes.vlines(value, 0.0, 1.0, color=_DENSITY, linewidth=2.5)
    axes.scatter(
        [value],
        [1.0],
        marker="^",
        s=55,
        color=_DENSITY,
        zorder=3,
    )
    axes.set(xlim=(value - padding, value + padding), ylim=(0.0, 1.18))


def _plot_interval(
    axes: Axes,
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> None:
    lower, upper = _weighted_quantiles(values, weights, (0.05, 0.95))
    axes.plot(
        [lower, upper],
        [0.025, 0.025],
        transform=axes.get_xaxis_transform(),
        color=_DENSITY,
        linewidth=5,
        solid_capstyle="round",
        alpha=0.8,
    )


def _plot_empirical_rug(axes: Axes, distribution: Distribution) -> None:
    values, weights = _positive_mass(distribution)
    if not values.size:
        return

    relative_weights = weights / weights.max()
    for value, relative_weight in zip(values, relative_weights, strict=True):
        axes.vlines(
            value,
            0.0,
            0.1,
            transform=axes.get_xaxis_transform(),
            color=_EMPIRICAL,
            linewidth=1.0 + 3.0 * relative_weight,
            alpha=0.85,
            zorder=4,
        )


def _add_legend(
    axes: Axes,
    bootstrap_handle: Patch | Line2D,
    estimate: float,
) -> None:
    estimate_handle = Line2D(
        [],
        [],
        color=_DENSITY,
        linestyle=(0, (4, 3)),
        label=f"Estimate  {estimate:,.6g}",
    )
    empirical_handle = Line2D(
        [],
        [],
        color=_EMPIRICAL,
        marker="|",
        markersize=10,
        linestyle="none",
        label="Search estimates",
    )
    axes.legend(
        handles=(bootstrap_handle, estimate_handle, empirical_handle),
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=_MUTED,
    )


def _weighted_kde(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    bandwidth = _scott_bandwidth(values, weights)
    grid = np.linspace(
        values.min() - _KDE_EXTENT * bandwidth,
        values.max() + _KDE_EXTENT * bandwidth,
        _GRID_POINTS,
    )
    offsets = (grid[:, np.newaxis] - values) / bandwidth
    kernels = np.exp(-0.5 * np.square(offsets))
    density = kernels @ weights / (bandwidth * math.sqrt(2.0 * math.pi))
    return grid, density


def _scott_bandwidth(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    mean = np.sum(values * weights)
    variance = np.sum(weights * np.square(values - mean))
    effective_size = 1.0 / np.sum(np.square(weights))
    bandwidth = math.sqrt(float(variance)) * effective_size ** (-0.2)
    scale = max(float(np.ptp(values)), float(np.max(np.abs(values))), 1.0)
    return max(bandwidth, float(np.finfo(np.float64).eps * scale))


def _weighted_quantiles(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantiles: tuple[float, float],
) -> tuple[float, float]:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative_weights = np.cumsum(weights[order])
    lower, upper = (
        float(sorted_values[np.searchsorted(cumulative_weights, quantile)])
        for quantile in quantiles
    )
    return lower, upper


def _positive_mass(
    distribution: Distribution,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = np.asarray(distribution.data, dtype=np.float64)
    if not data.size:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    positive = data[:, 1] > 0.0
    values = data[positive, 0]
    weights = data[positive, 1]
    return values, weights / weights.sum()
