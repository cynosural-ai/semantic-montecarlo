"""Tests for the CLI's run-artifact serialization.

Covers ``_save_run`` and ``_run_dir`` — the pure-I/O parts of the CLI that
persist a :class:`RunResult`. No pipeline or network involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_montecarlo.cli import _run_dir, _save_run
from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.schemas.run_result import RunResult
from semantic_montecarlo.schemas.search import NumericAnswer
from semantic_montecarlo.schemas.usage import Usage


def _result(answers: list[NumericAnswer]) -> RunResult:
    return RunResult(
        question="How many people live in France?",
        unit="people",
        paraphrases=["How many people live in France?"],
        answers=answers,
        distribution=Distribution(data=[(67.0, 1.0)]),
        elapsed_seconds=2.5,
        model="openrouter/free",
        paraphrase_usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        search_usage=Usage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
    )


def _answer(value: float) -> NumericAnswer:
    return NumericAnswer(
        reasoning="r", value=value, confidence=0.8, sources=["https://example.com"]
    )


def test_run_dir_creates_timestamped_subdir(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    # Subdir name is YYYYMMDD_HHMMSS — 8 digits, underscore, 6 digits.
    assert len(run_dir.name) == 15
    assert run_dir.name[8] == "_"


def test_save_run_writes_result_and_searches(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    answers = [_answer(67.0), _answer(68.0)]
    _save_run(
        _result(answers),
        run_dir,
        parameters={"n_paraphrases": 5, "n_resamples": 10000, "seed": None},
    )

    result_path = run_dir / "result.json"
    searches_path = run_dir / "searches.json"
    assert result_path.is_file()
    assert searches_path.is_file()

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["model"] == "openrouter/free"
    assert result["elapsed_seconds"] == 2.5
    assert result["question"] == "How many people live in France?"
    assert result["unit"] == "people"
    assert result["parameters"] == {
        "n_paraphrases": 5,
        "n_resamples": 10000,
        "seed": None,
    }
    assert result["distribution"] == [{"value": 67.0, "probability": 1.0}]


def test_save_run_writes_per_stage_usage(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _save_run(_result([_answer(67.0)]), run_dir, parameters={})

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    usage = result["usage"]
    assert set(usage) == {"paraphrase", "search", "total"}
    assert usage["paraphrase"]["total_tokens"] == 15
    assert usage["search"]["total_tokens"] == 60
    # total is the component-wise sum of the two stages.
    assert usage["total"]["total_tokens"] == 75
    assert usage["total"]["prompt_tokens"] == 50


def test_save_run_searches_match_answers(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    answers = [_answer(67.0), _answer(68.0), _answer(70.0)]
    _save_run(_result(answers), run_dir, parameters={})

    searches = json.loads((run_dir / "searches.json").read_text(encoding="utf-8"))
    assert len(searches) == 3
    assert searches[0]["value"] == 67.0
    assert searches[0]["sources"] == ["https://example.com"]


def test_save_run_overwrites_existing_dir_contents(tmp_path: Path) -> None:
    # _run_dir uses mkdir(exist_ok=True); _save_run writes fresh files.
    run_dir = _run_dir(tmp_path)
    _save_run(_result([_answer(1.0)]), run_dir, parameters={})
    _save_run(_result([_answer(2.0)]), run_dir, parameters={})

    searches = json.loads((run_dir / "searches.json").read_text(encoding="utf-8"))
    assert len(searches) == 1
    assert searches[0]["value"] == 2.0


def test_save_run_creates_nested_output_dir(tmp_path: Path) -> None:
    # _run_dir mkdir(parents=True); deeply nested output_dir must work.
    nested = tmp_path / "a" / "b" / "c"
    run_dir = _run_dir(nested)
    _save_run(_result([_answer(1.0)]), run_dir, parameters={})
    assert (run_dir / "result.json").is_file()
