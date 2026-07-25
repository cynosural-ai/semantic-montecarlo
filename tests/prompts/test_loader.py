"""
Tests for the prompt loader.

Uses pytest's ``tmp_path`` so parse/render paths get a real file, no
monkeypatching of resource readers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_montecarlo.prompts import PromptTemplate, load
from semantic_montecarlo.prompts import loader as loader_mod


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """The loader is module-level cached; clear it before each test."""
    loader_mod.load.cache_clear()


def test_load_bundled_paraphraser() -> None:
    tpl = load("paraphraser")
    assert isinstance(tpl, PromptTemplate)
    assert tpl.name == "paraphraser"
    assert set(tpl.fields) == {"system", "user"}


def test_load_is_cached(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("name: p\nuser: |\n  hi\n", encoding="utf-8")
    first = load("p", prompts_dir=tmp_path)
    second = load("p", prompts_dir=tmp_path)
    assert first is second  # same object identity proves lru_cache hit


def test_render_substitutes_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text('name: p\nuser: "Hello {who}!"\n', encoding="utf-8")
    tpl = load("p", prompts_dir=tmp_path)
    assert tpl.render("user", who="world") == "Hello world!"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load("nope", prompts_dir=tmp_path)


def test_missing_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("name: p\nuser: |\n  hi\n", encoding="utf-8")
    tpl = load("p", prompts_dir=tmp_path)
    with pytest.raises(KeyError):
        tpl.render("assistant")
