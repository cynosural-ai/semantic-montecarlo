"""Load YAML prompt files, cache the parse, return a PromptTemplate."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Prompts live next to this module and travel with the installed package
# (declared as package-data in pyproject.toml).
_PROMPTS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptTemplate:
    """
    A loaded prompt file.

    Attributes:
        name: Logical name from the YAML ``name`` key.
        fields: Named text bodies (e.g. ``system``, ``user``).
    """

    name: str
    fields: dict[str, str]

    def render(self, field: str, /, **kwargs: object) -> str:
        """
        Render ``field`` with ``str.format`` substitution.

        Args:
            field: Which named field to render (e.g. ``"user"``).
            **kwargs: Substitution values for ``{placeholder}`` tokens.

        Returns:
            The rendered text.
        """
        return self.fields[field].format(**kwargs)


@lru_cache(maxsize=None)
def load(name: str, *, prompts_dir: Path = _PROMPTS_DIR) -> PromptTemplate:
    """
    Load ``{prompts_dir}/{name}.yaml`` once, cache, return a template.

    Args:
        name: Prompt name (file stem without ``.yaml``).
        prompts_dir: Directory to read from; defaults to the bundled prompts.

    Returns:
        A :class:`PromptTemplate`.
    """
    path = prompts_dir / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fields = {k: v for k, v in data.items() if k != "name"}
    return PromptTemplate(name=data["name"], fields=fields)
