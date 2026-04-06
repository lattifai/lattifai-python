"""Glossary loading and merging for LattifAI translation."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_glossary(glossary_file: Optional[str] = None) -> dict[str, str]:
    """Load glossary from a file.

    Supports YAML and Markdown table formats.

    YAML format:
        source_term: target_translation
        another_term: another_translation

    Markdown table format:
        | Source | Target |
        |--------|--------|
        | term1  | trans1 |
        | term2  | trans2 |

    Args:
        glossary_file: Path to glossary file.

    Returns:
        Dictionary mapping source terms to target translations.
    """
    if not glossary_file:
        return {}

    path = Path(glossary_file)
    if not path.exists():
        logger.warning("Glossary file not found: %s", glossary_file)
        return {}

    content = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        return _load_yaml_glossary(content)
    elif path.suffix == ".md":
        return _load_markdown_glossary(content)
    else:
        logger.warning("Unsupported glossary format: %s (use .yaml or .md)", path.suffix)
        return {}


def _load_yaml_glossary(content: str) -> dict[str, str]:
    """Parse YAML glossary."""
    try:
        import yaml

        data = yaml.safe_load(content)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}
    except ImportError:
        # Fallback: simple key: value parsing
        glossary = {}
        for line in content.splitlines():
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                key = key.strip().strip('"').strip("'")
                value = value.strip().strip('"').strip("'")
                if key and value:
                    glossary[key] = value
        return glossary


def _load_markdown_glossary(content: str) -> dict[str, str]:
    """Parse Markdown table glossary."""
    glossary = {}
    in_table = False

    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            in_table = False
            continue

        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 2:
            continue

        # Skip separator lines
        if all(set(c) <= set("- :") for c in cells):
            in_table = True
            continue

        # Skip header if we haven't seen separator yet
        if not in_table:
            continue

        source, target = cells[0], cells[1]
        if source and target:
            glossary[source] = target

    return glossary


def merge_glossaries(
    user_glossary: dict[str, str],
    analysis_terms: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Merge glossaries with priority: user > analysis.

    Args:
        user_glossary: User-provided glossary (highest priority).
        analysis_terms: Terms extracted from content analysis.

    Returns:
        Merged glossary dictionary.
    """
    merged = {}
    if analysis_terms:
        merged.update(analysis_terms)
    merged.update(user_glossary)  # User glossary overrides
    return merged
