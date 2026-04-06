"""Tests for LattifAIEntrypoint — config.toml injection via __arguments__."""

from dataclasses import dataclass
from typing import Annotated, Optional
from unittest.mock import patch

import fiddle as fdl
import nemo_run as run
from nemo_run.cli.cli_parser import parse_cli_args
from nemo_run.config import Partial

from lattifai.cli.entrypoint import _apply_toml_defaults

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@dataclass
class _DiarConfig:
    _toml_section = "diarization"

    infer_speakers: bool = False
    verbose: bool = False
    model_name: str = "pyannote/default"


@dataclass
class _LLMConfig:
    _toml_section = ""  # No TOML section — should be skipped

    model_name: str = "some-model"


@dataclass
class _TransConfig:
    _toml_section = "translation"

    target_lang: str = "zh"
    bilingual: bool = True


def _my_func(
    diar: Annotated[Optional[_DiarConfig], run.Config[_DiarConfig]] = None,
    trans: Annotated[Optional[_TransConfig], run.Config[_TransConfig]] = None,
):
    pass


def _mock_toml(data: dict):
    """Patch resolve_toml_raw_value to read from *data* dict."""

    def _fake(section, key):
        return data.get(section, {}).get(key)

    return patch("lattifai.cli.entrypoint.resolve_toml_raw_value", side_effect=_fake)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyTomlDefaults:
    """_apply_toml_defaults injects config.toml for unset fields only."""

    def test_injects_for_unset_field(self):
        """Field not in __arguments__ gets TOML value."""
        config = parse_cli_args(_my_func, [], output_type=Partial)
        with _mock_toml({"diarization": {"infer_speakers": True}}):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        assert built.keywords["diar"].infer_speakers is True

    def test_preserves_explicit_cli_arg(self):
        """Field explicitly set via CLI is never overridden by TOML."""
        config = parse_cli_args(_my_func, ["diar.infer_speakers=false"], output_type=Partial)
        with _mock_toml({"diarization": {"infer_speakers": True}}):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        assert built.keywords["diar"].infer_speakers is False  # CLI wins

    def test_preserves_explicit_true(self):
        """CLI explicit True is preserved even when TOML says False."""
        config = parse_cli_args(_my_func, ["diar.infer_speakers=true"], output_type=Partial)
        with _mock_toml({"diarization": {"infer_speakers": False}}):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        assert built.keywords["diar"].infer_speakers is True

    def test_multiple_fields_injected(self):
        """Multiple fields in same section are all injected."""
        config = parse_cli_args(_my_func, [], output_type=Partial)
        with _mock_toml({"diarization": {"infer_speakers": True, "verbose": True}}):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        diar = built.keywords["diar"]
        assert diar.infer_speakers is True
        assert diar.verbose is True

    def test_no_toml_section_skipped(self):
        """Config without _toml_section is not touched."""

        def _func(cfg: Annotated[Optional[_LLMConfig], run.Config[_LLMConfig]] = None):
            pass

        config = parse_cli_args(_func, [], output_type=Partial)
        with _mock_toml({"": {"model_name": "overridden"}}):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        assert built.keywords["cfg"].model_name == "some-model"  # Unchanged

    def test_dict_values_skipped(self):
        """Nested TOML tables (dicts) are not injected as field values."""
        config = parse_cli_args(_my_func, [], output_type=Partial)
        with _mock_toml({"diarization": {"llm": {"model_name": "qwen"}}}):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        assert built.keywords["diar"].model_name == "pyannote/default"

    def test_missing_toml_keeps_default(self):
        """When TOML has no value, dataclass default remains."""
        config = parse_cli_args(_my_func, [], output_type=Partial)
        with _mock_toml({}):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        assert built.keywords["diar"].infer_speakers is False
        assert built.keywords["diar"].model_name == "pyannote/default"

    def test_multiple_configs_resolved(self):
        """Multiple Config params each resolve from their own TOML section."""
        config = parse_cli_args(_my_func, [], output_type=Partial)
        with _mock_toml(
            {
                "diarization": {"infer_speakers": True},
                "translation": {"target_lang": "ja"},
            }
        ):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        assert built.keywords["diar"].infer_speakers is True
        assert built.keywords["trans"].target_lang == "ja"

    def test_partial_cli_override_with_toml(self):
        """CLI overrides one field, TOML fills another — both work."""
        config = parse_cli_args(_my_func, ["diar.verbose=true"], output_type=Partial)
        with _mock_toml({"diarization": {"infer_speakers": True, "verbose": False}}):
            _apply_toml_defaults(config)

        built = fdl.build(config)
        diar = built.keywords["diar"]
        assert diar.verbose is True  # CLI wins
        assert diar.infer_speakers is True  # TOML fills
