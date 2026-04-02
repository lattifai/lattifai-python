"""Tests for resolve_toml_raw_value — raw TOML value reading with nested section support."""

from lattifai.config.toml_mixin import resolve_toml_raw_value


class TestResolveTomlRawValue:
    """resolve_toml_raw_value reads raw values from ~/.lattifai/config.toml.

    These tests hit the real config file (if present) or get None.
    The entrypoint injection tests (test_entrypoint.py) cover the mocked flow.
    """

    def test_returns_none_for_missing_section(self):
        """Non-existent section returns None without error."""
        result = resolve_toml_raw_value("__nonexistent_test_section__", "key")
        assert result is None

    def test_returns_none_for_missing_key(self):
        """Existing section with missing key returns None."""
        # "auth" section exists in most config.toml setups
        result = resolve_toml_raw_value("auth", "__nonexistent_key__")
        assert result is None

    def test_dotted_section_returns_none_for_missing(self):
        """Dotted (nested) section returns None when not present."""
        result = resolve_toml_raw_value("__nonexistent__.llm", "model_name")
        assert result is None
