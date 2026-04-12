"""Tests for honor_meta_chapters (hard-constraint chapter locking).

Two tiers of tests:

1. **Unit tests** — no LLM needed. Exercise prompt rendering, the
   `_should_lock_chapters` gate, and the `_apply_locked_chapters`
   post-process that enforces the meta chapter list over the model output.

2. **Integration tests** — run against a real LLM. Auto-skip unless a usable
   `[summarization]` config is reachable (GEMINI_API_KEY or OPENAI_API_KEY in
   env, or config.toml with api_key). This matches the rest of the
   llm-dependent tests in this repo.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from lattifai.config.summarization import SummarizationConfig
from lattifai.summarization.prompts import (
    build_reduce_user_prompt,
    build_summary_user_prompt,
)
from lattifai.summarization.schema import (
    SummaryChapter,
    SummaryInput,
    SummaryResult,
    summary_result_from_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

META_CHAPTERS = [
    {"title": "Intro", "start": 0.0},
    {"title": "Topic A", "start": 60.0},
    {"title": "Topic B", "start": 180.0},
    {"title": "Closing", "start": 300.0},
]


def make_input(chapters=None, text: str = "Hello world. This is a test.") -> SummaryInput:
    return SummaryInput(
        title="Test",
        text=text,
        chapters=chapters if chapters is not None else META_CHAPTERS,
        source_type="captions",
        source_lang="en",
    )


def llm_available() -> bool:
    """Best-effort check whether a real LLM call would work.

    We only need *some* pathway: env var, or a config.toml value. We do NOT
    call out to the network — tests that actually need the LLM will skip
    themselves when ``_can_construct_client()`` fails at runtime.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        from lattifai.config.toml_mixin import resolve_toml_raw_value

        for section in ("summarization.llm", "summarization"):
            for key in ("api_key",):
                if resolve_toml_raw_value(section, key):
                    return True
    except Exception:
        pass
    return False


def _can_construct_client() -> bool:
    """Deep check: try to actually build the LLM client for the default config.

    Uses the fallback model path in SummarizationConfig; if the client
    constructor raises (typically ``ValueError: No API key``), skip.
    """
    try:
        cfg = SummarizationConfig()
        cfg.llm.create_client()  # raises when credentials missing
        return True
    except Exception:
        return False


LLM_REASON = "Requires a usable [summarization] LLM config " "(GEMINI_API_KEY / OPENAI_API_KEY or config.toml api_key)"


# ---------------------------------------------------------------------------
# Unit tests — no LLM required
# ---------------------------------------------------------------------------


class TestConfigDefault:
    """honor_meta_chapters is on by default."""

    def test_default_is_true(self):
        assert SummarizationConfig().honor_meta_chapters is True

    def test_can_disable(self):
        c = SummarizationConfig(honor_meta_chapters=False)
        assert c.honor_meta_chapters is False


class TestPromptLocking:
    """Prompt rendering toggles the LOCKED CHAPTERS block."""

    def test_lock_renders_locked_block(self):
        p = build_summary_user_prompt(make_input(), lang="en", length="short", lock_chapters=True)
        assert "LOCKED CHAPTERS" in p
        assert "do NOT modify" in p
        # each chapter must appear
        for ch in META_CHAPTERS:
            assert ch["title"] in p

    def test_soft_renders_suggested_block(self):
        p = build_summary_user_prompt(make_input(), lang="en", length="short", lock_chapters=False)
        assert "LOCKED CHAPTERS" not in p
        assert "Chapters:" in p

    def test_no_chapters_renders_neither(self):
        p = build_summary_user_prompt(
            make_input(chapters=[]),
            lang="en",
            length="short",
            lock_chapters=True,  # requesting lock with no chapters is a no-op
        )
        assert "LOCKED CHAPTERS" not in p
        assert "Chapters:" not in p

    def test_reduce_prompt_locks_chapters(self):
        p = build_reduce_user_prompt(
            [{"chapters": [{"title": "X", "start": 0}]}],
            title="T",
            lang="en",
            length="short",
            source_type="captions",
            locked_chapters=META_CHAPTERS,
        )
        assert "LOCKED CHAPTERS" in p
        assert "must contain exactly these chapters" in p.lower() or "MUST" in p
        for ch in META_CHAPTERS:
            assert ch["title"] in p

    def test_reduce_prompt_without_lock(self):
        p = build_reduce_user_prompt(
            [{"chapters": []}],
            title="T",
            lang="en",
            length="short",
            source_type="captions",
        )
        assert "LOCKED CHAPTERS" not in p


class TestApplyLockedChapters:
    """Post-process: force-sync result.chapters to match input.chapters."""

    def _summarizer(self, **cfg_overrides):
        from lattifai.summarization.summarizer import ContentSummarizer

        cfg = SummarizationConfig(**cfg_overrides) if cfg_overrides else SummarizationConfig()
        # Build a summariser with a mocked client — we only use the
        # post-process method here, not the client.
        return ContentSummarizer(cfg, client=object())  # type: ignore[arg-type]

    def _model_result(self, chapters: list[SummaryChapter]) -> SummaryResult:
        return SummaryResult(
            title="T",
            overview="o",
            chapters=chapters,
        )

    def test_locks_drifted_count(self):
        """LLM collapsed 4 chapters → 2; post-process restores 4."""
        s = self._summarizer()
        si = make_input()
        drifted = self._model_result(
            [
                SummaryChapter(title="Merged A", start=0.0, summary="a summary", quotes=["q1"]),
                SummaryChapter(title="Merged B", start=180.0, summary="b summary", quotes=[]),
            ]
        )
        out = s._apply_locked_chapters(drifted, si)
        assert len(out.chapters) == 4
        assert [c.title for c in out.chapters] == [c["title"] for c in META_CHAPTERS]
        assert [c.start for c in out.chapters] == [c["start"] for c in META_CHAPTERS]
        # Drift was recorded
        assert out.metadata.get("chapters_locked") is True
        assert out.metadata.get("chapters_drift_before_lock") == 2

    def test_keeps_aligned_model_content_by_start(self):
        """When model's start is within 1s of a locked start, port over its summary+quotes."""
        s = self._summarizer()
        si = make_input()
        aligned = self._model_result(
            [
                SummaryChapter(title="x", start=0.0, summary="intro text", quotes=["iq"]),
                SummaryChapter(title="x", start=60.2, summary="A text", quotes=["aq"]),
                SummaryChapter(title="x", start=180.0, summary="B text", quotes=[]),
                SummaryChapter(title="x", start=300.0, summary="closing text", quotes=[]),
            ]
        )
        out = s._apply_locked_chapters(aligned, si)
        assert [c.summary for c in out.chapters] == [
            "intro text",
            "A text",
            "B text",
            "closing text",
        ]
        assert out.chapters[0].quotes == ["iq"]
        assert out.chapters[1].quotes == ["aq"]

    def test_positional_fallback_when_starts_misaligned(self):
        """No start match within 1s → fall back to positional mapping."""
        s = self._summarizer()
        si = make_input()
        misaligned = self._model_result(
            [
                SummaryChapter(title="x", start=5.0, summary="pos0", quotes=[]),
                SummaryChapter(title="x", start=999.0, summary="pos1", quotes=[]),
                SummaryChapter(title="x", start=888.0, summary="pos2", quotes=[]),
                SummaryChapter(title="x", start=777.0, summary="pos3", quotes=[]),
            ]
        )
        out = s._apply_locked_chapters(misaligned, si)
        # Positional mapping: chapter[i] gets model[i]
        # (Index 0 start=0 matches model[0] start=5.0 within 5s? No — tolerance is 1s,
        # so this is positional fallback.)
        assert [c.summary for c in out.chapters] == ["pos0", "pos1", "pos2", "pos3"]

    def test_empty_summary_when_model_too_short(self):
        """Model produced fewer chapters than locked + no start match → remaining get empty summary."""
        s = self._summarizer()
        si = make_input()
        short = self._model_result(
            [
                SummaryChapter(title="x", start=0.0, summary="only one", quotes=[]),
            ]
        )
        out = s._apply_locked_chapters(short, si)
        assert len(out.chapters) == 4
        assert out.chapters[0].summary == "only one"  # positional match works
        # Others fall back positionally — chapters beyond model length get empty
        assert out.chapters[1].summary == ""
        assert out.chapters[2].summary == ""
        assert out.chapters[3].summary == ""

    def test_respects_disable_flag(self):
        """With honor_meta_chapters=False, result passes through unchanged."""
        s = self._summarizer(honor_meta_chapters=False)
        si = make_input()
        drifted = self._model_result(
            [
                SummaryChapter(title="Only", start=0.0, summary="merged", quotes=[]),
            ]
        )
        out = s._apply_locked_chapters(drifted, si)
        assert len(out.chapters) == 1
        assert out.chapters[0].title == "Only"
        assert "chapters_locked" not in out.metadata

    def test_no_meta_chapters_passes_through(self):
        """No meta chapters → nothing to lock, result unchanged."""
        s = self._summarizer()
        si = make_input(chapters=[])
        model = self._model_result(
            [
                SummaryChapter(title="LLM chose this", start=0.0, summary="x", quotes=[]),
            ]
        )
        out = s._apply_locked_chapters(model, si)
        assert len(out.chapters) == 1
        assert out.chapters[0].title == "LLM chose this"

    def test_end_time_filled_between_locked_chapters(self):
        """Missing ``end`` on locked chapters is auto-filled from next start."""
        s = self._summarizer()
        si = make_input()
        out = s._apply_locked_chapters(self._model_result([]), si)
        assert out.chapters[0].end == 60.0
        assert out.chapters[1].end == 180.0
        assert out.chapters[2].end == 300.0
        # Last chapter has no next — end stays 0 (renderer handles)
        assert out.chapters[3].end == 0.0


class TestSummarizeFlowLocksChapters:
    """End-to-end test of the summarize() pipeline with a MOCKED LLM.

    The mock returns a drifted chapter list; we verify the final
    SummaryResult is locked back to the meta chapters.
    """

    def test_single_pass_locks_chapters_with_mocked_llm(self):
        from lattifai.summarization.summarizer import ContentSummarizer

        cfg = SummarizationConfig()  # honor_meta_chapters=True
        summariser = ContentSummarizer(cfg, client=object())  # type: ignore[arg-type]

        # Mock the LLM JSON response with 2 drifted chapters
        mock_llm = AsyncMock(
            return_value={
                "title": "T",
                "overview": "hi",
                "chapters": [
                    {"title": "Drift A", "start": 0.0, "summary": "A", "quotes": []},
                    {"title": "Drift B", "start": 100.0, "summary": "B", "quotes": []},
                ],
                "entities": [],
                "tags": [],
                "seo_title": "",
                "seo_description": "",
            }
        )
        with patch.object(summariser, "_call_llm", mock_llm):
            result = asyncio.run(summariser.summarize(make_input()))

        # Drift detected → locked to 4
        assert len(result.chapters) == 4
        assert [c.title for c in result.chapters] == [c["title"] for c in META_CHAPTERS]
        assert [c.start for c in result.chapters] == [c["start"] for c in META_CHAPTERS]
        assert result.metadata.get("chapters_locked") is True


# ---------------------------------------------------------------------------
# Integration tests — require a usable LLM
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not llm_available() or not _can_construct_client(), reason=LLM_REASON)
class TestHonorMetaChaptersIntegration:
    """End-to-end: real LLM + meta chapters → locked output.

    These tests skip automatically when no LLM credentials are reachable.
    Run with e.g. ``GEMINI_API_KEY=... pytest tests/summarization``.
    """

    SAMPLE_TEXT = (
        "[00:00] Welcome. Today we'll talk about three things.\n"
        "[00:30] First, let's discuss the weather and its patterns.\n"
        "[01:05] Next, the role of coffee in modern productivity.\n"
        "[02:00] Finally, a quick recap of what we covered.\n"
        "[03:00] Thanks for tuning in.\n"
    )

    @pytest.fixture
    def summariser(self):
        from lattifai.summarization.summarizer import ContentSummarizer

        cfg = SummarizationConfig()  # defaults: honor_meta_chapters=True
        client = cfg.llm.create_client()
        return ContentSummarizer(cfg, client)

    def test_locked_chapters_preserved_end_to_end(self, summariser):
        meta = [
            {"title": "Welcome", "start": 0.0},
            {"title": "Weather", "start": 30.0},
            {"title": "Coffee", "start": 65.0},
            {"title": "Recap", "start": 120.0},
            {"title": "Outro", "start": 180.0},
        ]
        si = SummaryInput(
            title="Morning Show",
            text=self.SAMPLE_TEXT,
            chapters=meta,
            source_type="captions",
            source_lang="en",
        )
        result = asyncio.run(summariser.summarize(si))
        # Locked: count + titles + starts match meta exactly
        assert len(result.chapters) == len(meta)
        assert [c.title for c in result.chapters] == [m["title"] for m in meta]
        assert [c.start for c in result.chapters] == [m["start"] for m in meta]

    def test_no_meta_chapters_llm_generates_own(self, summariser):
        si = SummaryInput(
            title="Morning Show",
            text=self.SAMPLE_TEXT,
            chapters=[],
            source_type="captions",
            source_lang="en",
        )
        result = asyncio.run(summariser.summarize(si))
        # No lock → LLM picks its own count. Just assert it produced something.
        assert len(result.chapters) >= 1
        assert "chapters_locked" not in result.metadata

    def test_disable_flag_allows_drift(self):
        """With honor_meta_chapters=False, LLM may output different chapters."""
        from lattifai.summarization.summarizer import ContentSummarizer

        cfg = SummarizationConfig(honor_meta_chapters=False)
        client = cfg.llm.create_client()
        s = ContentSummarizer(cfg, client)
        meta = [{"title": f"Ch{i}", "start": float(i * 20)} for i in range(8)]
        si = SummaryInput(
            title="Morning Show",
            text=self.SAMPLE_TEXT,
            chapters=meta,
            source_type="captions",
            source_lang="en",
        )
        result = asyncio.run(s.summarize(si))
        # No lock metadata present — drift is allowed
        assert "chapters_locked" not in result.metadata


# ---------------------------------------------------------------------------
# Schema round-trip sanity
# ---------------------------------------------------------------------------


class TestSchemaChaptersRoundtrip:
    def test_from_dict_preserves_chapters(self):
        data = {
            "title": "T",
            "chapters": [
                {"title": "A", "start": 0.0, "end": 60.0, "summary": "a"},
                {"title": "B", "start": 60.0, "end": 120.0, "summary": "b"},
            ],
        }
        r = summary_result_from_dict(data, fallback_title="fb")
        assert [c.title for c in r.chapters] == ["A", "B"]
        assert r.chapters[0].start == 0.0
        assert r.chapters[1].end == 120.0
