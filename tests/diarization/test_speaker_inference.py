"""Tests for speaker name inference: extract_candidate_names, SpeakerNameInferrer, _build_speaker_context."""

from typing import Any, Optional

import pytest

from lattifai.diarization.speaker import SpeakerNameInferrer, extract_candidate_names
from lattifai.llm.base import BaseLLMClient

# ---------------------------------------------------------------------------
# Fake LLM client for deterministic testing
# ---------------------------------------------------------------------------


class FakeLLMClient(BaseLLMClient):
    """Deterministic LLM client that returns pre-configured responses."""

    def __init__(self, responses: list):
        super().__init__(api_key="test-key", model="fake-model")
        self._responses = list(responses)
        self.prompts: list = []

    @property
    def provider_name(self) -> str:
        return "fake"

    async def generate(self, prompt: str, *, model=None, system=None, temperature=None) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("No fake responses left")
        return str(self._responses.pop(0))

    async def generate_json(self, prompt: str, *, model=None, system=None, temperature=None) -> Any:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("No fake responses left")
        return self._responses.pop(0)


class FailingLLMClient(BaseLLMClient):
    """LLM client that always raises an exception."""

    def __init__(self):
        super().__init__(api_key="test-key", model="fail-model")

    @property
    def provider_name(self) -> str:
        return "fail"

    async def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("LLM unavailable")

    async def generate_json(self, prompt: str, **kwargs) -> Any:
        raise RuntimeError("LLM unavailable")


# ===========================================================================
# A. extract_candidate_names tests
# ===========================================================================


class TestExtractCandidateNames:
    def test_empty_context_none(self):
        assert extract_candidate_names(None) == {}

    def test_empty_context_string(self):
        assert extract_candidate_names("") == {}

    def test_channel_host(self):
        ctx = "Channel/Host: Zhang San\nTitle: Some Episode"
        result = extract_candidate_names(ctx)
        assert result["host"] == ["Zhang San"]

    def test_channel_rejected_as_show_name(self):
        ctx = "Channel/Host: Tech Podcast\n"
        result = extract_candidate_names(ctx)
        assert "host" not in result

    def test_channel_rejected_mlst(self):
        ctx = "Channel/Host: MLST\n"
        result = extract_candidate_names(ctx)
        assert "host" not in result

    def test_channel_rejected_talk_show(self):
        """Show names containing 'talk' or 'street' should be rejected."""
        ctx = "Channel/Host: Machine Learning Street Talk\n"
        result = extract_candidate_names(ctx)
        assert "host" not in result

    def test_title_guest_dash(self):
        ctx = "Title: Li Si — Deep Learning Advances\n"
        result = extract_candidate_names(ctx)
        assert result["guest"] == ["Li Si"]

    def test_title_guest_with_episode_number(self):
        ctx = "Title: #42 Wang Wu — New Research\n"
        result = extract_candidate_names(ctx)
        assert result["guest"] == ["Wang Wu"]

    def test_title_no_dash_no_guest(self):
        ctx = "Title: A Great Discussion About AI\n"
        result = extract_candidate_names(ctx)
        assert "guest" not in result

    def test_chinese_host_same_line(self):
        """Issue 5: 【主播】张三 on the same line should be captured."""
        ctx = "【主播】张三\n【嘉宾】李四\n"
        result = extract_candidate_names(ctx)
        assert "张三" in result.get("host", [])
        assert "李四" in result.get("guest", [])

    def test_chinese_host_next_line(self):
        ctx = "【主播】\n张三\n\n其他内容"
        result = extract_candidate_names(ctx)
        assert "张三" in result.get("host", [])

    def test_chinese_guest_multiline(self):
        ctx = "【嘉宾】\n王五，AI研究员\n赵六，数据科学家\n\n"
        result = extract_candidate_names(ctx)
        guests = result.get("guest", [])
        assert "王五" in guests
        assert "赵六" in guests

    def test_chinese_host_bracket_variant(self):
        ctx = "[主持]\n陈七\n\n"
        result = extract_candidate_names(ctx)
        assert "陈七" in result.get("host", [])

    def test_joined_by_pattern(self):
        ctx = "Alice Smith and Bob Jones are joined by Charlie Lee in this episode."
        result = extract_candidate_names(ctx)
        hosts = result.get("host", [])
        guests = result.get("guest", [])
        assert "Alice Smith" in hosts
        assert "Bob Jones" in hosts
        assert "Charlie Lee" in guests

    def test_title_strip_prefix(self):
        """Issue 8: 教授陈伟 should strip title prefix and keep 陈伟."""
        ctx = "【嘉宾】\n教授陈伟\n\n"
        result = extract_candidate_names(ctx)
        assert "陈伟" in result.get("guest", [])

    def test_title_strip_english_prefix(self):
        ctx = "【嘉宾】\nDr. Smith\n\n"
        result = extract_candidate_names(ctx)
        assert "Smith" in result.get("guest", [])

    def test_title_strip_suffix(self):
        ctx = "【嘉宾】\n陈伟教授\n\n"
        result = extract_candidate_names(ctx)
        assert "陈伟" in result.get("guest", [])

    def test_reject_pure_role(self):
        """Pure role names like CEO should be rejected."""
        ctx = "【嘉宾】\nCEO\n\n"
        result = extract_candidate_names(ctx)
        assert "guest" not in result

    def test_reject_composite_role(self):
        ctx = "【嘉宾】\n创始管理合伙人\n\n"
        result = extract_candidate_names(ctx)
        assert "guest" not in result

    def test_parenthetical_suffix_stripped(self):
        ctx = "【嘉宾】\n陈伟（教授）\n\n"
        result = extract_candidate_names(ctx)
        assert "陈伟" in result.get("guest", [])

    def test_name_too_short_rejected(self):
        ctx = "【嘉宾】\nA\n\n"
        result = extract_candidate_names(ctx)
        assert "guest" not in result

    def test_drake_not_corrupted_by_dr_strip(self):
        """English name 'Drake' should not be corrupted by Dr prefix stripping."""
        ctx = "Channel/Host: Drake Johnson\n"
        result = extract_candidate_names(ctx)
        # "Drake Johnson" should remain intact (Dr strip requires period or space after)
        assert "Drake Johnson" in result.get("host", [])


# ===========================================================================
# B. SpeakerNameInferrer tests
# ===========================================================================


class TestSpeakerNameInferrer:
    @pytest.fixture
    def speaker_texts(self):
        return {
            "SPEAKER_00": ["Hello everyone, welcome to the show", "So what's your take on AI?"],
            "SPEAKER_01": [
                "Thanks for having me. I've been working on language models for years.",
                "The key insight is that scaling laws are predictable.",
            ],
        }

    def test_valid_mapping(self, speaker_texts):
        llm = FakeLLMClient([{"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        result = inferrer(speaker_texts)
        assert result == {"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}

    def test_filters_unknown_speaker_keys(self, speaker_texts):
        llm = FakeLLMClient([{"SPEAKER_00": "Host", "SPEAKER_99": "Unknown"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        result = inferrer(speaker_texts)
        assert "SPEAKER_99" not in result
        assert result == {"SPEAKER_00": "Host"}

    def test_filters_empty_values(self, speaker_texts):
        llm = FakeLLMClient([{"SPEAKER_00": "Host", "SPEAKER_01": "  "}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        result = inferrer(speaker_texts)
        assert "SPEAKER_01" not in result

    def test_llm_failure_returns_empty(self, speaker_texts):
        inferrer = SpeakerNameInferrer(llm_client=FailingLLMClient())
        result = inferrer(speaker_texts)
        assert result == {}

    def test_non_dict_returns_empty(self, speaker_texts):
        llm = FakeLLMClient([["not", "a", "dict"]])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        result = inferrer(speaker_texts)
        assert result == {}

    def test_llm_output_trusted_without_candidate_filtering(self, speaker_texts):
        """LLM output is trusted — no candidate-based filtering applied."""
        context = "Channel/Host: Alice\nTitle: Bob Smith — Topic\n"
        llm = FakeLLMClient([{"SPEAKER_00": "Alice", "SPEAKER_01": "Charlie"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        result = inferrer(speaker_texts, context=context)
        # "Charlie" is not a candidate but should still be kept
        assert result == {"SPEAKER_00": "Alice", "SPEAKER_01": "Charlie"}

    def test_no_candidates_allows_any_name(self, speaker_texts):
        """Without candidates, any valid name should pass through."""
        llm = FakeLLMClient([{"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        result = inferrer(speaker_texts, context="A podcast about technology")
        assert result == {"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}

    def test_prompt_with_candidates(self, speaker_texts):
        """When candidates exist, prompt should mention 'candidate list'."""
        context = "Channel/Host: Alice\nTitle: Bob — Topic\n"
        llm = FakeLLMClient([{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        inferrer(speaker_texts, context=context)
        assert "candidate list" in llm.prompts[0]

    def test_prompt_without_candidates(self, speaker_texts):
        """Issue 10: Without candidates, prompt should not mention 'candidate list'."""
        llm = FakeLLMClient([{"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        inferrer(speaker_texts, context="A random topic")
        assert "candidate list" not in llm.prompts[0]
        assert "self-introductions" in llm.prompts[0]

    def test_truncation_200(self, speaker_texts):
        """Issue 7: Text samples should be truncated at 200 chars."""
        long_text = "A" * 500
        texts = {"SPEAKER_00": [long_text]}
        llm = FakeLLMClient([{"SPEAKER_00": "Host"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        inferrer(texts)
        prompt = llm.prompts[0]
        # The prompt should contain the truncated version, not the full 500 chars
        assert "A" * 200 + "..." in prompt
        assert "A" * 201 not in prompt


# ===========================================================================
# C. _build_speaker_context tests
# ===========================================================================


class TestBuildSpeakerContext:
    def test_full_metadata(self):
        from lattifai.client import _build_speaker_context

        meta = {
            "title": "Episode 1 — Great Talk",
            "uploader": "Tech Channel",
            "description": "A discussion about AI.\n\nTimestamps:\n0:00 - Intro",
        }
        ctx = _build_speaker_context(meta)
        assert "Title: Episode 1 — Great Talk" in ctx
        assert "Channel/Host: Tech Channel" in ctx
        assert "A discussion about AI." in ctx

    def test_empty_metadata(self):
        from lattifai.client import _build_speaker_context

        assert _build_speaker_context({}) is None

    def test_no_description(self):
        from lattifai.client import _build_speaker_context

        meta = {"title": "My Episode", "uploader": "Host Name"}
        ctx = _build_speaker_context(meta)
        assert "Title: My Episode" in ctx
        assert "Channel/Host: Host Name" in ctx
        assert "Description" not in ctx

    def test_channel_fallback(self):
        from lattifai.client import _build_speaker_context

        meta = {"title": "Ep", "channel": "Fallback Channel"}
        context = _build_speaker_context(meta)
        assert "Channel/Host: Fallback Channel" in context

    def test_structured_speakers_field(self):
        """When structured speakers are present, use them directly."""
        from lattifai.client import _build_speaker_context

        meta = {
            "title": "Jensen Huang Interview",
            "speakers": [
                {"name": "Sarah Guo", "role": "host"},
                {"name": "Elad Gil", "role": "host"},
                {"name": "Jensen Huang", "role": "guest"},
            ],
        }
        context = _build_speaker_context(meta)
        assert "Channel/Host: Sarah Guo, Elad Gil" in context
        assert "Guests: Jensen Huang" in context
