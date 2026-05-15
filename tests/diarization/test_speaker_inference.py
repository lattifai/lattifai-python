"""Tests for speaker name inference: extract_candidate_names, SpeakerNameInferrer, _build_speaker_context."""

import json
import textwrap
from pathlib import Path
from typing import Any

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

    def test_channel_rejected_descriptive_name(self):
        """Descriptive channel names that look like phrases, not person names,
        should be rejected even without a show-keyword suffix.

        Regression for TheDiaryOfACEO case where "The Diary Of A CEO" was
        incorrectly assigned as the host name (the actual host is Steven Bartlett).
        """
        ctx = "Channel/Host: The Diary Of A CEO\n"
        result = extract_candidate_names(ctx)
        assert "host" not in result

    def test_channel_single_word_rejected(self):
        """Single-word channel handles (not a real person name) should be rejected."""
        ctx = "Channel/Host: Apolas\n"
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
            "SPEAKER_00": [
                "Hello everyone, welcome to the show",
                "So what's your take on AI?",
            ],
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
        # Use multi-word names so both Channel/Host and Title pass
        # _looks_like_person_name (single-word handles are rejected).
        context = "Channel/Host: Alice Smith\nTitle: Bob Jones — Topic\n"
        llm = FakeLLMClient([{"SPEAKER_00": "Alice Smith", "SPEAKER_01": "Bob Jones"}])
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


# ===========================================================================
# D. Structured Speakers block — affiliation / aliases / bio extraction
# ===========================================================================


class TestStructuredSpeakersBlock:
    def test_extract_affiliation(self):
        ctx = textwrap.dedent(
            """\
            Title: Deep Dive into RNA Design
            Channel/Host: Alice Chen
            Speakers:
            - Alice Chen (host) @ Anthropic
            - Bob Smith (guest) @ Atomic AI
            """
        )
        result = extract_candidate_names(ctx)
        assert "Alice Chen" in result.get("host", [])
        assert "Bob Smith" in result.get("guest", [])
        assert result["affiliations"]["Alice Chen"] == "Anthropic"
        assert result["affiliations"]["Bob Smith"] == "Atomic AI"

    def test_extract_affiliation_with_parens(self):
        """Affiliation itself may contain parens — must not be confused with name parens."""
        ctx = textwrap.dedent(
            """\
            Speakers:
            - Alice Chen (host) @ Anthropic (research engineer)
            """
        )
        result = extract_candidate_names(ctx)
        assert result["affiliations"]["Alice Chen"] == "Anthropic (research engineer)"

    def test_extract_aliases(self):
        ctx = textwrap.dedent(
            """\
            Speakers:
            - Shawn Wang (guest) @ Smol AI
              aliases: Swyx, swyx
            """
        )
        result = extract_candidate_names(ctx)
        assert "Shawn Wang" in result.get("guest", [])
        assert result["aliases"]["Shawn Wang"] == ["Swyx", "swyx"]

    def test_extract_bio(self):
        ctx = textwrap.dedent(
            """\
            Speakers:
            - Bob Smith (guest) @ Stanford
              bio: PhD candidate working on RLHF and scaling laws.
            """
        )
        result = extract_candidate_names(ctx)
        assert "RLHF" in result["bios"]["Bob Smith"]

    def test_legacy_meta_md_still_works(self):
        """Old meta.md format (only name + role) keeps working via legacy regex paths."""
        ctx = "Channel/Host: Alice Chen\nTitle: Bob Smith — Topic\n"
        result = extract_candidate_names(ctx)
        assert "Alice Chen" in result.get("host", [])
        assert "Bob Smith" in result.get("guest", [])
        # No structured fields present → auxiliary maps should be absent
        assert "affiliations" not in result
        assert "aliases" not in result
        assert "bios" not in result

    def test_role_inferred_as_guest_for_unknown_role(self):
        """Roles outside _STRUCTURED_ROLE_KEYWORDS fall back to guest bucket."""
        ctx = textwrap.dedent(
            """\
            Speakers:
            - Alice Chen @ Anthropic
            """
        )
        result = extract_candidate_names(ctx)
        # No role keyword → treated as guest
        assert "Alice Chen" in result.get("guest", [])
        assert result["affiliations"]["Alice Chen"] == "Anthropic"

    def test_blank_line_inside_block_does_not_truncate(self):
        """Regression: blank lines between/within entries must not silently drop
        the rest of the Speakers block.

        Bug found by Codex review on commit a50039d: the line-scanner ended the
        block on the first blank line after any committed entry, so a meta.md
        author who separated speakers with blank lines for readability would
        lose every entry after the first.
        """
        ctx = textwrap.dedent(
            """\
            Speakers:
            - Alice Chen (host) @ Anthropic

            - Bob Smith (guest) @ Stanford
              bio: PhD in RLHF.

            - Carol Wu (guest) @ DeepMind
            """
        )
        result = extract_candidate_names(ctx)
        assert "Alice Chen" in result.get("host", [])
        assert "Bob Smith" in result.get("guest", [])
        assert "Carol Wu" in result.get("guest", [])
        assert result["affiliations"]["Bob Smith"] == "Stanford"
        assert result["affiliations"]["Carol Wu"] == "DeepMind"

    def test_name_alias_paren_preserved_not_treated_as_role(self):
        """Regression: ``Shawn Wang (Swyx)`` — the non-role parenthetical must
        survive as an alias, not be mistakenly recorded as the speaker's role.

        Bug found by Codex review on commit a50039d: ``_add()`` ran its generic
        paren-role handler on names emitted from the structured ``Speakers:``
        block, so ``Shawn Wang (Swyx)`` ended up with ``roles[Shawn Wang] ==
        "Swyx"`` and the prompt rendered ``Shawn Wang — Swyx``.
        """
        ctx = textwrap.dedent(
            """\
            Speakers:
            - Shawn Wang (Swyx) (guest) @ Smol AI
            """
        )
        result = extract_candidate_names(ctx)
        assert "Shawn Wang" in result.get("guest", [])
        # Role must come from the explicit (guest) keyword, NOT from "(Swyx)".
        assert result.get("roles", {}).get("Shawn Wang") in (None, "guest")
        # The non-role paren should be preserved as an alias.
        assert "Swyx" in result.get("aliases", {}).get("Shawn Wang", [])


# ===========================================================================
# D2. _resolve_context — caps on aliases / topics / prior_episodes / description
# ===========================================================================


class TestResolveContextCaps:
    """Regressions for unbounded-context inflation (Codex review on a50039d)."""

    def test_aliases_capped_in_speakers_block(self, tmp_path: Path):
        """A pathological aliases list must not bloat the prompt unboundedly."""
        from lattifai.cli.diarize import _resolve_context

        many_aliases = [f"alias{i}" for i in range(500)]
        meta_path = tmp_path / "ep.meta.md"
        meta_path.write_text(
            textwrap.dedent(
                """\
                ---
                speakers:
                  - name: Alice Chen
                    role: host
                    aliases: {aliases}
                ---
                """
            ).format(aliases=many_aliases),
            encoding="utf-8",
        )
        result = _resolve_context(str(meta_path))
        assert result is not None
        # Find the aliases line and count items
        for line in result.split("\n"):
            if line.strip().startswith("aliases:"):
                items = [a.strip() for a in line.split(":", 1)[1].split(",") if a.strip()]
                assert len(items) <= 8, f"aliases unbounded ({len(items)} entries)"
                break
        else:
            pytest.fail("aliases line missing")

    def test_topics_capped(self, tmp_path: Path):
        from lattifai.cli.diarize import _resolve_context

        many_topics = [f"topic_{i}" for i in range(200)]
        meta_path = tmp_path / "ep.meta.md"
        meta_path.write_text(
            textwrap.dedent(
                """\
                ---
                title: Bloat Test
                topics: {topics}
                ---
                """
            ).format(topics=many_topics),
            encoding="utf-8",
        )
        result = _resolve_context(str(meta_path))
        assert result is not None
        # Topics line should be capped
        for line in result.split("\n"):
            if line.startswith("Topics:"):
                items = [t.strip() for t in line[len("Topics:") :].split(",") if t.strip()]
                assert len(items) <= 20, f"topics unbounded ({len(items)} entries)"
                break
        else:
            pytest.fail("Topics line missing")

    def test_prior_episodes_capped(self, tmp_path: Path):
        from lattifai.cli.diarize import _resolve_context

        many_eps = [f"Episode {i}: title" for i in range(50)]
        meta_path = tmp_path / "ep.meta.md"
        meta_path.write_text(
            textwrap.dedent(
                """\
                ---
                title: Bloat Test
                prior_episodes: {eps}
                ---
                """
            ).format(eps=many_eps),
            encoding="utf-8",
        )
        result = _resolve_context(str(meta_path))
        assert result is not None
        # Count "- Episode" entries under "Prior episodes:"
        idx = result.find("Prior episodes:\n")
        assert idx >= 0
        block_text = result[idx + len("Prior episodes:\n") :]
        # block ends at next "Description:" or EOF
        end = block_text.find("\nDescription:")
        block = block_text[:end] if end >= 0 else block_text
        entries = [ln for ln in block.split("\n") if ln.strip().startswith("- ")]
        assert len(entries) <= 5, f"prior_episodes unbounded ({len(entries)} entries)"

    def test_naming_writes_named_file_to_disk(self, tmp_path: Path, monkeypatch):
        """Regression: ``lai diarize naming`` must write the named caption to disk
        instead of erroring out with ``Caption.write() got an unexpected keyword
        argument 'output_format'``.

        Bug discovered while eval'ing the lai-diarize skill against
        https://youtu.be/la0CaZ2R8EY on 2026-05-15. The CLI printed the speaker
        mapping but never produced the .named.json output because
        ``cap.write(path, output_format=...)`` passes a kwarg that
        ``Caption.write`` does not accept.
        """
        from lattifai.config.diarization import DiarizationLLMConfig

        # Minimal diarized input — Caption.read needs a recognised schema.
        src = {
            "supervisions": [
                {"start": 0.0, "end": 1.0, "text": "hi there", "speaker": "SPEAKER_00"},
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "hello back",
                    "speaker": "SPEAKER_01",
                },
                {
                    "start": 2.0,
                    "end": 3.0,
                    "text": "so what's your take?",
                    "speaker": "SPEAKER_00",
                },
                {
                    "start": 3.0,
                    "end": 4.0,
                    "text": "great question",
                    "speaker": "SPEAKER_01",
                },
            ]
        }
        in_path = tmp_path / "ep.diarized.json"
        out_path = tmp_path / "ep.named.json"
        in_path.write_text(json.dumps(src), encoding="utf-8")

        fake = FakeLLMClient([{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}])
        monkeypatch.setattr(DiarizationLLMConfig, "create_client", lambda self: fake)

        from lattifai.cli.diarize import naming

        naming(input_caption=str(in_path), output_caption=str(out_path))

        assert out_path.exists(), "named output file must be written to disk"
        written = json.loads(out_path.read_text(encoding="utf-8"))
        speakers = {s["speaker"] for s in written["supervisions"]}
        assert speakers == {"Alice", "Bob"}, f"speaker mapping not applied: {speakers}"

    def test_description_cap_exact_800_no_overflow(self, tmp_path: Path):
        """Regression: description with ellipsis must stay AT or below 800 chars,
        not 801.

        Bug found by Codex review on commit a50039d.
        """
        from lattifai.cli.diarize import _resolve_context

        long_para = "A" * 5000  # one long line
        meta_path = tmp_path / "ep.meta.md"
        meta_path.write_text(
            textwrap.dedent(
                f"""\
                ---
                title: Off-by-one Test
                ---

                {long_para}
                """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(meta_path))
        assert result is not None
        desc_idx = result.find("Description:\n")
        desc_body = result[desc_idx + len("Description:\n") :]
        assert len(desc_body) <= 800, f"description cap overflow: {len(desc_body)} chars"


# ===========================================================================
# E. _build_prompt — Speaker Background section
# ===========================================================================


class TestPromptSpeakerBackground:
    @pytest.fixture
    def speaker_texts(self):
        return {
            "SPEAKER_00": ["Welcome to the show", "What is your take?"],
            "SPEAKER_01": [
                "Thanks. I work on RLHF at Stanford.",
                "Scaling laws matter.",
            ],
        }

    def test_prompt_includes_speaker_background_when_rich(self, speaker_texts):
        context = textwrap.dedent(
            """\
            Channel/Host: Alice Chen
            Speakers:
            - Alice Chen (host) @ Anthropic
            - Bob Smith (guest) @ Stanford
              aliases: Bob
              bio: PhD candidate working on RLHF.
            """
        )
        llm = FakeLLMClient([{"SPEAKER_00": "Alice Chen", "SPEAKER_01": "Bob Smith"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        inferrer(speaker_texts, context=context)
        prompt = llm.prompts[0]
        assert "## Speaker Background" in prompt
        assert "Affiliation: Stanford" in prompt
        assert "Aliases: Bob" in prompt
        assert "Bio:" in prompt and "RLHF" in prompt

    def test_prompt_omits_speaker_background_when_no_signals(self, speaker_texts):
        """No affiliation/aliases/bio → no Speaker Background section."""
        context = "Channel/Host: Alice Chen\nTitle: Bob Smith — Topic\n"
        llm = FakeLLMClient([{"SPEAKER_00": "Alice Chen", "SPEAKER_01": "Bob Smith"}])
        inferrer = SpeakerNameInferrer(llm_client=llm)
        inferrer(speaker_texts, context=context)
        assert "## Speaker Background" not in llm.prompts[0]


# ===========================================================================
# F. _resolve_context — meta.md → context string
# ===========================================================================


class TestResolveContext:
    def test_speakers_block_with_affiliation_and_bio(self, tmp_path: Path):
        from lattifai.cli.diarize import _resolve_context

        meta_path = tmp_path / "ep.meta.md"
        meta_path.write_text(
            textwrap.dedent(
                """\
                ---
                title: Deep Dive
                speakers:
                  - name: Alice Chen
                    role: host
                    affiliation: Anthropic
                    aliases:
                      - Alice
                    bio: Host of the show.
                  - name: Bob Smith
                    role: guest
                    affiliation: Stanford AI Lab
                    bio: PhD candidate working on RLHF.
                topics:
                  - RLHF
                  - scaling laws
                prior_episodes:
                  - "Ep 42: pretraining"
                ---

                A discussion about reinforcement learning from human feedback.
                """
            ),
            encoding="utf-8",
        )

        result = _resolve_context(str(meta_path))
        assert result is not None
        assert "Channel/Host: Alice Chen" in result
        assert "Guests: Bob Smith" in result
        assert "Speakers:" in result
        assert "- Alice Chen (host) @ Anthropic" in result
        assert "- Bob Smith (guest) @ Stanford AI Lab" in result
        assert "aliases: Alice" in result
        assert "bio: Host of the show." in result
        assert "Topics: RLHF, scaling laws" in result
        assert "Prior episodes:" in result
        assert "Ep 42: pretraining" in result
        assert "reinforcement learning" in result

        # End-to-end: the same string must round-trip through extract_candidate_names.
        candidates = extract_candidate_names(result)
        assert candidates["affiliations"]["Alice Chen"] == "Anthropic"
        assert candidates["affiliations"]["Bob Smith"] == "Stanford AI Lab"
        assert candidates["aliases"]["Alice Chen"] == ["Alice"]
        assert "RLHF" in candidates["bios"]["Bob Smith"]

    def test_description_cap_at_800_chars(self, tmp_path: Path):
        """Description is capped at 800 chars rather than 3 lines."""
        from lattifai.cli.diarize import _resolve_context

        long_para = " ".join(["word"] * 300)  # ~1500 chars
        meta_path = tmp_path / "ep.meta.md"
        meta_path.write_text(
            textwrap.dedent(
                f"""\
                ---
                title: Long Description Ep
                ---

                {long_para}
                """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(meta_path))
        assert result is not None
        # Extract the description section and assert it stays under cap + safety.
        desc_idx = result.find("Description:\n")
        assert desc_idx >= 0
        desc_body = result[desc_idx + len("Description:\n") :]
        assert len(desc_body) <= 820  # 800 cap + ellipsis tolerance

    def test_legacy_meta_md_no_new_fields_still_resolves(self, tmp_path: Path):
        """meta.md without affiliation/aliases/bio fields still produces valid context."""
        from lattifai.cli.diarize import _resolve_context

        meta_path = tmp_path / "ep.meta.md"
        meta_path.write_text(
            textwrap.dedent(
                """\
                ---
                title: Legacy Episode
                speakers:
                  - name: Alice Chen
                    role: host
                  - name: Bob Smith
                    role: guest
                ---

                Short description.
                """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(meta_path))
        assert result is not None
        assert "Channel/Host: Alice Chen" in result
        assert "Guests: Bob Smith" in result
        # Speakers block is emitted (entries have no affiliation though)
        assert "- Alice Chen (host)" in result
        assert "- Bob Smith (guest)" in result
        # And the legacy regex paths still extract candidates
        candidates = extract_candidate_names(result)
        assert "Alice Chen" in candidates.get("host", [])
        assert "Bob Smith" in candidates.get("guest", [])
