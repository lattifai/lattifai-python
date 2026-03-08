"""Tests for translation pipeline workflow and artifact persistence."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from lattifai.config.translation import TranslationConfig
from lattifai.llm import BaseLLMClient
from lattifai.translation.base import BaseTranslator


class FakeLLMClient(BaseLLMClient):
    """Deterministic LLM client for translation pipeline tests."""

    def __init__(self, responses: list):
        super().__init__(api_key="test-key", model="fake-model")
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    async def generate(self, prompt: str, *, model=None, system=None, temperature=None) -> str:  # noqa: ARG002
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("No fake responses left for generate()")
        value = self._responses.pop(0)
        return str(value)

    async def generate_json(self, prompt: str, *, model=None, system=None, temperature=None):  # noqa: ARG002
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("No fake responses left for generate_json()")
        return self._responses.pop(0)


def _make_supervisions(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=text, translation="", target_lang=None) for text in texts]


def test_refined_mode_saves_full_artifact_set(tmp_path):
    responses = [
        {
            "terminology": [{"source": "Moat", "translation": "economic moat", "context": "business strategy"}],
            "metaphor_map": [{"source": "rear view mirror", "intent": "being chased", "strategy": "translate intent"}],
            "style": "technical",
            "register": "talk",
            "notes": "Keep concise",
        },
        [
            {"original": "Moat matters", "translated": "The moat matters a lot"},
            {"original": "They are catching up", "translated": "They are catching up quickly"},
        ],
        {
            "items": [
                {"revised": "The moat is critical", "critique": "More natural emphasis", "changed": True},
                {"revised": "They are rapidly catching up", "critique": "Sharper urgency", "changed": True},
            ]
        },
    ]
    client = FakeLLMClient(responses)
    config = TranslationConfig(
        provider="gemini",
        api_key="test-key",
        mode="refined",
        bilingual=True,
        save_artifacts=True,
        artifacts_dir=str(tmp_path),
        ask_refine_after_normal=False,
    )
    translator = BaseTranslator(config, client)
    supervisions = _make_supervisions("Moat matters", "They are catching up")

    asyncio.run(translator.translate_captions(supervisions, config))

    assert supervisions[0].translation == "The moat is critical"
    assert supervisions[1].translation == "They are rapidly catching up"
    assert (tmp_path / "01-analysis.md").exists()
    assert (tmp_path / "02-prompt.md").exists()
    assert (tmp_path / "03-draft.md").exists()
    assert (tmp_path / "04-critique.md").exists()
    assert (tmp_path / "05-revision.md").exists()
    assert (tmp_path / "translation.md").exists()


def test_normal_mode_can_upgrade_to_refined_without_retranslation():
    responses = [
        {
            "terminology": [{"source": "Moat", "translation": "economic moat", "context": "business strategy"}],
            "style": "technical",
            "register": "talk",
            "notes": "",
        },
        [
            {"original": "Moat", "translated": "economic moat"},
            {"original": "Competition", "translated": "competition"},
        ],
        {
            "items": [
                {"revised": "economic moat", "critique": "No change", "changed": False},
                {"revised": "market competition", "critique": "More specific", "changed": True},
            ]
        },
    ]
    client = FakeLLMClient(responses)
    config = TranslationConfig(
        provider="gemini",
        api_key="test-key",
        mode="normal",
        bilingual=True,
        ask_refine_after_normal=False,
    )
    translator = BaseTranslator(config, client)
    supervisions = _make_supervisions("Moat", "Competition")
    source_texts = [sup.text for sup in supervisions]

    asyncio.run(translator.translate_captions(supervisions, config))
    asyncio.run(translator.refine_existing_draft(supervisions, config, source_texts=source_texts))

    assert len(client.prompts) == 3
    assert "Review these" in client.prompts[-1]
    assert supervisions[1].translation == "market competition"


def test_refined_mode_accepts_legacy_review_array_schema():
    responses = [
        {"terminology": [], "style": "casual", "register": "vlog"},
        [
            {"original": "Hello", "translated": "hello"},
            {"original": "World", "translated": "world"},
        ],
        ["hello there", "world"],
    ]
    client = FakeLLMClient(responses)
    config = TranslationConfig(
        provider="gemini",
        api_key="test-key",
        mode="refined",
        bilingual=True,
        ask_refine_after_normal=False,
    )
    translator = BaseTranslator(config, client)
    supervisions = _make_supervisions("Hello", "World")

    asyncio.run(translator.translate_captions(supervisions, config))

    assert supervisions[0].translation == "hello there"
    assert supervisions[1].translation == "world"
