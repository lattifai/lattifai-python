"""Local LLM client using HuggingFace transformers.

Supports any causal-LM or multimodal model loadable via
AutoModelForCausalLM / AutoModelForImageTextToText.

Usage:
    from lattifai.llm import create_client

    client = create_client("transformers", model="google/gemma-4-E2B")
    result = client.generate_sync("Summarize the following text: ...")
"""

import asyncio
import logging
from typing import Any, Optional

from lattifai.llm.base import BaseLLMClient, parse_json_response

logger = logging.getLogger(__name__)


class TransformersClient(BaseLLMClient):
    """Local LLM client backed by HuggingFace transformers.

    Loads models lazily on first use. Supports both pure-text (AutoModelForCausalLM)
    and multimodal (AutoModelForImageTextToText) architectures.

    Args:
        model: HuggingFace model ID (e.g. "google/gemma-4-E2B").
        device: Device placement — "auto" (default), "cpu", "cuda", "mps".
        dtype: Torch dtype string — "bfloat16", "float16", "float32", or None (auto).
        max_new_tokens: Default max tokens for generation.
    """

    # Model names that require AutoModelForImageTextToText
    _MULTIMODAL_KEYWORDS = ("gemma-4", "gemma-3n")

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        device: str = "auto",
        dtype: Optional[str] = None,
        max_new_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(api_key=None, model=model, **kwargs)
        self._device = device
        self._dtype_str = dtype
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    @property
    def provider_name(self) -> str:
        return "transformers"

    def _is_multimodal(self, model_name: str) -> bool:
        """Check if model needs AutoModelForImageTextToText."""
        name_lower = model_name.lower()
        return any(k in name_lower for k in self._MULTIMODAL_KEYWORDS)

    def _load_model(self):
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return

        import torch

        model_name = self._resolve_model()

        # Resolve dtype
        if self._dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif self._dtype_str == "float16":
            dtype = torch.float16
        elif self._dtype_str == "float32":
            dtype = torch.float32
        elif self._dtype_str is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self._dtype_str}")

        # Resolve device_map
        device_map = self._device
        if device_map == "cuda":
            device_map = "cuda:0"

        if self._is_multimodal(model_name):
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._tokenizer = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=dtype, device_map=device_map
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map)

        logger.info("Loaded %s on %s (dtype=%s)", model_name, device_map, dtype)

    def _generate_text(self, messages: list[dict], temperature: Optional[float], max_new_tokens: Optional[int]) -> str:
        """Synchronous text generation from chat messages."""
        import torch

        self._load_model()
        tokenizer = self._tokenizer
        model = self._model

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = {"max_new_tokens": max_new_tokens or self._max_new_tokens}
        if temperature is not None and temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Run blocking inference in executor to avoid stalling the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_text, messages, temperature, None)

    async def generate_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        json_system = ((system + "\n") if system else "") + "Respond only with valid JSON. No commentary."
        text = await self.generate(prompt, model=model, system=json_system, temperature=temperature)
        return parse_json_response(text)
