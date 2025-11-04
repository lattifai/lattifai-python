"""LattifAI client implementation."""

import asyncio
import os
from typing import Dict, List, Optional, Tuple, Union

import colorful
from lhotse.utils import Pathlike

from lattifai.base_client import AsyncAPIClient, SyncAPIClient
from lattifai.errors import (
    AlignmentError,
    ConfigurationError,
    LatticeDecodingError,
    LatticeEncodingError,
    LattifAIError,
    SubtitleProcessingError,
    handle_exception,
)
from lattifai.io import SubtitleFormat, SubtitleIO, Supervision
from lattifai.tokenizer import AsyncLatticeTokenizer
from lattifai.utils import _load_tokenizer, _load_worker, _resolve_model_path, _select_device


class LattifAI(SyncAPIClient):
    """Synchronous LattifAI client."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model_name_or_path: str = "Lattifai/Lattice-1-Alpha",
        device: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, int] = 120.0,
        max_retries: int = 2,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("LATTIFAI_API_KEY")
        if api_key is None:
            raise ConfigurationError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the LATTIFAI_API_KEY environment variable"
            )

        if base_url is None:
            base_url = os.environ.get("LATTIFAI_BASE_URL")
        if not base_url:
            base_url = "https://api.lattifai.com/v1"

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )

        model_path = _resolve_model_path(model_name_or_path)
        device = _select_device(device)

        self.tokenizer = _load_tokenizer(self, model_path, device)
        self.worker = _load_worker(model_path, device)
        self.device = device

    def alignment(
        self,
        audio: Pathlike,
        subtitle: Pathlike,
        format: Optional[SubtitleFormat] = None,
        split_sentence: bool = False,
        return_details: bool = False,
        output_subtitle_path: Optional[Pathlike] = None,
    ) -> Tuple[List[Supervision], Optional[Pathlike]]:
        """Perform alignment on audio and subtitle/text.

        Args:
            audio: Audio file path
            subtitle: Subtitle/Text to align with audio
            format: Input subtitle format (srt, vtt, ass, txt). Auto-detected if None
            split_sentence: Enable intelligent sentence re-splitting based on punctuation semantics
            return_details: Return word-level alignment details in Supervision.alignment field
            output_subtitle_path: Output path for aligned subtitle (optional)

        Returns:
            Tuple containing:
                - List of aligned Supervision objects with timing information
                - Output subtitle path (if output_subtitle_path was provided)

        Raises:
            SubtitleProcessingError: If subtitle file cannot be parsed
            LatticeEncodingError: If lattice graph generation fails
            AlignmentError: If audio alignment fails
            LatticeDecodingError: If lattice decoding fails
        """
        try:
            # step1: parse text or subtitles
            print(colorful.cyan(f"📖 Step 1: Reading subtitle file from {subtitle}"))
            try:
                supervisions = SubtitleIO.read(subtitle, format=format)
                print(colorful.green(f"         ✓ Parsed {len(supervisions)} subtitle segments"))
            except Exception as e:
                raise SubtitleProcessingError(
                    f"Failed to parse subtitle file: {subtitle}",
                    subtitle_path=str(subtitle),
                    context={"original_error": str(e)},
                )

            # step2: make lattice by call Lattifai API
            print(colorful.cyan("🔗 Step 2: Creating lattice graph from segments"))
            try:
                supervisions, lattice_id, lattice_graph = self.tokenizer.tokenize(
                    supervisions, split_sentence=split_sentence
                )
                print(colorful.green(f"         ✓ Generated lattice graph with ID: {lattice_id}"))
            except Exception as e:
                text_content = " ".join([sup.text for sup in supervisions]) if supervisions else ""
                raise LatticeEncodingError(text_content, original_error=e)

            # step3: search lattice graph with audio
            print(colorful.cyan(f"🔍 Step 3: Searching lattice graph with audio: {audio}"))
            try:
                lattice_results = self.worker.alignment(audio, lattice_graph)
                print(colorful.green("         ✓ Lattice search completed"))
            except Exception as e:
                raise AlignmentError(
                    f"Audio alignment failed for {audio}",
                    audio_path=str(audio),
                    subtitle_path=str(subtitle),
                    context={"original_error": str(e)},
                )

            # step4: decode lattice results to aligned segments
            print(colorful.cyan("🎯 Step 4: Decoding lattice results to aligned segments"))
            try:
                alignments = self.tokenizer.detokenize(
                    lattice_id, lattice_results, supervisions=supervisions, return_details=return_details
                )
                print(colorful.green(f"         ✓ Successfully aligned {len(alignments)} segments"))
            except LatticeDecodingError as e:
                print(colorful.red("         x Failed to decode lattice alignment results"))
                raise e
            except Exception as e:
                print(colorful.red("         x Failed to decode lattice alignment results"))
                raise LatticeDecodingError(lattice_id, original_error=e)

            # step5: export alignments to target format
            if output_subtitle_path:
                try:
                    SubtitleIO.write(alignments, output_path=output_subtitle_path)
                    print(colorful.green(f"🎉🎉🎉🎉🎉 Subtitle file written to: {output_subtitle_path}"))
                except Exception as e:
                    raise SubtitleProcessingError(
                        f"Failed to write output file: {output_subtitle_path}",
                        subtitle_path=str(output_subtitle_path),
                        context={"original_error": str(e)},
                    )
            return (alignments, output_subtitle_path)

        except (SubtitleProcessingError, LatticeEncodingError, AlignmentError, LatticeDecodingError):
            # Re-raise our specific errors as-is
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            raise AlignmentError(
                "Unexpected error during alignment process",
                audio_path=str(audio),
                subtitle_path=str(subtitle),
                context={"original_error": str(e), "error_type": e.__class__.__name__},
            )


class AsyncLattifAI(AsyncAPIClient):
    """Asynchronous LattifAI client."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model_name_or_path: str = "Lattifai/Lattice-1-Alpha",
        device: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, int] = 120.0,
        max_retries: int = 2,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("LATTIFAI_API_KEY")
        if api_key is None:
            raise ConfigurationError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the LATTIFAI_API_KEY environment variable"
            )

        if base_url is None:
            base_url = os.environ.get("LATTIFAI_BASE_URL")
        if not base_url:
            base_url = "https://api.lattifai.com/v1"

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )

        model_path = _resolve_model_path(model_name_or_path)
        device = _select_device(device)

        self.tokenizer = _load_tokenizer(self, model_path, device, tokenizer_cls=AsyncLatticeTokenizer)
        self.worker = _load_worker(model_path, device)
        self.device = device

    async def alignment(
        self,
        audio: Pathlike,
        subtitle: Pathlike,
        format: Optional[SubtitleFormat] = None,
        split_sentence: bool = False,
        return_details: bool = False,
        output_subtitle_path: Optional[Pathlike] = None,
    ) -> Tuple[List[Supervision], Optional[Pathlike]]:
        try:
            print(colorful.cyan(f"📖 Step 1: Reading subtitle file from {subtitle}"))
            try:
                supervisions = await asyncio.to_thread(SubtitleIO.read, subtitle, format=format)
                print(colorful.green(f"         ✓ Parsed {len(supervisions)} subtitle segments"))
            except Exception as e:
                raise SubtitleProcessingError(
                    f"Failed to parse subtitle file: {subtitle}",
                    subtitle_path=str(subtitle),
                    context={"original_error": str(e)},
                )

            print(colorful.cyan("🔗 Step 2: Creating lattice graph from segments"))
            try:
                supervisions, lattice_id, lattice_graph = await self.tokenizer.tokenize(
                    supervisions,
                    split_sentence=split_sentence,
                )
                print(colorful.green(f"         ✓ Generated lattice graph with ID: {lattice_id}"))
            except Exception as e:
                text_content = " ".join([sup.text for sup in supervisions]) if supervisions else ""
                raise LatticeEncodingError(text_content, original_error=e)

            print(colorful.cyan(f"🔍 Step 3: Searching lattice graph with audio: {audio}"))
            try:
                lattice_results = await asyncio.to_thread(self.worker.alignment, audio, lattice_graph)
                print(colorful.green("         ✓ Lattice search completed"))
            except Exception as e:
                raise AlignmentError(
                    f"Audio alignment failed for {audio}",
                    audio_path=str(audio),
                    subtitle_path=str(subtitle),
                    context={"original_error": str(e)},
                )

            print(colorful.cyan("🎯 Step 4: Decoding lattice results to aligned segments"))
            try:
                alignments = await self.tokenizer.detokenize(
                    lattice_id, lattice_results, supervisions=supervisions, return_details=return_details
                )
                print(colorful.green(f"         ✓ Successfully aligned {len(alignments)} segments"))
            except LatticeDecodingError as e:
                print(colorful.red("         x Failed to decode lattice alignment results"))
                raise e
            except Exception as e:
                print(colorful.red("         x Failed to decode lattice alignment results"))
                raise LatticeDecodingError(lattice_id, original_error=e)

            if output_subtitle_path:
                try:
                    await asyncio.to_thread(SubtitleIO.write, alignments, output_subtitle_path)
                    print(colorful.green(f"🎉🎉🎉🎉🎉 Subtitle file written to: {output_subtitle_path}"))
                except Exception as e:
                    raise SubtitleProcessingError(
                        f"Failed to write output file: {output_subtitle_path}",
                        subtitle_path=str(output_subtitle_path),
                        context={"original_error": str(e)},
                    )

            return (alignments, output_subtitle_path)

        except (SubtitleProcessingError, LatticeEncodingError, AlignmentError, LatticeDecodingError):
            raise
        except Exception as e:
            raise AlignmentError(
                "Unexpected error during alignment process",
                audio_path=str(audio),
                subtitle_path=str(subtitle),
                context={"original_error": str(e), "error_type": e.__class__.__name__},
            )


if __name__ == "__main__":
    client = LattifAI()
    import sys

    if len(sys.argv) == 5:
        audio, subtitle, output, split_sentence = sys.argv[1:]
        split_sentence = split_sentence.lower() in ("true", "1", "yes")
    else:
        audio = "tests/data/SA1.wav"
        subtitle = "tests/data/SA1.TXT"
        output = None
        split_sentence = False

    (alignments, output_subtitle_path) = client.alignment(
        audio, subtitle, output_subtitle_path=output, split_sentence=split_sentence, return_details=True
    )
