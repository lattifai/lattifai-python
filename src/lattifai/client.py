"""LattifAI client implementation."""

import logging
import os
from pathlib import Path
from typing import Any, Awaitable, BinaryIO, Callable, Dict, Optional, Union

import colorful
from dotenv import load_dotenv
from lhotse.utils import Pathlike

from lattifai.base_client import AsyncAPIClient, SyncAPIClient
from lattifai.errors import (
    AlignmentError,
    ConfigurationError,
    LatticeDecodingError,
    LatticeEncodingError,
    LattifAIError,
    ModelLoadError,
    SubtitleProcessingError,
    handle_exception,
)
from lattifai.io import SubtitleFormat, SubtitleIO
from lattifai.tokenizer import LatticeTokenizer
from lattifai.workers import Lattice1AlphaWorker

load_dotenv()


class LattifAI(SyncAPIClient):
    """Synchronous LattifAI client."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model_name_or_path: str = 'Lattifai/Lattice-1-Alpha',
        device: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, int] = 120.0,
        max_retries: int = 2,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get('LATTIFAI_API_KEY')
        if api_key is None:
            raise ConfigurationError(
                'The api_key client option must be set either by passing api_key to the client '
                'or by setting the LATTIFAI_API_KEY environment variable'
            )

        if base_url is None:
            base_url = os.environ.get('LATTIFAI_BASE_URL')
        if not base_url:
            base_url = 'https://api.lattifai.com/v1'

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )

        # Initialize components
        if not Path(model_name_or_path).exists():
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                model_path = snapshot_download(repo_id=model_name_or_path, repo_type='model')
            except LocalEntryNotFoundError:
                try:
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                    model_path = snapshot_download(repo_id=model_name_or_path, repo_type='model')
                except Exception as e:
                    raise ModelLoadError(model_name_or_path, original_error=e)
            except Exception as e:
                raise ModelLoadError(model_name_or_path, original_error=e)
        else:
            model_path = model_name_or_path

        # device setup
        if device is None:
            import torch

            device = 'cpu'
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'

        try:
            self.tokenizer = LatticeTokenizer.from_pretrained(
                client_wrapper=self,
                model_path=model_path,
                device=device,
            )
        except Exception as e:
            raise ModelLoadError(f'tokenizer from {model_path}', original_error=e)

        try:
            self.worker = Lattice1AlphaWorker(model_path, device=device, num_threads=8)
        except Exception as e:
            raise ModelLoadError(f'worker from {model_path}', original_error=e)

    def alignment(
        self,
        audio: Pathlike,
        subtitle: Pathlike,
        format: Optional[SubtitleFormat] = None,
        split_sentence: bool = False,
        output_subtitle_path: Optional[Pathlike] = None,
    ) -> str:
        """Perform alignment on audio and subtitle/text.

        Args:
            audio: Audio file path
            subtitle: Subtitle/Text to align with audio
            format: Output format (srt, vtt, ass, txt)
            split_sentence: Whether to split sentences during processing
            output_subtitle_path: Path to save output file

        Returns:
            Aligned subtitles in specified format

        Raises:
            SubtitleProcessingError: If subtitle file cannot be parsed
            LatticeEncodingError: If lattice graph generation fails
            AlignmentError: If audio alignment fails
            LatticeDecodingError: If lattice decoding fails
        """
        try:
            # step1: parse text or subtitles
            print(colorful.cyan(f'📖 Step 1: Reading subtitle file from {subtitle}'))
            try:
                supervisions = SubtitleIO.read(subtitle, format=format)
                print(colorful.green(f'         ✓ Parsed {len(supervisions)} subtitle segments'))
            except Exception as e:
                raise SubtitleProcessingError(
                    f'Failed to parse subtitle file: {subtitle}',
                    subtitle_path=str(subtitle),
                    context={'original_error': str(e)},
                )

            # step2: make lattice by call Lattifai API
            print(colorful.cyan('🔗 Step 2: Creating lattice graph from text'))
            try:
                lattice_id, lattice_graph = self.tokenizer.tokenize(supervisions, split_sentence=split_sentence)
                print(colorful.green(f'         ✓ Generated lattice graph with ID: {lattice_id}'))
            except Exception as e:
                text_content = ' '.join([sup.text for sup in supervisions]) if supervisions else ''
                raise LatticeEncodingError(text_content, original_error=e)

            # step3: align audio with text
            print(colorful.cyan(f'🎵 Step 3: Performing alignment on audio file: {audio}'))
            try:
                lattice_results = self.worker.alignment(audio, lattice_graph)
                print(colorful.green('         ✓ Alignment completed successfully'))
            except Exception as e:
                raise AlignmentError(
                    f'Audio alignment failed for {audio}',
                    audio_path=str(audio),
                    subtitle_path=str(subtitle),
                    context={'original_error': str(e)},
                )

            # step4: decode the lattice paths
            print(colorful.cyan('🔍 Step 4: Decoding lattice paths to final alignments'))
            try:
                alignments = self.tokenizer.detokenize(lattice_id, lattice_results)
                print(colorful.green(f'         ✓ Decoded {len(alignments)} aligned segments'))
            except Exception as e:
                print(colorful.red('         x Failed to decode lattice alignment results'))
                raise LatticeDecodingError(lattice_id, original_error=e)

            # step5: export alignments to target format
            if output_subtitle_path:
                try:
                    SubtitleIO.write(alignments, output_path=output_subtitle_path)
                    print(colorful.green(f'🎉🎉🎉🎉🎉 Subtitle file written to: {output_subtitle_path}'))
                except Exception as e:
                    raise SubtitleProcessingError(
                        f'Failed to write output file: {output_subtitle_path}',
                        subtitle_path=str(output_subtitle_path),
                        context={'original_error': str(e)},
                    )

            return output_subtitle_path or alignments

        except (SubtitleProcessingError, LatticeEncodingError, AlignmentError, LatticeDecodingError):
            # Re-raise our specific errors as-is
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            raise AlignmentError(
                'Unexpected error during alignment process',
                audio_path=str(audio),
                subtitle_path=str(subtitle),
                context={'original_error': str(e), 'error_type': e.__class__.__name__},
            )


if __name__ == '__main__':
    client = LattifAI()
    import sys

    if len(sys.argv) == 4:
        audio, subtitle, output = sys.argv[1:]
    else:
        audio = 'tests/data/SA1.wav'
        subtitle = 'tests/data/SA1.TXT'
        output = None

    alignments = client.alignment(audio, subtitle, output_subtitle_path=output, split_sentence=True)
