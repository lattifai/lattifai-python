from typing import List, Optional

from lhotse.utils import Pathlike

from ..config import CaptionConfig
from .caption import Caption
from .gemini_reader import GeminiReader, GeminiSegment
from .gemini_writer import GeminiWriter
from .reader import CaptionReader, InputCaptionFormat
from .supervision import Supervision
from .text_parser import normalize_text
from .writer import CaptionWriter

__all__ = [
    "Caption",
    "CaptionReader",
    "CaptionWriter",
    "CaptionIO",
    "Supervision",
    "GeminiReader",
    "GeminiWriter",
    "GeminiSegment",
    "normalize_text",
    "Captioner",
]


class CaptionIO:
    """
    I/O interface for reading and writing caption files.

    This class provides both Caption-based and legacy List[Supervision] interfaces.
    """

    def __init__(self):
        pass

    @classmethod
    def read(
        cls, caption: Pathlike, format: Optional[InputCaptionFormat] = None, normalize_text: Optional[bool] = False
    ) -> Caption:
        """
        Read caption file and return Caption object with metadata.

        Args:
            caption: Path to caption file or caption content
            format: Caption format (auto-detected if not provided)
            normalize_text: Whether to normalize text during reading

        Returns:
            Caption object containing supervisions and metadata
        """
        return CaptionReader.read(caption, format=format, normalize_text=normalize_text)

    @classmethod
    def read_supervisions(
        cls, caption: Pathlike, format: Optional[InputCaptionFormat] = None, normalize_text: Optional[bool] = False
    ) -> List[Supervision]:
        """
        Legacy method: Read caption file and return only supervisions (backward compatibility).

        Args:
            caption: Path to caption file or caption content
            format: Caption format (auto-detected if not provided)
            normalize_text: Whether to normalize text during reading

        Returns:
            List of Supervision objects
        """
        caption_obj = cls.read(caption, format=format, normalize_text=normalize_text)
        return caption_obj.supervisions

    @classmethod
    def write(
        cls,
        caption: Caption,
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> Pathlike:
        """
        Write Caption object to file.

        Args:
            caption: Caption object or list of supervisions to write
            output_path: Path to output caption file
            include_speaker_in_text: Whether to include speaker labels in text

        Returns:
            Path to the written file
        """
        # Support both Caption objects and List[Supervision] for backward compatibility
        if isinstance(caption, Caption):
            supervisions = caption.supervisions
        elif isinstance(caption, list):
            supervisions = caption
        else:
            raise TypeError(f"Expected Caption or List[Supervision], got {type(caption)}")

        return CaptionWriter.write(supervisions, output_path, include_speaker_in_text=include_speaker_in_text)


class Captioner:
    """
    High-level interface for caption reading and writing with configuration.
    """

    def __init__(self, config: Optional[CaptionConfig] = None):
        if config is None:
            config = CaptionConfig()
        self.config = config

    def read(
        self,
        input_path: Optional[Pathlike] = None,
        format: Optional[InputCaptionFormat] = None,
        normalize_text: Optional[bool] = False,
    ) -> Caption:
        """
        Read caption file and return Caption object.

        Args:
            input_path: Path to caption file (uses config if not provided)
            format: Caption format (uses config if not provided)
            normalize_text: Whether to normalize text (uses config if not provided)

        Returns:
            Caption object containing supervisions and metadata
        """
        if not input_path:
            input_path = self.config.input_path
            assert self.config.check_sanity() is True
        return CaptionIO.read(
            input_path,
            format=format or self.config.input_format,
            normalize_text=normalize_text or self.config.normalize_text,
        )

    def read_supervisions(
        self,
        input_path: Optional[Pathlike] = None,
        format: Optional[InputCaptionFormat] = None,
        normalize_text: Optional[bool] = False,
    ) -> List[Supervision]:
        """
        Legacy method: Read caption file and return only supervisions.

        Args:
            input_path: Path to caption file (uses config if not provided)
            format: Caption format (uses config if not provided)
            normalize_text: Whether to normalize text (uses config if not provided)

        Returns:
            List of Supervision objects
        """
        caption = self.read(input_path, format, normalize_text)
        return caption.supervisions

    def write(
        self,
        caption: Caption,
        output_path: Optional[Pathlike] = None,
    ) -> Pathlike:
        """
        Write Caption object to file.

        Args:
            caption: Caption object or list of supervisions to write
            output_path: Path to output file (uses config if not provided)

        Returns:
            Path to the written file
        """
        output_path = output_path or self.config.output_path
        return CaptionIO.write(
            caption,
            output_path,
            include_speaker_in_text=self.config.include_speaker_in_text,
        )
