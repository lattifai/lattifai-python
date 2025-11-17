"""
Prompt management utilities for workflows
"""

from pathlib import Path
from typing import Optional


class PromptLoader:
    """Load and manage prompts from the prompts directory"""

    def __init__(self):
        self.prompts_dir = Path(__file__).parent

    def load_prompt(self, category: str, name: str) -> str:
        """
        Load a prompt from the prompts directory

        Args:
            category: Subdirectory name (e.g., 'gemini', 'openai')
            name: Prompt file name without extension (e.g., 'transcription_gem')

        Returns:
            Prompt text content

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        prompt_path = self.prompts_dir / category / f"{name}.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")

        return prompt_path.read_text(encoding="utf-8").strip()

    def get_gemini_transcription_prompt(self) -> str:
        """Get the Gemini transcription Gem prompt"""
        return self.load_prompt("gemini", "transcription_gem")


# Global instance
_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get or create the global PromptLoader instance"""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader
