"""Assembles the prompts sent to the AI model when checking a transcription for likely OCR mistakes."""

from dataclasses import dataclass
from typing import Optional

from . import ocr_fragments as F


@dataclass
class TranscriptionReviewPromptSpec:
    """Holds the settings for one transcription-review request and builds its prompt text.

    Create one of these with the language the transcription is written in
    and whether kanbun-specific review guidance is needed, then call
    ``system_prompt()`` and ``user_prompt(text)`` to get the finished
    instructions to send the AI model. The wording itself lives in
    ``ocr_fragments.py``; this class only decides which pieces to include.
    The AI model's response schema intentionally has no field for the model
    name — ``TranscriptionReviewService`` records that separately, after
    the model's JSON response comes back.

    Attributes:
        language: The full language name the transcription is written in
                   (e.g. ``'Japanese'``).
        kanbun: Whether the transcription contains kanbun (see the plugin's
                module docstring) with kundoku annotations, so the model
                should treat those annotations as intentional rather than
                flagging them as errors.
        kanbun_main: Whether the transcription was produced in
                     main-character-only mode (annotations were
                     intentionally left out during OCR), so the model
                     should not flag their absence as an error.
        system_note: Extra instructions a professor typed in via the
                     ``--notes`` flag, appended to the system prompt, or
                     ``None`` if not supplied.
        user_note: Extra instructions a professor typed in via the
                   ``--notes`` flag, appended to the user prompt, or
                   ``None`` if not supplied.
    """

    language: str
    kanbun: bool = False
    kanbun_main: bool = False
    system_note: Optional[str] = None
    user_note: Optional[str] = None

    def system_prompt(self) -> str:
        """Build the system prompt: the model's reviewer role, kanbun guidance if applicable, and the required JSON report format.

        Returns:
            The finished system-prompt text.
        """
        kanbun_note = (
            F.TRANSCRIPTION_REVIEW_KANBUN_MAIN_NOTE if self.kanbun_main else
            F.TRANSCRIPTION_REVIEW_KANBUN_NOTE if self.kanbun else
            None
        )
        sections = [
            F.TRANSCRIPTION_REVIEW_ROLE.format(language=self.language),
            kanbun_note,
            F.TRANSCRIPTION_REVIEW_APPROACH,
            F.TRANSCRIPTION_REVIEW_SCHEMA,
            F.TRANSCRIPTION_REVIEW_RULES,
            F.ADDITIONAL_INSTRUCTIONS.format(note=self.system_note) if self.system_note else None,
        ]
        return "\n\n".join(s for s in sections if s)

    def user_prompt(self, text: str = "[transcription text would appear here]") -> str:
        """Build the user prompt: the actual transcription text to review, plus the review request.

        Args:
            text: The transcription text to check for OCR errors. Defaults
                  to a placeholder, used by ``--dry-run`` to preview the
                  prompt without a real transcription on hand.

        Returns:
            The finished user-prompt text.
        """
        parts = [
            F.TRANSCRIPTION_REVIEW_USER_BASE.format(language=self.language),
            F.TRANSCRIPTION_REVIEW_TEXT_BLOCK.format(text=text),
            F.ADDITIONAL_NOTES.format(note=self.user_note) if self.user_note else None,
        ]
        return "\n\n".join(s for s in parts if s)
