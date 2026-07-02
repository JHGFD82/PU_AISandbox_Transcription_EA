"""Assembles the system and user prompts sent to the AI model for East-Asia OCR (image-to-text transcription)."""

from dataclasses import dataclass
from typing import Optional

from . import ocr_fragments as F


@dataclass
class OcrPromptSpec:
    """Holds the settings for one OCR request and builds the prompt text to send the AI model from them.

    A "prompt" is the written instructions sent to the AI model along with
    the image: the system prompt sets the model's role and ground rules,
    the user prompt is the specific per-image request. Create one of these
    with the settings for a particular transcription (which language,
    whether the script is vertical, whether kanbun handling is needed, and
    so on), then call ``system_prompt()``, ``user_prompt()``, and — for OCR
    passes after the first — ``refinement_prompt()`` to get the finished
    text. All of the actual wording lives in ``ocr_fragments.py``; this
    class only decides which fragments to include and in what order, based
    on which options are turned on.

    Attributes:
        target_language: The full language name to transcribe (e.g.
                          ``'Japanese'``), used both to tell the model what
                          language to expect and to look up
                          language-specific script guidance.
        vertical: Whether the source text runs top-to-bottom in
                  right-to-left columns, as in traditional Japanese and
                  Chinese printing, rather than left-to-right rows.
        kanbun: Whether the image contains kanbun (see the plugin's module
                docstring for what that means) and every kundoku annotation
                should be transcribed exactly as written.
        kanbun_main: Whether the image contains kanbun but only the large
                     main-line characters should be transcribed, omitting
                     the small annotations. Mutually exclusive with
                     ``kanbun`` at the command-line level.
        spread: Whether the image is a two-page spread (two facing pages
                scanned together) that should be read as one continuous
                document.
        tables: Whether to hint the model that any tabular data in the
                image should be returned as a Markdown table.
        system_note: Extra instructions a professor typed in via the
                     ``--notes`` flag, appended to the system prompt, or
                     ``None`` if not supplied.
        user_note: Extra instructions a professor typed in via the
                   ``--notes`` flag, appended to the user prompt, or
                   ``None`` if not supplied.
    """

    target_language: str
    vertical: bool = False
    kanbun: bool = False
    kanbun_main: bool = False
    spread: bool = False
    tables: bool = False
    system_note: Optional[str] = None
    user_note: Optional[str] = None

    def _script_note(self) -> str:
        """Pick the script-specific guidance to include: kanbun-specific if either kanbun flag is set, else the target language's general guidance (if any)."""
        if self.kanbun_main:
            return F.KANBUN_MAIN_SCRIPT_NOTE
        if self.kanbun:
            return F.KANBUN_SCRIPT_NOTE
        return F.OCR_SCRIPT_GUIDANCE.get(self.target_language, "")

    def system_prompt(self) -> str:
        """Build the system prompt: the model's role, script guidance, and transcription rules.

        Returns:
            The finished system-prompt text, with only the sections that
            apply to this request's settings (e.g. the vertical-script
            block is only included when ``vertical`` is ``True``).
        """
        script_note = self._script_note()
        kanbun_note = (
            F.KANBUN_MAIN_OCR_NOTE if self.kanbun_main else
            F.KANBUN_OCR_NOTE if self.kanbun else
            None
        )
        sections = [
            F.OCR_SYSTEM_BASE.format(target=self.target_language),
            ("SCRIPT NOTES:\n" + script_note) if script_note else None,
            F.OCR_VERTICAL_BLOCK if self.vertical else None,
            F.OCR_SPREAD_NOTE if self.spread else None,
            F.ADDITIONAL_INSTRUCTIONS.format(note=kanbun_note) if kanbun_note else None,
            F.OCR_RULES,
            F.TABLE_HINT_SYSTEM if self.tables else None,
            F.ADDITIONAL_INSTRUCTIONS.format(note=self.system_note) if self.system_note else None,
        ]
        return "\n\n".join(s for s in sections if s)

    def user_prompt(self) -> str:
        """Build the user prompt: the actual per-image transcription request sent alongside the image.

        Returns:
            The finished user-prompt text, with only the sections that
            apply to this request's settings.
        """
        script_note = self._script_note()
        if self.kanbun_main:
            base = F.OCR_USER_BASE_KANBUN_MAIN
        elif self.kanbun:
            base = F.OCR_USER_BASE_KANBUN
        else:
            base = F.OCR_USER_BASE.format(target=self.target_language)
        parts = [
            base,
            ("SCRIPT REMINDER: " + script_note) if script_note else None,
            F.OCR_VERTICAL_REINFORCEMENT if self.vertical else None,
            F.OCR_USER_RULES,
            F.TABLE_HINT_USER if self.tables else None,
            F.ADDITIONAL_NOTES.format(note=self.user_note) if self.user_note else None,
        ]
        return "\n\n".join(s for s in parts if s)

    def refinement_prompt(self) -> str:
        """Build the prompt for an OCR refinement pass (``--passes`` > 1): asks the model to re-check its own prior transcription against the image.

        Returns:
            The finished refinement-prompt text, sent along with the
            image and the model's previous transcription attempt.
        """
        script_note = self._script_note()
        parts = [
            F.OCR_REFINEMENT_BASE,
            ("SCRIPT REMINDER: " + script_note) if script_note else None,
            F.OCR_VERTICAL_REINFORCEMENT if self.vertical else None,
        ]
        return "\n\n".join(s for s in parts if s)
