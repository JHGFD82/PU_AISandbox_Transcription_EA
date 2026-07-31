"""Checks a previously-produced transcription for likely OCR mistakes and reports them as a structured summary."""

import json
import logging
import re
from typing import Any, Optional

from ..models import (
    get_model_system_role,
)
from ..tracking.token_tracker import TokenTracker
from .api_errors import handle_api_errors
from .base_service import BaseService
from .prompts import TranscriptionReviewPromptSpec
from ..settings import (
    TRANSCRIPTION_REVIEW_ROLE,
    TRANSCRIPTION_REVIEW_TEMPERATURE,
    TRANSCRIPTION_REVIEW_TOP_P,
    TRANSCRIPTION_REVIEW_MAX_TOKENS,
)


class TranscriptionReviewService(BaseService):
    """Reviews AI-transcribed text for OCR errors and returns a structured JSON report.

    The model assesses the overall quality of the transcription, identifies the
    probable source, and reports each suspected error with candidates in descending
    confidence order.  The actual model name used is injected into ``meta.model``
    by the service after parsing the response, rather than relying on the model to
    self-report its name.
    """

    # The same models the base plugin reviews with. This service is the East
    # Asian counterpart of that one, not a different job, and reviewing a
    # Japanese transcription is if anything the harder of the two — so it must
    # not quietly run on something else. Without this it fell through to
    # whichever model in the catalogue happened to be cheapest, which meant a
    # professor's choice was honoured for English and ignored for Japanese,
    # Korean and Chinese.
    model_role = TRANSCRIPTION_REVIEW_ROLE

    def __init__(
        self,
        api_key: str,
        professor: Optional[str] = None,
        token_tracker: Optional[TokenTracker] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Set up the service with the professor's API key and this run's model/sampling settings.

        Args:
            api_key: The professor's PortKey API key (a secret string that
                     authorizes calls to the AI model), already resolved by
                     ``SandboxProcessor``.
            professor: The professor's identifier (e.g. ``'heller'``), used
                       when recording token usage.
            token_tracker: The shared object that records how many tokens
                           (the small chunks of text the model processes and
                           bills by) this run consumes, or ``None`` to
                           create one automatically.
            model: A specific AI model to use instead of the default review
                   model, or ``None`` for the default.
            temperature: A custom sampling temperature (controls how
                         predictable vs. varied the model's output is), or
                         ``None`` to use the review default.
            top_p: A custom nucleus-sampling value (an alternative way of
                   controlling output variety), or ``None`` for the default.
            max_tokens: A custom maximum response length in tokens, or
                        ``None`` for the default.
        """
        super().__init__(api_key, professor, token_tracker, None, model, temperature, top_p, max_tokens)

    def build_prompts(
        self,
        language: str,
        kanbun: bool = False,
        kanbun_main: bool = False,
        text: str = "[transcription text would appear here]",
    ) -> tuple[str, str]:
        """Build the review prompts without calling the AI model — used to preview a request under ``--dry-run`` or ``--notes``.

        Args:
            language: The full language name the transcription is written
                      in (e.g. ``'Japanese'``).
            kanbun: Whether the transcription contains kanbun with kundoku
                    annotations that should be treated as intentional
                    rather than flagged as errors.
            kanbun_main: Whether the transcription was produced in
                         main-character-only mode, so the model shouldn't
                         flag missing annotations as errors.
            text: Placeholder or real transcription text to show in the
                  preview.

        Returns:
            A ``(system_prompt, user_prompt)`` pair of the exact text that
            would be sent to the AI model.
        """
        spec = TranscriptionReviewPromptSpec(
            language=language,
            kanbun=kanbun,
            kanbun_main=kanbun_main,
            system_note=self.system_note,
            user_note=self.user_note,
        )
        return spec.system_prompt(), spec.user_prompt(text)

    def _call_api(
        self,
        model: str,
        system_role: str,
        system_prompt: str,
        user_prompt: str,
    ) -> Any:
        """Send the review prompts to the AI model and return its raw response."""
        temperature, top_p, max_tokens = self._resolve_sampling_params(
            model, TRANSCRIPTION_REVIEW_TEMPERATURE, TRANSCRIPTION_REVIEW_TOP_P, TRANSCRIPTION_REVIEW_MAX_TOKENS
        )
        messages = [
            {"role": system_role, "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._create_completion(
            model, messages, max_tokens,
            temperature=temperature, top_p=top_p,
        )

    @staticmethod
    def _inject_model_and_validate(raw: str, model: str, language: str) -> str:
        """Strip markdown fences, inject the model name, and return pretty-printed JSON.

        If the response cannot be parsed as JSON, returns the raw string with a
        logged warning so the caller still has something to show the user.
        """
        # Some models wrap JSON in ```json ... ``` despite being told not to.
        clean = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        clean = re.sub(r"\n?```$", "", clean).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            logging.warning(
                "TranscriptionReviewService: model returned non-JSON response; "
                "displaying raw output."
            )
            return raw

        # Inject the actual model name — more reliable than asking the model to self-report.
        if isinstance(data.get("meta"), dict):
            data["meta"]["model"] = model
            if not data["meta"].get("language"):
                data["meta"]["language"] = language

        return json.dumps(data, ensure_ascii=False, indent=2)

    def review_transcription(
        self,
        text: str,
        language: str,
        kanbun: bool = False,
        kanbun_main: bool = False,
    ) -> str:
        """Check a transcription for likely OCR mistakes and return the AI model's findings as a JSON report string.

        The report includes an overall quality assessment, a guess at the
        source document's genre/period, and a list of specific suspected
        errors with one or more corrected candidates each, most-likely
        first.

        Args:
            text: The transcription text to check for errors — the
                  *output* of a prior transcription run, not the original
                  image or document.
            language: The full language name the transcription is written
                      in (e.g. ``'Japanese'``), as returned by
                      ``parse_single_language_code``.
            kanbun: Whether the text contains kanbun with kundoku
                    annotations that should be treated as intentional
                    rather than flagged as errors.
            kanbun_main: Whether the transcription was produced in
                         main-character-only mode (okurigana, furigana,
                         kaeriten intentionally omitted), so their absence
                         shouldn't be flagged as an error.

        Returns:
            A JSON-formatted string report (pretty-printed for readability),
            or an empty string if the AI model's response had no usable
            content.
        """
        model = self._get_model()
        system_role = get_model_system_role(model)
        spec = TranscriptionReviewPromptSpec(
            language=language,
            kanbun=kanbun,
            kanbun_main=kanbun_main,
            system_note=self.system_note,
            user_note=self.user_note,
        )
        system_prompt = spec.system_prompt()
        user_prompt = spec.user_prompt(text)

        logging.info(f"Reviewing transcription ({language}, {len(text)} chars) with model: {model}")

        try:
            response = self._call_api(model, system_role, system_prompt, user_prompt)
        except Exception as e:
            handle_api_errors(e, model)
            raise

        self._record_response_usage(response, model)
        content = self._extract_response_content(response)
        if content is not None:
            return self._inject_model_and_validate(content.strip(), model, language)
        return ""
