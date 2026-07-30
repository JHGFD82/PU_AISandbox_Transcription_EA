"""Turns an image of East-Asia-language text into typed text by calling the AI model, with kanbun and vertical-script support."""

import logging
import os
from typing import Optional, Any

from ..models import (
    model_supports_vision, get_vision_capable_models, resolve_model,
    get_model_system_role,
    get_model_max_completion_tokens, maybe_sync_model_pricing,
)
from .base_service import BaseService
from ..console import print_pass_result
from ..processors.image_processor import ImageProcessor
from ..tracking.token_tracker import TokenTracker
from .constants import MAX_RETRIES
from .prompts import OcrPromptSpec

from src.settings import OCR_ROLE
from ..settings import (
    OCR_TEMPERATURE,
    OCR_MAX_TOKENS,
    OCR_TOP_P,
    OCR_FREQUENCY_PENALTY,
    OCR_PRESENCE_PENALTY,
)



class ImageProcessorService(BaseService):
    """Reads text out of an image (OCR) for Chinese, Japanese, Korean, or English, via the AI model.

    Every ``transcribe`` command ends up calling this class's
    ``process_image_ocr()`` method once per image. Three settings —
    ``kanbun``, ``kanbun_main``, and ``tables`` — are set as plain
    attributes on the instance (by ``transcription-ea/plugin.py``, from the
    corresponding command-line flags) rather than passed as arguments to
    every method, since they stay the same for an entire transcription run.
    See the plugin's module docstring for what kanbun means.
    """

    # Which models this service's work should use — see
    # src/runtime/model_role.py. Read by BaseService._get_model().
    # The base transcription plugin's role: same job, different languages.
    model_role = OCR_ROLE

    def __init__(self, api_key: str, professor: Optional[str] = None, token_tracker: Optional[TokenTracker] = None, token_tracker_file: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, max_tokens: Optional[int] = None):
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
            token_tracker_file: Path to the token-usage file to read/write,
                                 or ``None`` to use the default location.
            model: A specific AI model to use instead of the default OCR
                   model (e.g. ``'gpt-4o'``), or ``None`` for the default.
            temperature: A custom sampling temperature (controls how
                         predictable vs. varied the model's output is), or
                         ``None`` to use the OCR default.
            top_p: A custom nucleus-sampling value (an alternative way of
                   controlling output variety), or ``None`` for the default.
            max_tokens: A custom maximum response length in tokens, or
                        ``None`` for the default.
        """
        super().__init__(api_key, professor, token_tracker, token_tracker_file, model, temperature, top_p, max_tokens)
        self.image_processor = ImageProcessor()
        self.kanbun: bool = False
        self.kanbun_main: bool = False
        self.tables: bool = False

    def _create_ocr_prompt(self, target_language: str, vertical: bool = False, spread: bool = False) -> tuple[str, str]:
        """Build the system and user prompt text for one OCR request, combining the caller's arguments with this instance's kanbun/tables settings."""
        spec = OcrPromptSpec(
            target_language=target_language,
            vertical=vertical,
            spread=spread,
            kanbun=self.kanbun,
            kanbun_main=self.kanbun_main,
            tables=self.tables,
            system_note=self.system_note,
            user_note=self.user_note,
        )
        return spec.system_prompt(), spec.user_prompt()

    def build_prompts(self, target_language: str, vertical: bool = False, spread: bool = False) -> tuple[str, str]:
        """Build the OCR prompts without calling the AI model — used to preview a request under ``--dry-run`` or ``--notes``.

        Args:
            target_language: The full language name to transcribe (e.g.
                              ``'Japanese'``).
            vertical: Whether the source text runs top-to-bottom in
                      right-to-left columns.
            spread: Whether the image is a two-page spread.

        Returns:
            A ``(system_prompt, user_prompt)`` pair of the exact text that
            would be sent to the AI model.
        """
        return self._create_ocr_prompt(target_language, vertical=vertical, spread=spread)

    def _build_refinement_prompt(self, target_language: str, vertical: bool = False, spread: bool = False) -> str:
        """Build the prompt for a refinement pass (``--passes`` 2 and later), asking the model to re-check its own prior transcription."""
        spec = OcrPromptSpec(target_language=target_language, vertical=vertical, spread=spread, kanbun=self.kanbun, kanbun_main=self.kanbun_main)
        return spec.refinement_prompt()

    def _call_ocr_api(self, model: str, system_role: str, system_prompt: str,
                      user_prompt: str, data_url: str, max_tokens: int) -> Any:
        """Call the OCR API with the correct token-limit parameter for the model."""
        messages: list[dict[str, Any]] = [
            {"role": system_role, "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                self._build_image_content_block(model, data_url)
            ]},
        ]
        temperature = self.custom_temperature if self.custom_temperature is not None else OCR_TEMPERATURE
        top_p = self.custom_top_p if self.custom_top_p is not None else OCR_TOP_P
        if self.custom_temperature is not None or self.custom_top_p is not None:
            logging.debug(f"OCR API params: temperature={temperature}, top_p={top_p}")
        return self._create_completion(
            model, messages, max_tokens,
            temperature=temperature, top_p=top_p,
            frequency_penalty=OCR_FREQUENCY_PENALTY,
            presence_penalty=OCR_PRESENCE_PENALTY,
        )

    def _call_refinement_api(self, model: str, system_role: str, system_prompt: str,
                              first_user_prompt: str, data_url: str,
                              prior_transcription: str, refinement_prompt: str,
                              max_tokens: int) -> Any:
        """Call the OCR API for a refinement pass, providing prior transcription as context."""
        messages: list[dict[str, Any]] = [
            {"role": system_role, "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": first_user_prompt},
                self._build_image_content_block(model, data_url)
            ]},
            {"role": "assistant", "content": prior_transcription},
            {"role": "user", "content": [
                {"type": "text", "text": refinement_prompt},
                self._build_image_content_block(model, data_url)
            ]},
        ]
        temperature = self.custom_temperature if self.custom_temperature is not None else OCR_TEMPERATURE
        top_p = self.custom_top_p if self.custom_top_p is not None else OCR_TOP_P
        return self._create_completion(
            model, messages, max_tokens,
            temperature=temperature, top_p=top_p,
            frequency_penalty=OCR_FREQUENCY_PENALTY,
            presence_penalty=OCR_PRESENCE_PENALTY,
        )

    def _run_single_refinement_pass(self, model: str, system_role: str, system_prompt: str,
                                     user_prompt: str, data_url: str, prior_transcription: str,
                                     refinement_prompt: str, max_tokens: int, pass_num: int) -> str:
        """Execute one refinement pass with retry logic and return the refined transcription."""
        def body(attempt: int) -> Any:
            logging.debug(f"Making OCR refinement API call (pass {pass_num})")
            response = self._call_refinement_api(
                model, system_role, system_prompt,
                user_prompt, data_url, prior_transcription,
                refinement_prompt, max_tokens,
            )
            self._record_response_usage(response, model, critical=True)
            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                content = response.choices[0].message.content
                if content is None:
                    logging.warning(f"Refinement pass {pass_num} returned None content.")
                    return None
                if not isinstance(content, str):
                    logging.warning(f"Unexpected content type {type(content)}: {content!r}. Retrying...")
                    return None
                if not content.strip():
                    logging.warning(f"Refinement pass {pass_num} returned empty content (attempt {attempt + 1}/{MAX_RETRIES}). Retrying...")
                    return None
                return content
            logging.warning("No choices in refinement API response. Retrying...")
            return None

        return self._run_with_retry(
            body, model, f"OCR refinement pass {pass_num}",
            timeout_msg=f"OCR refinement pass {pass_num} returned no content after maximum retries",
        )

    def process_image_ocr(self, file_path: str, target_language: str, output_format: str = "console", vertical: bool = False, spread: bool = False, passes: int = 1) -> str:
        """Read one image file and return the text the AI model transcribed from it.

        Automatically retries the AI model call if it returns an empty or
        malformed response. If ``passes`` is greater than 1, each
        additional pass sends the image and the previous pass's
        transcription back to the model, asking it to review and correct
        its own earlier work — useful for difficult handwriting or
        low-quality scans.

        Args:
            file_path: The full path to the image file to transcribe.
            target_language: The full language name to transcribe (e.g.
                              ``'Japanese'``).
            output_format: Unused by this method directly; kept for
                            interface compatibility with callers.
            vertical: Whether the source text runs top-to-bottom in
                      right-to-left columns.
            spread: Whether the image is a two-page spread.
            passes: How many OCR passes to run (1 = no refinement).

        Returns:
            The transcribed text after all requested passes complete.

        Raises:
            ValueError: If the resolved model doesn't support image input.
        """
        model = self._get_model()

        # Verify model supports vision
        if not model_supports_vision(model):
            vision_models = get_vision_capable_models()
            raise ValueError(
                f"Model '{model}' does not support image processing. "
                f"Please use one of the following vision-capable models: {vision_models}"
            )

        system_prompt, user_prompt = self._create_ocr_prompt(target_language, vertical=vertical, spread=spread)

        # Convert image to data URL
        try:
            data_url = self.image_processor.local_image_to_data_url(file_path)
        except Exception as e:
            logging.error(f"Failed to process image {os.path.basename(file_path)}: {e}")
            raise

        system_role = get_model_system_role(model)
        max_tokens = self.custom_max_tokens if self.custom_max_tokens is not None else get_model_max_completion_tokens(model, OCR_MAX_TOKENS)

        # --- Pass 1: initial transcription ---
        if passes > 1 and not self._suppress_inline_print:
            print(f"  Pass 1/{passes}: Initial transcription...")

        def body(attempt: int) -> Any:
            logging.debug(f'Making OCR API call to model: {model} (system role: {system_role}, max_tokens: {max_tokens})')
            response = self._call_ocr_api(model, system_role, system_prompt, user_prompt, data_url, max_tokens)
            self._record_response_usage(response, model, critical=True)
            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                content = response.choices[0].message.content
                if content is None:
                    logging.warning(f'Response content is None. Raw message: {response.choices[0].message}')
                    return None
                if not isinstance(content, str):
                    logging.warning(f'Unexpected content type {type(content)}: {content!r}. Retrying...')
                    return None
                if not content.strip():
                    logging.warning(f'Response returned empty content (attempt {attempt + 1}/{MAX_RETRIES}). Retrying...')
                    return None
                return content
            logging.warning('No choices in API response. Retrying...')
            return None

        transcription = self._run_with_retry(
            body, model, "OCR",
            timeout_msg="OCR returned no content after maximum retries — check model response format in debug logs",
        )

        if passes > 1 and not self._suppress_inline_print:
            print_pass_result(f"Pass 1/{passes} result", transcription)

        # --- Refinement passes ---
        refinement_prompt = self._build_refinement_prompt(target_language, vertical=vertical, spread=spread)
        for pass_num in range(2, passes + 1):
            if not self._suppress_inline_print:
                print(f"  Pass {pass_num}/{passes}: Refining...")
            logging.info(f"Starting OCR refinement pass {pass_num}/{passes}")
            transcription = self._run_single_refinement_pass(
                model, system_role, system_prompt,
                user_prompt, data_url, transcription,
                refinement_prompt, max_tokens, pass_num,
            )
            if pass_num < passes and not self._suppress_inline_print:
                print_pass_result(f"Pass {pass_num}/{passes} result", transcription)

        return transcription
