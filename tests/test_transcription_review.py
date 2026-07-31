"""Tests for plugins/transcription-ea/plugin.py::_run_transcription_review.

This is a regression test for a bug where plugin.py's run() called
sandbox.process_transcription_review(...), a SandboxProcessor method that
never existed (not a Mixin, not defined anywhere), so every
transcription_review run for Chinese/Japanese/Korean raised AttributeError.
The fix replaced that call with this module-level helper, mirroring the
base transcription plugin's own _run_transcription_review() (see
plugins/transcription/tests/test_transcription_review.py) but extended to
pass kanbun/kanbun_main through to the service.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.errors import CLIError
from src.runtime.plugin_loader import load_plugins
from src.runtime.sandbox_processor import SandboxProcessor

# The plugin folder name has a hyphen ("transcription-ea"), so it can't be
# imported with normal dotted syntax (`import plugins.transcription-ea.plugin`
# is invalid Python). The plugin loader registers it in sys.modules under
# the literal string key "pu_plugin.transcription-ea.plugin" instead — see
# _load_one() in src/runtime/plugin_loader.py.
load_plugins(Path(__file__).resolve().parents[2])
_run_transcription_review = sys.modules["pu_plugin.transcription-ea.plugin"]._run_transcription_review


def _make_sandbox() -> SandboxProcessor:
    """Return a bare SandboxProcessor with transcription_review_service mocked."""
    proc = SandboxProcessor.__new__(SandboxProcessor)
    proc.transcription_review_service = MagicMock()
    return proc


class TestRunTranscriptionReview:

    def test_delegates_to_service_and_prints(self, capsys):
        sandbox = _make_sandbox()
        sandbox.transcription_review_service.review_transcription.return_value = '{"errors": []}'
        _run_transcription_review(sandbox, "some text", "Japanese")
        out = capsys.readouterr().out
        assert '{"errors": []}' in out

    def test_saves_to_output_file(self, tmp_path):
        sandbox = _make_sandbox()
        sandbox.transcription_review_service.review_transcription.return_value = '{"errors": []}'
        out_file = str(tmp_path / "report.json")
        _run_transcription_review(sandbox, "text", "Japanese", output_file=out_file)
        assert (tmp_path / "report.json").read_text() == '{"errors": []}'

    def test_exception_wrapped_as_cli_error(self):
        sandbox = _make_sandbox()
        sandbox.transcription_review_service.review_transcription.side_effect = RuntimeError("API fail")
        with pytest.raises(CLIError, match="Error during transcription review"):
            _run_transcription_review(sandbox, "text", "Japanese")

    def test_kanbun_and_kanbun_main_passed_through_to_service(self):
        sandbox = _make_sandbox()
        sandbox.transcription_review_service.review_transcription.return_value = '{}'
        _run_transcription_review(sandbox, "text", "Japanese", kanbun=True, kanbun_main=False)
        sandbox.transcription_review_service.review_transcription.assert_called_once_with(
            "text", "Japanese", kanbun=True, kanbun_main=False
        )

    def test_kanbun_and_kanbun_main_default_to_false(self):
        sandbox = _make_sandbox()
        sandbox.transcription_review_service.review_transcription.return_value = '{}'
        _run_transcription_review(sandbox, "text", "Japanese")
        sandbox.transcription_review_service.review_transcription.assert_called_once_with(
            "text", "Japanese", kanbun=False, kanbun_main=False
        )


class TestReviewUsesTheModelsItWasToldTo:
    """The same models the base plugin reviews with, not whatever is cheapest.

    This service is the East Asian counterpart of the base plugin's, not a
    different job. It carried no model preference at all, so it fell through to
    the cheapest model in the catalogue — meaning a professor's choice was
    honoured for English and silently ignored for Japanese, Korean and Chinese,
    which is the harder reading of the two.
    """

    def _service_class(self):
        # Already loaded at the top of this module.
        return sys.modules["src.services.transcription_review_service"].TranscriptionReviewService

    def test_it_declares_a_model_preference(self):
        assert self._service_class().model_role is not None

    def test_it_is_the_same_preference_the_base_plugin_uses(self):
        from src.settings import TRANSCRIPTION_REVIEW_ROLE

        assert self._service_class().model_role is TRANSCRIPTION_REVIEW_ROLE

    def test_it_names_models_rather_than_leaving_the_choice_open(self):
        """An empty preference is the same as none: the cheapest wins again."""
        role = self._service_class().model_role
        assert role.models, "the preference names no models, so nothing is preferred"

    def test_english_and_east_asian_review_agree_on_the_models(self):
        """The two are the same job; only the guidance given to them differs."""
        base = sys.modules["src.services.transcription_review_service"]
        from src.settings import TRANSCRIPTION_REVIEW_ROLE

        assert base.TranscriptionReviewService.model_role.models == TRANSCRIPTION_REVIEW_ROLE.models
