"""
Tests for plugin.py's web UI composer integration: the vertical/spread/
passes/kanbun-mode/preserve-tables fields contributed to the base
transcription plugin's transcribe composer via
register_extension_ui_hooks()/ExtensionUiHooks (src/runtime/ui_action.py).

See plugins/translation-ea/tests/test_kanbun_ui_hook.py for the sibling
mechanism on the translate side, and this plugin's plugin.py "Web UI
composer integration" section for why vertical/spread/passes aren't
touched by _apply_ea_transcribe_ui_hook (they're read directly by the base
plugin's own run_ui_action instead, since they're real keyword arguments
process_image/process_image_folder accept rather than sandbox attributes).

Registered under action_id="transcribe" — see ExtensionUiHooks's docstring
for why action_id is part of the registry key (translation-ea registers its
own, unrelated Kanbun checkbox under the same "jp" token but
action_id="translate", so the two must not collide).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.runtime.plugin_loader import load_plugins
from src.runtime.ui_action import apply_extension_ui_hooks, get_extension_ui_fields

_PLUGINS_DIR = Path(__file__).resolve().parents[2]

# Loading both plugins registers these hooks as a side effect of importing
# plugins/transcription-ea/plugin.py (see its "Web UI composer integration"
# section) — same reason test_transcription_cli.py loads both plugins via
# _make_parser()/load_plugins() rather than importing plugin.py directly.
# This also loads translation/translation-ea (same "plugins" directory),
# which is exactly why action_id matters here — see the module docstring.
load_plugins(_PLUGINS_DIR)


def _make_sandbox():
    sandbox = SimpleNamespace()
    sandbox.image_processor_service = SimpleNamespace(kanbun=False, kanbun_main=False, tables=False)
    return sandbox


class TestFieldRegistration:

    @pytest.mark.parametrize("token", ["zh", "jp", "kr"])
    def test_each_ea_language_offers_the_full_field_set(self, token):
        names = {f.name for f in get_extension_ui_fields("transcribe", token)}
        assert names == {"vertical", "spread", "passes", "kanbun_mode", "preserve_tables"}

    def test_english_gets_no_ea_fields(self):
        assert get_extension_ui_fields("transcribe", "en") == []

    def test_kanbun_mode_is_a_select_with_three_choices(self):
        field = next(f for f in get_extension_ui_fields("transcribe", "jp") if f.name == "kanbun_mode")
        assert field.kind == "select"
        assert field.required is False
        assert {c["value"] for c in field.choices} == {"none", "kanbun", "kanbun_main"}

    def test_vertical_and_spread_and_preserve_tables_are_optional_checkboxes(self):
        fields = {f.name: f for f in get_extension_ui_fields("transcribe", "jp")}
        for name in ("vertical", "spread", "preserve_tables"):
            assert fields[name].kind == "checkbox"
            assert fields[name].required is False

    def test_passes_is_an_optional_text_field(self):
        field = next(f for f in get_extension_ui_fields("transcribe", "jp") if f.name == "passes")
        assert field.kind == "text"
        assert field.required is False

    def test_translate_action_gets_no_transcribe_fields_for_jp(self):
        # Regression coverage for the action_id-collision fix:
        # translation-ea's own "jp" registration (the Kanbun checkbox) is a
        # completely different, single-field registration — it must never
        # be returned when asking for transcribe's fields.
        names = {f.name for f in get_extension_ui_fields("translate", "jp")}
        assert names.isdisjoint({"vertical", "spread", "passes", "kanbun_mode", "preserve_tables"})


class TestApplyEaTranscribeUiHook:

    def test_kanbun_mode_kanbun_sets_kanbun_attribute(self):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("transcribe", "jp", sandbox, {"kanbun_mode": "kanbun"})
        assert sandbox.image_processor_service.kanbun is True
        assert sandbox.image_processor_service.kanbun_main is False

    def test_kanbun_mode_kanbun_main_sets_kanbun_main_attribute(self):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("transcribe", "jp", sandbox, {"kanbun_mode": "kanbun_main"})
        assert sandbox.image_processor_service.kanbun_main is True
        assert sandbox.image_processor_service.kanbun is False

    def test_kanbun_mode_none_sets_neither(self):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("transcribe", "jp", sandbox, {"kanbun_mode": "none"})
        assert sandbox.image_processor_service.kanbun is False
        assert sandbox.image_processor_service.kanbun_main is False

    def test_missing_kanbun_mode_sets_neither(self):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("transcribe", "jp", sandbox, {})
        assert sandbox.image_processor_service.kanbun is False
        assert sandbox.image_processor_service.kanbun_main is False

    @pytest.mark.parametrize("truthy", ["true", "1", "on", "yes", "TRUE"])
    def test_preserve_tables_truthy_spellings_set_tables_attribute(self, truthy):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("transcribe", "jp", sandbox, {"preserve_tables": truthy})
        assert sandbox.image_processor_service.tables is True

    def test_preserve_tables_unchecked_leaves_tables_false(self):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("transcribe", "jp", sandbox, {"preserve_tables": "false"})
        assert sandbox.image_processor_service.tables is False

    def test_vertical_spread_passes_are_not_touched_by_apply(self):
        # These are read directly by the base plugin's own run_ui_action,
        # not by this hook — apply() must leave the sandbox otherwise
        # untouched even when they're present in the submitted fields.
        sandbox = _make_sandbox()
        apply_extension_ui_hooks(
            "transcribe", "jp", sandbox, {"vertical": "true", "spread": "true", "passes": "3"},
        )
        assert not hasattr(sandbox.image_processor_service, "vertical")
        assert not hasattr(sandbox.image_processor_service, "spread")
        assert not hasattr(sandbox.image_processor_service, "passes")

    def test_apply_is_a_noop_for_english(self):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("transcribe", "en", sandbox, {"kanbun_mode": "kanbun", "preserve_tables": "true"})
        assert sandbox.image_processor_service.kanbun is False
        assert sandbox.image_processor_service.tables is False

    def test_apply_is_a_noop_for_the_translate_action_with_the_same_token(self):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("translate", "jp", sandbox, {"kanbun_mode": "kanbun", "preserve_tables": "true"})
        assert sandbox.image_processor_service.kanbun is False
        assert sandbox.image_processor_service.tables is False

    @pytest.mark.parametrize("token", ["zh", "jp", "kr"])
    def test_apply_works_for_every_registered_language(self, token):
        sandbox = _make_sandbox()
        apply_extension_ui_hooks("transcribe", token, sandbox, {"kanbun_mode": "kanbun"})
        assert sandbox.image_processor_service.kanbun is True
