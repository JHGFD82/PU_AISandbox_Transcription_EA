"""PU_AISandbox Transcription EA (East Asia) plugin.

Adds Chinese, Japanese, and Korean to the ``transcribe`` (image-to-text OCR)
and ``transcription_review`` (OCR error-checking) commands the base
transcription plugin already provides for English. It also adds several
flags only these languages need: vertical script direction, two-page
spreads, kanbun, and multiple OCR passes. Running several images in
parallel (``-w``/``--workers``) is *not* language-specific, so that flag
lives on the base transcription plugin instead — this plugin's own
``run()`` still passes ``workers`` through to ``process_image_folder()``,
just reading it from whatever value the base plugin's flag parsed rather
than registering a second copy of the same flag itself.

Kanbun (漢文) is Classical Chinese text embedded in older Japanese sources.
Readers add kundoku annotations — small marks like 返り点 (kaeriten,
reading-order marks) and 送り仮名 (okurigana, grammatical hints) — next to
the Chinese characters so the passage can be read in Japanese word order.
``--kanbun`` tells the model to transcribe every mark exactly as written;
``--kanbun-main`` tells it to transcribe only the large main-line characters
and skip the small annotations (useful when only the underlying Chinese
text is wanted).

Clone this repo into ``plugins/transcription-ea/`` inside the main
PU_AISandbox repo. It **requires** the base ``plugins/transcription/``
plugin to also be installed — this plugin borrows the base plugin's image
handling and settings machinery rather than keeping its own separate copy
of everything.

HOW THIS PLUGIN SHARES A COMMAND WITH THE BASE PLUGIN
-------------------------------------------------------
This plugin declares ``handles = ['Chinese', 'Japanese', 'Korean']`` and the
base plugin declares ``handles = ['English']``. Because both plugins
register the same two commands, the plugin loader does not treat that as a
conflict — it notices both plugins declare a ``handles`` list and combines
them into one ``DispatchPlugin`` per command. At startup, the DispatchPlugin
calls the base plugin's ``register_subparsers()`` once to build the shared
command-line flags, then calls this plugin's ``register_command_flags()`` to
add the extra East-Asia-only flags on top. When a professor actually runs a
command, the DispatchPlugin checks which language was requested and calls
``run()`` on whichever plugin's ``handles`` list contains it — so
``transcribe jp ...`` always reaches this plugin, and ``transcribe en ...``
always reaches the base plugin.

If the base plugin is ever missing, this plugin falls back to its own
``register_subparsers()`` and builds full standalone commands for all four
languages — an unsupported but gracefully handled situation, mainly useful
for testing this plugin on its own.

HOW THIS PLUGIN'S CODE STAYS IMPORTABLE, AND WHY LOAD ORDER MATTERS
-----------------------------------------------------------------------
This plugin's prompt-building and API-calling classes live in this plugin's
own folder rather than in the main repository's ``src/`` folder, since the
main repo no longer ships any East-Asia-specific service code at all. To
keep those files importable under the ``src.services.*`` path other code
expects — the same path the base plugin's own English-only classes use —
``_register()`` (called once, at import time, below) loads each file
directly and inserts it into Python's registry of already-imported modules
(``sys.modules``) under that shared name, so other code can write a normal
``import src.services.whatever`` statement without knowing the file
actually lives inside this plugin's folder. This is the same mechanism the
base transcription plugin uses for its own (English-only) copies of these
same module names.

That shared naming is exactly why load order matters here. Plugins load in
alphabetical folder order, so the base plugin (folder ``transcription``)
always finishes registering its modules before this plugin (folder
``transcription-ea``) gets a turn. Left unhandled, that would mean the base
plugin's stripped-down English-only classes silently "win" the shared
``sys.modules`` slot, and every East-Asia-only feature this plugin adds
(kanbun, vertical script, table hints, and Chinese/Japanese/Korean-specific
guidance) would be quietly ignored, with no error raised — flags like
``--kanbun`` would parse fine but have no effect on the actual OCR prompt
sent to the model. ``_register()``'s ``override`` parameter exists to
prevent exactly that: for the modules this plugin extends, it forces this
plugin's own copy to replace the base plugin's, regardless of which loaded
first. See ``_register()``'s docstring below for the full explanation of
which modules need ``override=True`` and which don't.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module registration (must run at import time) ────────────────────────────

_PLUGIN_DIR = Path(__file__).parent


def _register(module_name: str, rel_path: str, override: bool = False) -> None:
    """Load one of this plugin's own files and expose it under a shared ``src.*`` import path.

    The base ``plugins/transcription/`` plugin registers its own English-only
    versions of these same module names (e.g. ``src.services.image_processor_service``)
    using the identical mechanism. Because plugins load in alphabetical folder
    order, the base plugin's ``transcription`` folder is always loaded before
    this plugin's ``transcription-ea`` folder, so the base plugin's modules
    land in Python's module registry (``sys.modules``) first.

    For a module this plugin only *falls back to* (currently just plugin
    settings — see the registration call below), that's fine: skip
    re-registering if something is already there. But for the five modules
    below that this plugin *extends* with East-Asia-only capability (kanbun,
    vertical script, table preservation, and so on), the base plugin's
    stripped-down version must not be allowed to win — this plugin's version
    has to replace it, even though it loads second. ``override=True`` does
    that: it re-loads and overwrites the registry entry instead of leaving
    the base plugin's copy in place.

    Args:
        module_name: The dotted import path to register the module under
                     (e.g. ``'src.services.image_processor_service'``), matching
                     the path other code already imports it from.
        rel_path: The module's real file location, relative to this plugin's
                  own folder (e.g. ``'src/services/image_processor_service.py'``).
        override: If ``False`` (the default), do nothing when a module is
                  already registered under ``module_name`` — first writer wins.
                  If ``True``, always (re-)load this plugin's own copy and
                  replace whatever is currently registered, even if the base
                  plugin got there first.
    """
    if module_name in sys.modules and not override:
        return
    path = _PLUGIN_DIR / rel_path
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]


# Register plugin settings first so service modules can import from src.settings.
# This plugin's own settings.py doesn't define DEFAULT_OCR_PASSES (its
# settings.toml doesn't override the OCR-passes default either), so it's
# meant to fall through to the base plugin's copy when both are installed —
# hence no override=True here, unlike the five registrations below.
_register(
    "pu_plugin.transcription.settings",
    "src/settings.py",
)

# Register in dependency order: fragments → specs → services. Each of these
# gives East-Asia-specific behavior (kanbun, vertical script, table hints,
# script-specific guidance for Chinese/Japanese/Korean) that the base
# plugin's own same-named modules don't have, so override=True makes sure
# this plugin's copy is the one every other plugin ends up using — see the
# override parameter's explanation in _register() above.
_register(
    "src.services.prompts.ocr_fragments",
    "src/services/prompts/ocr_fragments.py",
    override=True,
)
_register(
    "src.services.prompts.ocr",
    "src/services/prompts/ocr.py",
    override=True,
)
_register(
    "src.services.prompts.transcription_review",
    "src/services/prompts/transcription_review.py",
    override=True,
)
_register(
    "src.services.image_processor_service",
    "src/services/image_processor_service.py",
    override=True,
)
_register(
    "src.services.transcription_review_service",
    "src/services/transcription_review_service.py",
    override=True,
)

# ── Main-repo imports ─────────────────────────────────────────────────────────
# These are available because the main PU_AISandbox root is on sys.path
# when running from that repo's root directory.

from src.cli import add_common_flags, add_notes_flags           # noqa: E402
from src.config import parse_single_language_code, register_language  # noqa: E402
from src.errors import CLIError                                    # noqa: E402
from src.runtime.ui_action import UiField, register_extension_ui_hooks  # noqa: E402
from src.services.constants import DEFAULT_PARALLEL_WORKERS       # noqa: E402
from src.settings import DEFAULT_OCR_PASSES                       # noqa: E402

# Register languages supported by this plugin into the shared registry.
# Transcription is independent of translation plugins; registering here
# ensures these codes are always available when only transcription is installed.
register_language('en', 'English')
register_language('zh', 'Chinese')
register_language('jp', 'Japanese')
register_language('kr', 'Korean')


# ── Web UI composer integration ────────────────────────────────────────────────
# Contributes this plugin's East-Asia-only transcribe options to the base
# transcription plugin's composer job modal — shown as a subsection once a
# professor picks Chinese, Japanese, or Korean as the language in the image,
# the same trigger point transcription/plugin.py's run_ui_action/
# preview_ui_action already use for get_extension_ui_fields()/
# apply_extension_ui_hooks() (see src/runtime/ui_action.py's
# ExtensionUiHooks, and translation-ea/plugin.py's Kanbun registration for
# the sibling mechanism this mirrors on the translate side).
#
# Two different shapes, matching how the base plugin's run_ui_action reads
# them (see the comment there):
#   - vertical / spread / passes: read directly and generically by the base
#     plugin's own run_ui_action, because they're real keyword arguments
#     process_image/process_image_folder already accept — not settings this
#     plugin can just toggle as an attribute on the sandbox afterward.
#     Registering them here only controls when the composer *shows* them
#     (Chinese/Japanese/Korean only); the values still flow through the
#     base plugin's own field-parsing, exactly like ``workers`` does.
#   - kanbun_mode / preserve_tables: applied through this plugin's own
#     ``_apply_ea_transcribe_ui_hook`` below, since these DO map onto plain
#     attributes on ``sandbox.image_processor_service`` (``.kanbun``,
#     ``.kanbun_main``, ``.tables``) that this plugin's overridden service
#     class (see the ``_register(..., override=True)`` calls above) already
#     reads at call time.
#
# A single "kanbun_mode" select (none/kanbun/kanbun_main) is used instead of
# two separate checkboxes so the composer can't submit both at once — the
# same mutual exclusivity the CLI's own ``--kanbun``/``--kanbun-main``
# argparse group enforces.

_EA_TRANSCRIBE_FIELDS = [
    UiField(
        name="vertical", label="Vertical text (top-to-bottom, right-to-left columns)",
        kind="checkbox", required=False, group="East Asia options",
    ),
    UiField(
        name="spread", label="Two-page spread (two facing pages scanned together)",
        kind="checkbox", required=False, group="East Asia options",
    ),
    UiField(
        name="passes", label="Number of OCR passes (default 1)",
        kind="text", required=False, group="East Asia options",
    ),
    UiField(
        name="kanbun_mode", label="Kanbun (漢文) handling", kind="select", required=False,
        choices=[
            {"value": "none", "label": "Not kanbun (default)"},
            {"value": "kanbun", "label": "Kanbun — preserve all kundoku annotations"},
            {"value": "kanbun_main", "label": "Kanbun — main characters only, omit annotations"},
        ],
        group="East Asia options",
    ),
    UiField(
        name="preserve_tables", label="Preserve tables as Markdown",
        kind="checkbox", required=False, group="East Asia options",
    ),
]


def _apply_ea_transcribe_ui_hook(sandbox, fields: dict) -> None:
    """Apply this plugin's composer-only settings (kanbun mode, table preservation) to the job's image processor.

    Called by the base plugin's ``run_ui_action``/``preview_ui_action`` via
    ``apply_extension_ui_hooks`` for every submitted or previewed transcribe
    job where Chinese, Japanese, or Korean is the selected language. A
    no-op for any field left at its default — matching how the equivalent
    CLI flags simply aren't passed when a professor doesn't need them.

    ``vertical``/``spread``/``passes`` are deliberately NOT handled here —
    see this module's "Web UI composer integration" section above for why
    those are read directly by the base plugin's own ``run_ui_action``
    instead.

    Args:
        sandbox: The already-constructed ``SandboxProcessor`` for this job.
        fields: The full submitted/previewed composer fields dict; only
                this hook's own ``"kanbun_mode"``/``"preserve_tables"`` keys
                are read.
    """
    kanbun_mode = str(fields.get("kanbun_mode", "")).strip().lower()
    if kanbun_mode == "kanbun":
        sandbox.image_processor_service.kanbun = True
    elif kanbun_mode == "kanbun_main":
        sandbox.image_processor_service.kanbun_main = True

    preserve_tables = str(fields.get("preserve_tables", "")).strip().lower() in ("true", "1", "on", "yes")
    if preserve_tables:
        sandbox.image_processor_service.tables = True


for _token in ("zh", "jp", "kr"):
    register_extension_ui_hooks(
        action_id="transcribe",
        token=_token,
        fields=_EA_TRANSCRIBE_FIELDS,
        apply=_apply_ea_transcribe_ui_hook,
    )


# ── Shared execution helper ────────────────────────────────────────────────────

def _run_transcription_review(
    sandbox: "SandboxProcessor",  # noqa: F821 — imported lazily in run(), only for type hints
    text: str,
    language: str,
    kanbun: bool = False,
    kanbun_main: bool = False,
    output_file: Optional[str] = None,
) -> None:
    """Check a transcription for likely OCR mistakes and print (and optionally save) the result.

    This mirrors the base transcription plugin's own
    ``_run_transcription_review()`` helper (see
    ``plugins/transcription/plugin.py``), extended with the ``kanbun``/
    ``kanbun_main`` settings this plugin's languages need.

    Args:
        sandbox: The active ``SandboxProcessor`` for this run, which owns
                 the API key, model, and the transcription-review service
                 that actually calls the AI model.
        text: The transcription text to check for errors.
        language: The language the transcription is written in (e.g.
                  ``'Japanese'``).
        kanbun: Whether the text contains kanbun with kundoku annotations
                that should be treated as intentional rather than flagged
                as errors.
        kanbun_main: Whether the transcription was produced in
                     main-character-only mode, so the model shouldn't flag
                     missing annotations as errors.
        output_file: Where to save the review report as a text file, or
                     ``None`` to only print it to the screen.

    Raises:
        CLIError: If the AI model call fails.
    """
    from src.errors import CLIError
    from src.output.file_output import FileOutputHandler
    try:
        result_json = sandbox.transcription_review_service.review_transcription(
            text, language, kanbun=kanbun, kanbun_main=kanbun_main
        )
        print("\n" + result_json)
        if output_file:
            FileOutputHandler.save_to_text_file(result_json, output_file, label="Review")
    except Exception as e:
        logger.error(f"Error during transcription review: {e}", exc_info=True)
        raise CLIError(f"Error during transcription review: {e}") from e


# ── Plugin class ──────────────────────────────────────────────────────────────

class TranscriptionPlugin:
    """Adds Chinese, Japanese, and Korean support to OCR transcription and transcription review.

    Extends the base transcription plugin's ``transcribe`` and
    ``transcription_review`` commands (which handle English on their own)
    to also cover Chinese, Japanese, and Korean, plus flags those languages
    specifically need: vertical script direction, two-page spreads, kanbun
    (see the module docstring above), and multiple OCR passes. Parallel
    processing of a folder of images is handled by the base plugin's own
    ``-w``/``--workers`` flag, not registered again here — see the module
    docstring above. See the module docstring above for how this plugin
    combines with the base plugin at startup.
    """

    commands: list[str] = ["transcribe", "transcription_review"]
    # ``handles`` lists the full language names (as returned by
    # ``parse_single_language_code``) that this plugin services.  The plugin
    # loader uses this to merge with the base transcription plugin via
    # DispatchPlugin, routing each language to the correct plugin.
    handles: list[str] = ["Chinese", "Japanese", "Korean"]

    # ── Argument registration ─────────────────────────────────────────────────

    def register_command_flags(self, parser: argparse.ArgumentParser) -> None:
        """Add the East-Asia-only command-line flags to a command the base plugin already built.

        Called by ``DispatchPlugin`` once the base plugin has registered a
        subcommand (``transcribe`` or ``transcription_review``) and its
        shared flags (like ``-i``/``--input``, and — as of the base plugin
        gaining its own parallel-OCR support — ``-w``/``--workers``). This
        method figures out which of the two subcommands it was handed by
        reading the last word of the parser's program name, then adds the
        flags relevant to that command — e.g. ``transcribe`` gets
        ``--vertical``, ``--spread``, ``--kanbun``/``--kanbun-main``,
        ``--passes``, and ``--preserve-tables``, while
        ``transcription_review`` only gets the kanbun flags (the others
        don't apply to reviewing already-typed text). ``--workers`` isn't
        added here — it's not East-Asia-specific, so it belongs on the base
        plugin, not duplicated per language extension (this plugin used to
        add its own copy back when the base plugin didn't have one; keeping
        both would make ``argparse`` see the same option registered twice
        the moment this plugin is installed alongside the base plugin).

        Args:
            parser: The argparse subcommand parser the base plugin already
                    created (e.g. the parser for ``transcribe``), which this
                    method adds more flags onto in place.
        """
        command = parser.prog.rsplit(None, 1)[-1]
        if command == "transcribe":
            parser.add_argument(
                "-v", "--vertical", dest="vertical", action="store_true",
                help="Text is predominantly vertical (top-to-bottom, right-to-left columns)",
            )
            parser.add_argument(
                "--spread", dest="spread", action="store_true",
                help="Image is a two-page spread (two facing pages scanned together)",
            )
            ea_group = parser.add_argument_group("East Asia options")
            kanbun_group = ea_group.add_mutually_exclusive_group()
            kanbun_group.add_argument(
                "--kanbun", dest="kanbun", action="store_true",
                help="Image contains kanbun (漢文): preserve 返り点, 送り仮名, and other "
                     "kundoku annotations exactly as written",
            )
            kanbun_group.add_argument(
                "--kanbun-main", dest="kanbun_main", action="store_true",
                help="Image contains kanbun (漢文): transcribe ONLY the large main-line "
                     "kanji; omit okurigana, furigana, kaeriten, and other small annotations",
            )
            parser.add_argument(
                "-P", "--passes",
                dest="passes",
                type=int,
                default=DEFAULT_OCR_PASSES,
                metavar="N",
                help="Number of OCR passes (default: 1). "
                     "Passes > 1 send the image and prior transcription back to the "
                     "model for review and correction.",
            )
            parser.add_argument(
                "--preserve-tables", dest="preserve_tables", action="store_true",
                help="Hint to the model that tabular data should be returned as Markdown "
                     "tables; the output layer renders them as proper tables in PDF/DOCX "
                     "or ASCII in TXT.",
            )
        elif command == "transcription_review":
            review_ea_group = parser.add_argument_group("East Asia options")
            review_kanbun_group = review_ea_group.add_mutually_exclusive_group()
            review_kanbun_group.add_argument(
                "--kanbun", dest="kanbun", action="store_true",
                help="Text contains kanbun (漢文) with kundoku annotations (返り点, 送り仮名)",
            )
            review_kanbun_group.add_argument(
                "--kanbun-main", dest="kanbun_main", action="store_true",
                help="Transcription was produced in main-character-only mode "
                     "(okurigana, furigana, kaeriten were omitted intentionally — "
                     "do not flag their absence as errors)",
            )

    def register_subparsers(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        """Build full standalone ``transcribe``/``transcription_review`` commands for all four languages.

        This is a fallback path, only used when this plugin is running
        without the base transcription plugin installed alongside it — an
        unsupported but gracefully handled situation (see the module
        docstring above). In the normal setup, the base plugin builds these
        two commands via its own ``register_subparsers()``, and this plugin
        only adds its extra flags on top via ``register_command_flags()``
        above; this method never runs in that case.

        Args:
            subparsers: The shared subcommand registry passed in by the CLI
                        startup code, the same object every plugin's
                        commands get added to.
        """
        # ── transcribe ────────────────────────────────────────────────────────
        if "transcribe" not in subparsers.choices:
            tr = subparsers.add_parser("transcribe", help="Transcribe images using OCR")
            tr.add_argument(
                "language_code",
                type=parse_single_language_code,
                help=(
                    "Target language: en (English), zh (Chinese), "
                    "jp (Japanese), kr (Korean)"
                ),
            )
            tr.add_argument(
                "-i", "--input",
                dest="input_file",
                type=str,
                required=False,
                help="Input image file path, or a folder of images to process in order",
            )
            tr.add_argument("-v", "--vertical", dest="vertical", action="store_true",
                            help="Text is predominantly vertical (top-to-bottom, right-to-left columns)")
            tr.add_argument("--spread", dest="spread", action="store_true",
                            help="Image is a two-page spread (two facing pages scanned together)")
            ea_group = tr.add_argument_group("East Asia options")
            kanbun_group = ea_group.add_mutually_exclusive_group()
            kanbun_group.add_argument(
                "--kanbun", dest="kanbun", action="store_true",
                help="Image contains kanbun (漢文): preserve 返り点, 送り仮名, and other "
                     "kundoku annotations exactly as written",
            )
            kanbun_group.add_argument(
                "--kanbun-main", dest="kanbun_main", action="store_true",
                help="Image contains kanbun (漢文): transcribe ONLY the large main-line "
                     "kanji; omit okurigana, furigana, kaeriten, and other small annotations",
            )
            tr.add_argument(
                "-P", "--passes",
                dest="passes",
                type=int,
                default=DEFAULT_OCR_PASSES,
                metavar="N",
                help="Number of OCR passes (default: 1). "
                     "Passes > 1 send the image and prior transcription back to the "
                     "model for review and correction.",
            )
            tr.add_argument(
                "--preserve-tables", dest="preserve_tables", action="store_true",
                help="Hint to the model that tabular data should be returned as Markdown "
                     "tables; the output layer renders them as proper tables in PDF/DOCX "
                     "or ASCII in TXT.",
            )
            tr.add_argument(
                "-w", "--workers",
                dest="workers",
                type=int,
                default=DEFAULT_PARALLEL_WORKERS,
                metavar="N",
                help=(
                    "Number of parallel OCR workers when processing a folder of images "
                    "(default: %(default)s). Ignored for single-image input. Multi-pass "
                    "OCR within each image always runs sequentially."
                ),
            )
            add_common_flags(tr)
            add_notes_flags(tr)

        # ── transcription_review ──────────────────────────────────────────────
        if "transcription_review" not in subparsers.choices:
            rv = subparsers.add_parser(
                "transcription_review",
                help="Review AI transcription output for OCR errors (returns JSON report)",
            )
            rv.add_argument(
                "language_code",
                type=parse_single_language_code,
                help=(
                    "Language of the transcription: en (English), zh (Chinese), "
                    "jp (Japanese), kr (Korean)"
                ),
            )
            review_input_group = rv.add_mutually_exclusive_group(required=False)
            review_input_group.add_argument(
                "-i", "--input",
                dest="input_file",
                type=str,
                help="Path to a text file containing the transcription result to review",
            )
            review_input_group.add_argument(
                "-c", "--custom",
                dest="custom_text",
                action="store_true",
                help="Paste the transcription text interactively (end with --- on its own line)",
            )
            review_ea_group = rv.add_argument_group("East Asia options")
            review_kanbun_group = review_ea_group.add_mutually_exclusive_group()
            review_kanbun_group.add_argument(
                "--kanbun", dest="kanbun", action="store_true",
                help="Text contains kanbun (漢文) with kundoku annotations (返り点, 送り仮名)",
            )
            review_kanbun_group.add_argument(
                "--kanbun-main", dest="kanbun_main", action="store_true",
                help="Transcription was produced in main-character-only mode "
                     "(okurigana, furigana, kaeriten were omitted intentionally — "
                     "do not flag their absence as errors)",
            )
            add_common_flags(rv)
            add_notes_flags(rv)

    # ── Command execution ─────────────────────────────────────────────────────

    def run(
        self,
        args: argparse.Namespace,
        professor: str,
        model: Optional[str],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
    ) -> None:
        """Run the ``transcribe`` or ``transcription_review`` command for Chinese, Japanese, or Korean.

        Builds a ``SandboxProcessor`` (which resolves the professor's API
        key, sets up token/cost tracking, and lazily creates whichever
        services are needed), then branches on ``args.command`` to run the
        requested command: transcribing an image or folder of images with
        this plugin's East-Asia-specific options applied (vertical script,
        two-page spreads, kanbun, multiple OCR passes, parallel workers), or
        reviewing a previously-produced transcription for likely OCR errors.
        The base transcription plugin's ``process_image``/``process_image_folder``
        methods (attached to ``SandboxProcessor`` for every installed plugin
        to share) are reused here rather than duplicated.

        Args:
            args: The object holding all the parsed command-line flags for
                  this run (which command was invoked, the input file path,
                  whether ``--kanbun`` or ``--dry-run`` was passed, etc.).
            professor: The Princeton NetID whose configuration and API key
                       should be used for this run (e.g. ``'heller'``).
            model: The AI model explicitly requested on the command line, or
                   ``None`` to use this plugin's configured default.
            temperature: The requested sampling temperature (controls how
                         predictable vs. varied the model's wording is), or
                         ``None`` to use the default.
            top_p: The requested nucleus-sampling value (an alternative way
                   of controlling response variety), or ``None`` to use the
                   default.
            max_tokens: The requested maximum response length, in tokens
                        (the small chunks of text models process and bill
                        by), or ``None`` to use the default.

        Raises:
            CLIError: If a required input is missing, the wrong file type
                is supplied, ``--passes`` is less than 1, or the AI model
                call fails.
        """
        import os
        from src.runtime.sandbox_processor import SandboxProcessor

        sandbox = SandboxProcessor(
            professor,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        if args.command == "transcribe":
            # language_code is already resolved to a full name by parse_single_language_code
            target_language: str = args.language_code

            if getattr(args, 'notes', False):
                _vertical_flag = getattr(args, 'vertical', False)
                _preview_sys, _preview_usr = sandbox.image_processor_service.build_prompts(
                    target_language, vertical=_vertical_flag
                )
                sys_note, usr_note = sandbox._collect_notes(_preview_sys, _preview_usr)
                sandbox.image_processor_service.system_note = sys_note
                sandbox.image_processor_service.user_note = usr_note

            sandbox._apply_inline_notes(sandbox.image_processor_service, args)

            if getattr(args, 'kanbun', False):
                sandbox.image_processor_service.kanbun = True

            if getattr(args, 'kanbun_main', False):
                sandbox.image_processor_service.kanbun_main = True

            if getattr(args, 'preserve_tables', False):
                sandbox.image_processor_service.tables = True

            if getattr(args, 'dry_run', False):
                vertical_dr = getattr(args, 'vertical', False)
                spread_dr = getattr(args, 'spread', False)
                passes_dr = getattr(args, 'passes', 1)
                model_dr = sandbox.image_processor_service._get_model()
                sys_p, usr_p = sandbox.image_processor_service.build_prompts(target_language, vertical=vertical_dr, spread=spread_dr)
                note = "Image content would be base64-encoded and attached to the user message"
                if passes_dr > 1:
                    note += f"; {passes_dr} OCR passes would run sequentially"
                sandbox._dry_run_display(model_dr, sys_p, usr_p, note=note, **sandbox._sampling_kwargs(args))
                return

            if not args.input_file:
                raise CLIError("Input file is required for transcribe command. Use -i option.")

            input_path = os.path.abspath(args.input_file)
            output_file = sandbox._resolve_output_path(args)
            vertical = getattr(args, 'vertical', False)
            spread = getattr(args, 'spread', False)
            passes = getattr(args, 'passes', 1)
            workers = getattr(args, 'workers', 1)
            if passes < 1:
                raise CLIError("--passes must be at least 1.")

            if os.path.isdir(input_path):
                sandbox.process_image_folder(input_path, target_language, output_file, vertical=vertical, spread=spread, passes=passes, workers=workers)
            else:
                file_type = sandbox._detect_and_validate_file(input_path)
                if file_type != 'image':
                    raise CLIError(f"Transcribe command requires an image file or folder, but got {file_type}.")
                sandbox.process_image(input_path, target_language, output_file, vertical=vertical, spread=spread, passes=passes)

        else:  # transcription_review
            language: str = args.language_code  # already resolved by parse_single_language_code
            kanbun = getattr(args, 'kanbun', False)
            kanbun_main = getattr(args, 'kanbun_main', False)

            if getattr(args, 'notes', False):
                _preview_sys, _preview_usr = sandbox.transcription_review_service.build_prompts(language, kanbun=kanbun, kanbun_main=kanbun_main)
                sys_note, usr_note = sandbox._collect_notes(_preview_sys, _preview_usr)
                sandbox.transcription_review_service.system_note = sys_note
                sandbox.transcription_review_service.user_note = usr_note

            sandbox._apply_inline_notes(sandbox.transcription_review_service, args)

            if getattr(args, 'dry_run', False):
                model_dr = sandbox.transcription_review_service._get_model()
                sys_p, usr_p = sandbox.transcription_review_service.build_prompts(language, kanbun=kanbun, kanbun_main=kanbun_main)
                sandbox._dry_run_display(
                    model_dr, sys_p, usr_p,
                    note="Transcription text would be appended to the user prompt at runtime",
                    **sandbox._sampling_kwargs(args),
                )
                return

            if args.input_file:
                input_path = os.path.abspath(args.input_file)
                if not os.path.exists(input_path):
                    raise CLIError(f"Input file '{input_path}' not found.")
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if not text.strip():
                    raise CLIError(f"Input file '{input_path}' is empty.")
            elif args.custom_text:
                text = sandbox._collect_multiline("Paste the transcription result to review")
                if not text.strip():
                    raise CLIError("No transcription text provided.")
            else:
                raise CLIError(
                    "No input supplied.\n"
                    "  transcription_review expects the text output of a prior transcription, "
                    "not the original document or image.\n"
                    "  Use -i <file.txt> to supply a saved transcription file, "
                    "or -c to paste the text interactively."
                )

            output_file_r = sandbox._resolve_output_path(args)
            _run_transcription_review(
                sandbox, text, language, kanbun=kanbun, kanbun_main=kanbun_main, output_file=output_file_r
            )


plugin = TranscriptionPlugin()
