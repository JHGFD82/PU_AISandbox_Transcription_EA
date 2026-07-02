# Transcription EA Plugin — AI Coding Assistant Instructions

## Plugin Overview
This is the East-Asia extension for the `transcribe` and `transcription_review` commands in [PU AI Sandbox](https://github.com/princeton-oit/PU_AISandbox) — it adds Chinese, Japanese, and Korean (plus kanbun, vertical script, table preservation, multi-pass refinement, and parallel folder processing) on top of the base plugin's English-only support.

This repo lives at `plugins/transcription-ea/` inside the main PU_AISandbox repo, and **requires** `plugins/transcription/` (the base plugin) to also be installed — see "Relationship to the Base Plugin and Main Repo" below. All `src.*` imports (e.g. `src.cli`, `src.runtime`, `src.processors`) resolve against the main repo's `src/` — they are *not* in this plugin's directory.

---

## Repository Layout

```text
plugin.py                        ModePlugin entry point; also handles sys.modules injection
settings.toml                    Default model parameters (temperature, top_p, max_tokens, frequency/presence penalty)

conftest.py                      Inserts main repo root into sys.path for pytest
pytest.ini                       testpaths=tests, pythonpath=../..
src/
  settings.py                    Loads settings.toml; exposes OCR_* and TRANSCRIPTION_REVIEW_* constants
  services/
    image_processor_service.py   ImageProcessorService — single-image OCR, multi-pass refinement, folder processing
    transcription_review_service.py  TranscriptionReviewService — reviews AI OCR output; returns structured JSON report
    prompts/
      ocr_fragments.py           All raw prompt strings (source of truth); no logic, str.format() placeholders
      ocr.py                     OcrPromptSpec dataclass — assembles system + user + refinement prompts for OCR
      transcription_review.py    TranscriptionReviewPromptSpec dataclass — assembles prompts for review
tests/
  test_transcription_cli.py      CLI flag parsing and validation tests
```

---

## Architecture: sys.modules Injection

`plugin.py` calls `_register()` at import time to inject each `src/services/*` module into `sys.modules` under its canonical `src.services.*` name. This makes the plugin's local copies available to the main repo's runtime without duplicating the import paths. The base plugin (`plugins/transcription/`) registers its own copies under the **same** module names.

**Injection order matters** — always register in dependency order:
1. `pu_plugin.transcription.settings` (settings.py)
2. `src.services.prompts.ocr_fragments`
3. `src.services.prompts.ocr`
4. `src.services.prompts.transcription_review`
5. `src.services.image_processor_service`
6. `src.services.transcription_review_service`

**Load order and `override`**: plugins load in alphabetical folder order, so `transcription` (base) always finishes registering before `transcription-ea` gets a turn. For modules 2–6 above, `_register()` is called with `override=True`, which forces this plugin's copy to replace whatever the base plugin already registered — required, because those five modules are where this plugin's actual EA behavior (kanbun, vertical, tables, script guidance) lives, and the base plugin's own copies of the same module names are stripped-down English-only versions that silently discard those settings. **Do not remove `override=True` from those five calls** — without it, every EA-specific flag (`--kanbun`, `--vertical`, `--spread`, `--preserve-tables`) silently becomes a no-op whenever the base plugin is also installed (the normal case), with no error raised. This was a real, previously-shipped bug; see git history for the fix.

Registration #1 (`pu_plugin.transcription.settings`) is the one exception — it intentionally stays `override=False` (skip-if-already-registered). This plugin's own `settings.py`/`settings.toml` don't define `DEFAULT_OCR_PASSES`, so it's meant to fall through to the base plugin's copy, which does. If you add a new setting this plugin needs to override, add it to `src/settings.py` here first, then reconsider whether this registration also needs `override=True`.

---

## Prompt Architecture

### Fragments (`ocr_fragments.py`)
Single source of truth for all prompt text. Contains only constants and dicts — no logic. All variable parts use `str.format()` with named placeholders like `{target}`, `{language}`, `{text}`.

Key constants:
- `OCR_SYSTEM_BASE` — system role and task description; placeholder `{target}`
- `OCR_RULES` — rule block appended to the OCR system prompt
- `OCR_USER_BASE` — base user message; placeholder `{target}`
- `OCR_USER_RULES` — critical rules block in the OCR user message
- `OCR_REFINEMENT_BASE` — user message for refinement passes (pass 2+)
- `OCR_VERTICAL_BLOCK`, `OCR_VERTICAL_REINFORCEMENT` — vertical-text orientation notes
- `OCR_SPREAD_NOTE` — two-page spread layout note
- `KANBUN_SCRIPT_NOTE`, `KANBUN_MAIN_SCRIPT_NOTE` — script-guidance notes for kanbun modes
- `KANBUN_OCR_NOTE`, `KANBUN_MAIN_OCR_NOTE` — detailed kanbun annotation instructions
- `OCR_SCRIPT_GUIDANCE` — `{language: note}` dict for per-language script hints
- `TRANSCRIPTION_REVIEW_ROLE`, `TRANSCRIPTION_REVIEW_APPROACH`, `TRANSCRIPTION_REVIEW_SCHEMA`, `TRANSCRIPTION_REVIEW_RULES` — review pipeline fragments
- `TRANSCRIPTION_REVIEW_KANBUN_NOTE`, `TRANSCRIPTION_REVIEW_KANBUN_MAIN_NOTE` — kanbun-specific review guidance

### Prompt Specs (`ocr.py`, `transcription_review.py`)
`OcrPromptSpec` and `TranscriptionReviewPromptSpec` are `@dataclass` classes. They accept flags matching the CLI options and expose `system_prompt()` / `user_prompt()` (and `refinement_prompt()` for OCR) methods that assemble the final strings from fragments.

**When adding a new flag that affects prompts**: add a field to the relevant spec, add the fragment to `ocr_fragments.py`, and wire it in the spec's `system_prompt()` / `user_prompt()`.

---

## Settings

`src/settings.py` walks up from its own path to find the nearest `settings.toml` containing `[ocr]` or `[transcription_review]`. This means the user can edit either `plugins/transcription/settings.toml` or the main repo's root `settings.toml` — whichever is closer wins.

Exposed constants (all have fallback defaults in code):
- `OCR_TEMPERATURE`, `OCR_TOP_P`, `OCR_MAX_TOKENS`, `OCR_FREQUENCY_PENALTY`, `OCR_PRESENCE_PENALTY`
- `TRANSCRIPTION_REVIEW_TEMPERATURE`, `TRANSCRIPTION_REVIEW_TOP_P`, `TRANSCRIPTION_REVIEW_MAX_TOKENS`

---

## Services

### `ImageProcessorService`
- Accepts an image file path (single image) or folder of images.
- Single-image OCR: calls the vision API once (pass 1), then optionally runs refinement passes (2+) by sending the image and prior transcription back to the model.
- Folder mode: processes images in sorted order; `workers > 1` enables `ThreadPoolExecutor` parallel mode.
- `self.kanbun`, `self.kanbun_main`, `self.tables` are set by `plugin.py` before calling the service.
- `_get_model()` resolves to the catalog's `ocr` default unless overridden; rejects non-vision models.

### `TranscriptionReviewService`
- Accepts raw text (from a file or pasted input).
- Returns a structured JSON report: overall quality assessment, `global_replacements` for systematic errors, and per-line `corrections` for context-specific errors.
- `_inject_model_and_validate()` strips markdown fences and injects the actual model name into `meta.model` before returning the response.
- `plugin.py`'s `run()` calls the module-level `_run_transcription_review()` helper (defined in `plugin.py`, mirroring the base plugin's own helper of the same name) rather than a `SandboxProcessor` method, passing `kanbun`/`kanbun_main` through to `review_transcription()`. See `tests/test_transcription_review.py` for its test coverage.

---

## `plugin.py` — ModePlugin Contract

`TranscriptionPlugin` satisfies the main repo's extension-plugin interface (it declares `handles`, not just `commands`):
- `commands = ["transcribe", "transcription_review"]`
- `handles = ["Chinese", "Japanese", "Korean"]`
- `register_command_flags(parser)` — the normal path: adds EA-only flags to a subparser the base plugin already created, via `DispatchPlugin`
- `register_subparsers(subparsers)` — fallback path only, used if the base plugin is absent; builds full standalone subparsers for all four languages
- `run(args, professor, model, temperature, top_p, max_tokens)` — validates flags, wires services, delegates to `SandboxProcessor`

**Flag validation in `run()`**:
- `transcribe`: requires `-i` (no default); `--passes` must be ≥ 1; folder input uses workers, single-image ignores workers
- Both commands: `--kanbun` and `--kanbun-main` are a mutually exclusive group (enforced by argparse)

Always add new flag validation in `run()`, not in `register_subparsers`.

---

## Testing

Tests run from this plugin's directory using the main repo's venv:

```bash
cd plugins/transcription-ea
pytest                              # all tests
pytest -v                           # verbose
pytest -k "kanbun"                  # filter by keyword
```

`pytest.ini` sets `pythonpath = ../..` (main repo root) and `testpaths = tests`.  
`conftest.py` inserts the main repo root into `sys.path` at collection time for compatibility.

`tests/test_transcription_cli.py` uses `create_argument_parser(load_plugins(_PLUGINS_DIR))` where `_PLUGINS_DIR = Path(__file__).resolve().parents[2]` — two levels up from `tests/` points to `plugins/`, which is the correct plugins root.

---

## Common Patterns

- **Adding a new transcribe flag**: add it in `register_subparsers`, validate in `run()`, add a field to `OcrPromptSpec`, add the fragment to `ocr_fragments.py`.
- **Adding a new language-specific script note**: add an entry to `OCR_SCRIPT_GUIDANCE` in `ocr_fragments.py` keyed by the exact language name string (e.g. `"Vietnamese"`); it is picked up automatically by `OcrPromptSpec._script_note()`.
- **Adding a new kanbun mode**: add the fragment constants to `ocr_fragments.py`, add the flag to the kanbun mutually exclusive group in `register_subparsers`, add a field to `OcrPromptSpec`, wire in `_script_note()` and `system_prompt()` / `user_prompt()`.
- **Never** import from `pu_plugin.transcription.settings` outside `src/` — use the constants exported from `src.settings` (which resolves to whichever copy is active via sys.modules).

---

## Relationship to the Base Plugin and Main Repo

This plugin imports the following from the main repo at runtime (not available in this repo alone):
- `src.cli`: `add_common_flags`, `add_notes_flags`
- `src.config`: `parse_single_language_code`, `register_language`
- `src.errors`: `CLIError`
- `src.services.constants`: `DEFAULT_PARALLEL_WORKERS`
- `src.settings`: `DEFAULT_OCR_PASSES` (actually resolves to the base plugin's `settings.py` — see "Architecture: sys.modules Injection" above)
- `src.runtime.sandbox_processor`: `SandboxProcessor`

These imports are at module level in `plugin.py` and will fail if the plugin is loaded outside the main repo context. This is expected and by design.

This plugin also depends on the **base transcription plugin** (`plugins/transcription/`) being installed for two things it does *not* provide itself:
- `SandboxProcessor.process_image` / `process_image_folder` — Mixin methods registered by the base plugin under `src.runtime.image_handler` (see `_discover_plugin_mixins()` in `src/runtime/sandbox_processor.py`). This plugin's `run()` calls these directly; it has no image-handling Mixin of its own.
- The `transcribe`/`transcription_review` subcommands themselves, via the `DispatchPlugin` merge described above — this plugin's `register_subparsers()` is a fallback, not the normal path.

---

## Git Commit Format

Follow the same convention as the main repo:

```
<type>(<scope>): <short summary>   ← imperative mood, ≤ 72 chars

Why:
- <reason>

What changed:
- <change 1>
- <change 2>

Notes:
- <migration/compatibility details if any>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`, `build`  
Scope examples: `plugin`, `prompts`, `settings`, `services`, `tests`, `docs`

**Never run `git commit` or `git add`. The user handles all commits.**
