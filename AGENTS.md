# AGENTS.md

Instructions for AI coding agents (Claude Code, Cursor, etc.) working in this repository.
Human contributors: see `README.md` and `SETUP.md` instead — this file is agent-facing.

## Location
This file must live at the **repo root** (`dental-model/AGENTS.md`), alongside `README.md`
and `pyproject.toml`. Most agent tooling looks for `AGENTS.md` at the root by convention;
don't move it into a subfolder.

## Environment
- This project uses **`uv`**, never `venv`/`pip` directly. Run everything with `uv run ...`.
- Install/sync: `uv sync --all-extras --dev`
- Never call `pip install` directly — add dependencies with `uv add <package>` (or
  `uv add --dev <package>` for dev-only tools) so `pyproject.toml`/`uv.lock` stay in sync.

## Project shape (read before editing)
- `src/dental_model/` is the installable package; `notebooks/` is exploratory only —
  logic that needs to be reused or tested belongs in `src/`, not duplicated in a notebook.
- Two independent models exist: a YOLOv8 **detector** (`src/dental_model/detector/`) and a
  `timm` **classifier** (`src/dental_model/classifier/`). They are wired together in
  `src/dental_model/pipeline.py`. Don't merge them into one model without being asked —
  that's a deliberate architecture decision, see `dental-model-implementation-plan.md` §1.
- Dataset status (which of the 4 raw sources are actually used) is tracked in
  `configs/data_paths.yaml` under each source's `status` field
  (`required` / `optional` / `deferred_v2`). Check this before writing code that assumes
  a dataset is in scope.
- Hyperparameters live in `configs/*.yaml` — never hardcode them inside training scripts.

## Data
- Raw/interim/processed data under `data/` (lowercase) is gitignored. Never commit files
  from `data/raw/`, `data/interim/`, or `data/processed/` — if you find yourself about to
  `git add` something under `data/`, stop and check `.gitignore` instead.
- The folder is `data/`, lowercase, everywhere — not `Data/`. If you see a capitalized
  `Data/` anywhere in the tree, that's leftover drift from an earlier import; don't create
  new references to it.
- After changing `prepare_data.sh` or `unify_labels.py`, regenerate
  `data/processed/checksums.json` so the dataset snapshot stays traceable.

## Secrets
- `.env` is for local dev only and is gitignored — never commit it, never print its
  contents, never suggest committing `HF_TOKEN`/`WANDB_API_KEY` anywhere.
- The deployed Hugging Face Space does **not** read `.env`. Its secrets live in the
  Space's own Settings → Repository secrets. Don't conflate the two when editing
  deployment code or docs.

## Testing & linting
Before considering a change done, run:
```bash
uv run ruff check .
uv run pytest
```
Both must pass. Add or update a test in `tests/` for any change to `src/dental_model/data/`
or `src/dental_model/pipeline.py` — these are the two places most likely to break silently.

## CI/CD
- `.github/workflows/ci.yml` and `.github/workflows/deploy-space.yml` must stay under
  `.github/workflows/` — GitHub only discovers workflows at that exact path. Never move
  or duplicate them to the repo root.
- Don't edit `deploy-space.yml` to push on every push to every branch — it's scoped to
  `main` and to changes under `app/` on purpose, so the Space isn't redeployed on
  unrelated commits.

## Model/versioning conventions
- Model release tags: `v<major>.<minor>.<patch>-detector` / `-classifier`.
- Bump `pyproject.toml`'s `version` and add an entry to `CHANGELOG.md` for any change that
  affects a trained model's behavior (new data, new hyperparameters, new architecture) —
  not for pure refactors.

## What not to do
- Don't add `venv`, `requirements.txt` (root-level), or `pip freeze` output — this repo is
  `uv`-only. (`app/requirements.txt` is the one intentional exception: it's exported via
  `uv export` for the HF Spaces build, not maintained by hand.)
- Don't introduce a second, competing config system (e.g. argparse flags that duplicate
  `configs/*.yaml`) — extend the YAML configs instead.
- Don't claim a metric or eval result that wasn't actually produced by a logged run —
  `report/synopsis.md` and `app/model_card.md` should only contain real numbers from
  `models/*_runs/`, not placeholders left un-filled.
