# Football buildup prediction and attention diagnostics

Code for predicting provider xG from the shooting team's recorded events before a
shot, conditional on that shot having occurred. The final shot is retained only
as a target/context record and excluded from pre-shot model inputs.

This repository contains code, fixed hyperparameters and synthetic tests only.
It does **not** contain event data, prepared datasets, trained weights, results,
paper files or figures. StatsBomb access is obtained separately under the
appropriate licence; do not commit downloaded data or credentials.

## Installation

Use Python 3.11 or newer. The package has been tested with Python 3.12.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip install -e '.[test,hpo]'
python -m pytest
```

On Windows, activate with `.venv\Scripts\activate`. For compatible newer
dependency versions, omit the requirements file. Installed versions are recorded
in each local run; hardware and library changes can affect numerical results.

## Local data

Use a separate cache for each season. Expected files are:

```text
local/cache/development/       Historical season
  matches_c<ID>_s<ID>.parquet  Match metadata, including match_id and match_date
  events_m<ID>.parquet        StatsBomb event tables for every historical match
  lineups_m<ID>.parquet       Optional; needed for player-minutes summaries
local/cache/holdout/           Later season, same format
```

There must be exactly one matches file in each cache. For holdout, only matches
with cached events are included; keep that selection fixed when comparing runs.
The cache paths are stored locally and must remain available for player summaries.

An optional authenticated downloader is included:

```sh
python -m pip install -e '.[download]'
python -m football_attention.download --help
python -m football_attention.download --competition-id 38 --auto-season \
  --expected-season 2025/2026 --cache-dir local/cache/development
python -m football_attention.download --competition-id 38 --auto-season \
  --expected-season 2026/2027 --match-week-end 6 --available-only \
  --cache-dir local/cache/holdout
```

It prompts for credentials (password input is hidden) or reads `SB_USERNAME` and
`SB_PASSWORD`. Never put credentials in code, command-line arguments or Git.
Successful downloads are cached and reused. Choose an explicit date/round cutoff;
a later download may contain more matches than the original experiment.

## Run

From the repository root:

```sh
football-attention prepare --work-dir local/run \
  --development-cache local/cache/development \
  --holdout-cache local/cache/holdout

football-attention run --work-dir local/run --device mps
```

Use `--device cpu`, `cuda`, or `auto` on other machines. No download is started
by the analysis command. `run --dry-run` prints the planned commands without
training. To run selected stages, use e.g.

```sh
football-attention run --work-dir local/run --device mps --stages refit holdout
football-attention run --work-dir local/run --device mps \
  --stages alignment explanations recency
```

Stages run in dependency order:

| Stage | Purpose |
| --- | --- |
| `refit` | Fit full-buildup and final-event-only classical models plus five neural configurations on train + validation; evaluate development test |
| `holdout` | Evaluate the same frozen checkpoints on the next season |
| `alignment` | Fit the schedule-matched zero-alignment control; compare attention models in both periods |
| `explanations` | Add permutation Shapley, gradient × input, recency and deletion controls |
| `recency` | Aggregate paired attention-minus-recency diagnostics |
| `leakage` | Compare pre-shot, shot-only and full inputs with fixed classical baselines |
| `order` | Train ordered/shuffled GRUs and evaluate the two-by-two order design |
| `architecture` | Compare aligned-GRU and Transformer player rankings |
| `summaries` | Summarize target distribution, prediction ranges and calibration |

`holdout` requires `refit`; `alignment` and `explanations` require both;
`recency` requires `explanations`; `architecture` requires `refit`; `summaries`
requires `refit` and `holdout`.
`leakage` and `order` are separate development controls, not refits of the tuned models.
All stages are enabled by default. Full training and attribution evaluation are
substantial computations; the tests use only small, generated synthetic examples.

The frozen configuration already contains the selected boosting parameters. To
repeat their 100-trial Optuna searches on the training and validation partitions:

```sh
football-attention tune-boosting --work-dir local/run --trials 100
```

This writes resumable studies and best-parameter summaries under
`local/run/results/boosting_hpo/`; it never reads the development test or holdout.

Existing checkpoints and completed evaluations are reused. Input and source-code hashes,
configuration, device and library versions are checked before resuming. Use a new
work directory if these change. A lock prevents concurrent runs in the same work
directory. After an interrupted process, remove `.run.lock` only after confirming
that no process is still running. Load only checkpoints you created or trust.

## Method and configuration

- `configs/frozen.json` contains the selected hyperparameters, refit epochs and
  five random seeds. No Optuna database, trial history or evaluation results are needed.
- Preparation builds sequences of at most 20 recorded events including the reference
  shot. It uses the shooting team's events from the same possession, resolves
  event-specific outcomes and removes sequences without any visible pre-shot event
  (`n_events > 1`). IDs and temporal split boundaries are preserved.
- Development matches are split approximately 70/15/15 by date, keeping each date
  together. Holdout dates must follow development and match IDs cannot overlap.
- Vocabularies and aggregate-feature columns are fitted on the applicable training
  partition only. Actual feature names are exported locally to
  `results/final/classical/feature_names.json`. The final-event-only boosting
  baseline receives only the last pre-shot event's type, outcome, normalized start
  and end coordinates, and elapsed time since the preceding event.
- Full-buildup and final-event-only boosting use the same Optuna search space,
  100 trials, validation MSE objective and training/refit protocol. Both selected
  configurations are recorded in `configs/frozen.json`.
- The retained label `faithful_no_player_id` denotes the Transformer;
  `gru_attention_faithful` denotes occlusion-aligned GRU attention. These are
  implementation labels, not claims of general explanation faithfulness.
- Diagnostics include occlusion, Integrated Gradients, gradient × input,
  antithetic permutation Shapley, recency, deletion and pooling-parameter
  randomization. Paired intervals use crossed seed and match resampling, preserving
  comparisons within sequences; the default is 10,000 bootstrap replicates.
- The order control uses a fixed sequence-specific permutation and its original,
  separate development configuration. It measures input-order dependence.

Local outputs are under `local/run/results/`: predictions, MSE/MAE/R² summaries,
bootstrap intervals, event attributions, player summaries and checkpoints.
Prepared-data reports record the retained and excluded sequence counts.
No output is uploaded automatically.

## Figures

After `refit`, `holdout` and `explanations` complete:

```sh
python -m football_attention.plots --work-dir local/run
python -m football_attention.plots --help
```

This writes prediction/deletion, buildup-length, grouped boosting-importance and
player-response plots under `results/figures/`.
Add `--sequence-id` for an event-level pitch diagram from the development test.
Add `--team` and `--players` for player heatmaps; IDs and names are in the local
`data/development/player_id_to_name.json`. `--player-labels` can supply names with
positions. Heatmaps default to train + validation; use `--splits test` for the
development test. They use inference only, never retraining. The maps retain the
current spatial smoothing and show attention relative to uniform, not player value.
`--highlight-player` highlights a selected player in the response plot. Labels are
placed automatically, so a new dataset may require small presentational adjustments.

## Repository contents

`src/football_attention/` contains preprocessing, models, training, tuning,
diagnostics and the stage runner. `tests/` contains synthetic tests; `configs/`
contains model settings. Figure generators are data-driven; manual label
coordinates from the paper are not embedded. Historical repair scripts, old
experiment launchers and paper-specific exports are intentionally excluded.

Only this repository directory should be published, not its parent research
workspace. Review `git status --short` before committing; `.gitignore` excludes
local data, results, weights, environments and credentials. No software licence
has been selected yet; the authors should choose one before public release.
