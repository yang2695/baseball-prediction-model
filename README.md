# Baseball prediction model

[![Test and train baseball model](https://github.com/yang2695/baseball-prediction-model/actions/workflows/pipeline.yml/badge.svg)](https://github.com/yang2695/baseball-prediction-model/actions/workflows/pipeline.yml)

Predict the **home team's chance of winning an MLB game before first pitch**. This is an end-to-end Python baseball analytics project, with reproducible historical evaluation. It is not a betting system or a live 2026 forecast.

## Data and target

The pipeline fetches completed **2015–2025 regular-season games** from the public [MLB Stats API schedule endpoint](https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R). It normalizes game IDs, dates, MLB team IDs, scores, and a neutral-site indicator when supplied. It rejects an unexpectedly short historical season instead of silently scoring a partial year (2020 has a lower cutoff because the season was shortened). The downloaded records are cached in `data/raw/mlb_games.csv` and not committed.

**Target:** Whether the listed home team won. Unfinished games and ties are excluded. Each distinct MLB `gamePk` is used at most once.

The features are calculated strictly from each team's **earlier dates** in that season:

- Win percentage, runs scored and allowed, and run differential over the previous 10 games
- Season-to-date win percentage, run differential per game, and games played
- Days of rest, month, and a neutral-site flag

Season-opening priors are deliberately simple (50% win rate; 4.5 runs scored/allowed). Because the schedule does not guarantee usable first-pitch ordering for every historical game, **all games on one date get their pre-date snapshots**; a doubleheader result is never a feature for another game on that date.

## Reproduce

Python 3.11 or newer recommended, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m baseball_prediction.cli
```

Force a fresh download with `python -m baseball_prediction.cli --refresh`. Use `--data` to pick a different cache filename and `--output-dir` for a different output directory.

The run creates `artifacts/model.joblib` (fitted estimator), `artifacts/evaluation.json` (metrics and sample counts), `artifacts/test_predictions.csv` (each held-out prediction), and `artifacts/calibration.png`. Never load `.joblib` files from untrusted sources.

**No local setup required:** the [GitHub Actions workflow](https://github.com/yang2695/baseball-prediction-model/actions/workflows/pipeline.yml) installs dependencies, runs all tests, downloads the source, and trains the model on GitHub-hosted runners. It prints scores in the Actions log and uploads `baseball-model-results` (including the model and chart) as a workflow artifact.

## A fair time-based backtest

| Stage | Seasons | Purpose |
| --- | --- | --- |
| Initial training | 2015–2023 | Fit candidate models |
| Validation | 2024 | Select logistic regression or histogram gradient boosting by **log loss** |
| Final refit | 2015–2024 | Refit only the selected algorithm |
| Unseen test | 2025 | Report final performance once, with no test-set tuning |

Metrics are **log loss** (primary), Brier score, accuracy, and ROC AUC. Lower log loss/Brier are better: confidently wrong predictions are penalized. A fixed train-only home-win-rate benchmark and an **independently computed pregame Elo benchmark** are scored on the *same games*. Elo uses starting rating 1500, K = 20 and a 35-point home adjustment, resetting by season; its settings are **not tuned** to the 2025 test. Elo is **not a model feature** and is not a published FiveThirtyEight forecast.

No current-game final score, postseason result, or 2025 target enters model training, imputation, scaling, or selection. 2025 games earlier in a season can be used to predict later 2025 games, just as an actual pregame forecaster would; no later game can influence an earlier one.

## Limitations

This prototype does **not** account for announced starting pitchers, lineups, injuries, weather, travel, or individual player talent. It resets form and Elo each season and uses simple early-season priors. A neutral-site flag depends on what the API supplies. Testing on the much more recent 2025 season makes the result more relevant than older archived forecasts, but one held-out season cannot prove future performance.

## Code

- `baseball_prediction/data.py` — pull and validate completed MLB schedule games
- `baseball_prediction/features.py` — leakage-resistant team histories and pregame Elo
- `baseball_prediction/model.py` — compare, select, refit, and score models
- `baseball_prediction/report.py` — out-of-sample calibration plot
- `baseball_prediction/cli.py` — run the complete project
- `tests/` — data validation, same-date leakage, season resets, and temporal splits
- `.github/workflows/pipeline.yml` — GitHub-hosted tests and training

The raw data is retrieved from MLB; this repository's MIT license covers the project code, not a claim of ownership over MLB's underlying records.
