# Baseball prediction model

[![Test and train baseball model](https://github.com/yang2695/baseball-prediction-model/actions/workflows/pipeline.yml/badge.svg)](https://github.com/yang2695/baseball-prediction-model/actions/workflows/pipeline.yml)

Predict **the home team's pregame probability of winning an MLB game** using historical regular-season results. This is an end-to-end Python baseball analytics portfolio project, not a betting service or a live 2026 forecast.

## Data and question

[FiveThirtyEight's historical MLB Elo file](https://github.com/fivethirtyeight/data/tree/master/mlb-elo) provides game dates, teams, final scores, and **pregame** Elo forecasts. Its public sports forecasts were discontinued in 2023, so this project deliberately uses completed seasons **2015–2022** rather than treating the feed as live. The CSV is downloaded on demand, cached under `data/raw/`, and not committed; attribute the original dataset separately from this repo's MIT-licensed code.

**Target:** Did team1 (the home team, except that the venue may be neutral) win the game? Postseason games, unfinished games, and ties are excluded.

**Predictors, calculated as of each game's date:**
- Home and away teams' previous 10 games: win rate, runs scored/allowed, run differential
- Season-to-date win rates, run differential per game, and games played
- Days of rest for each team; neutral-site indicator; month of year

All teams start each season with neutral priors (50% win rate and 4.5 runs scored/allowed); only games on **earlier dates** are used. Because game times are not provided, no result from a doubleheader can inform another game on the same date. Crucially, **final score, postgame Elo, and FiveThirtyEight's pregame Elo probabilities are not model features**; Elo is only an external comparison.

## Reproduce the experiment

Python 3.11 or newer recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m baseball_prediction.cli
```

To reuse a different CSV location or force a download:

```bash
python -m baseball_prediction.cli --data data/raw/mlb_elo.csv --refresh
```

The runner produces four files in `artifacts/`: `model.joblib` (selected fitted estimator), `evaluation.json` (scores and sample counts), `test_predictions.csv` (one row per held-out game), and `calibration.png`. Generated files are intentionally ignored by Git. Do not load `.joblib` files from untrusted sources.

**Prefer running everything on GitHub?** The [Actions workflow](https://github.com/yang2695/baseball-prediction-model/actions/workflows/pipeline.yml) installs dependencies, executes unit tests, downloads historical data, trains the models, and uploads `baseball-model-results` as a downloadable workflow artifact. It also prints the actual test scores in the run log.

## Evaluation design

| Stage | Seasons | Purpose |
| --- | --- | --- |
| Initial training | 2015–2020 | Learn from past results |
| Validation | 2021 | Select logistic regression vs. histogram gradient boosting by **log loss** |
| Final refit | 2015–2021 | Refit selected algorithm, with 2022 still unseen |
| Final test | 2022 | Report one held-out comparison |

We report log loss (primary), Brier score, accuracy, and ROC AUC. Lower log loss/Brier means better probability forecasts. The two benchmarks are (1) the historical home-win frequency from **training years only** and (2) FiveThirtyEight's original **pregame** Elo probability from the same test games. The gradient booster is deliberately modest in complexity; model choice is made on 2021, **not** on 2022.

The model has real limitations: no announced starting pitchers, injuries, lineup changes, travel geography, or live data; all same-date games use pre-date history; season-opening priors are simple and not optimized. A random train/test shuffle would produce an unrealistically easy evaluation, so the split is chronological.

## Repository map

- `baseball_prediction/data.py`: retrieve and validate the original game file
- `baseball_prediction/features.py`: make leak-resistant pregame team histories
- `baseball_prediction/model.py`: compare estimators, choose and evaluate
- `baseball_prediction/report.py`: reliability plot
- `baseball_prediction/cli.py`: one-command experiment
- `tests/`: data cleaning, doubleheader leakage, new-season resets, and temporal split tests
- `.github/workflows/pipeline.yml`: GitHub-hosted test and training run

No model performance numbers are included in this README unless they have been observed in a completed run.
