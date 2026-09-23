# Baseball prediction model

A reproducible Python project for **predicting the home team's probability of winning an MLB game before first pitch**. Built as a baseball analytics portfolio project.

The project uses historical regular-season game results from [FiveThirtyEight's MLB Elo dataset](https://github.com/fivethirtyeight/data/tree/master/mlb-elo). Our predictive features are calculated from teams' **prior games**, not from the current game's final score or postgame ratings.

## Project plan

1. Download and validate historical games (the raw CSV is not committed).
2. Build team form, scoring, season record, and rest features as of each game date.
3. Train logistic regression and gradient boosting; pick the model using 2021 validation results.
4. Refit on 2015–2021 data and report genuinely held-out 2022 results.
5. Compare against a train-set home-win-rate baseline and FiveThirtyEight's **pregame** Elo probabilities.

Metrics: log loss (primary), Brier score, accuracy, and ROC AUC. Lower log loss/Brier are better. The model is not a betting recommendation or a live 2026 forecast.

## Development

This repository is being developed in small, meaningful commits. Usage and actual evaluation results will be added with the implementation. No performance figures are claimed until the pipeline has run.
