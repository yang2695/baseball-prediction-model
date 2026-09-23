"""Run the full historical training pipeline: python -m baseball_prediction.cli."""

import argparse
import json
from pathlib import Path

import joblib

from baseball_prediction.data import DEFAULT_RAW_PATH, download_games, load_games
from baseball_prediction.features import make_features
from baseball_prediction.model import train_and_evaluate
from baseball_prediction.report import plot_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate MLB win probabilities")
    parser.add_argument("--data", type=Path, default=DEFAULT_RAW_PATH,
                        help="Where to cache/read the FiveThirtyEight CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"),
                        help="Destination for model, JSON metrics, and test predictions")
    parser.add_argument("--refresh", action="store_true", help="Redownload the CSV")
    args = parser.parse_args()

    source = download_games(args.data, refresh=args.refresh)
    games = load_games(source)
    features = make_features(games)
    model, summary, predictions = train_and_evaluate(features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.output_dir / "model.joblib")
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    plot_calibration(predictions, args.output_dir / "calibration.png")
    print(json.dumps({
        "selected_model": summary["selected_model"],
        "game_counts": summary["game_counts"],
        "validation": summary["validation"],
        "test": summary["test"],
        "saved_to": str(args.output_dir),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
