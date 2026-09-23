"""Plot out-of-sample probability calibration."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd


def plot_calibration(predictions: pd.DataFrame, destination: Path) -> None:
    """Compare observed win rates with pregame probability forecasts."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for column, label in [
        ("predicted_home_win_probability", "Selected model"),
        ("elo_prob_home", "Independent pregame Elo"),
    ]:
        observed, predicted = calibration_curve(
            predictions["home_win"],
            predictions[column],
            n_bins=10,
            strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", label=label)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted home-win probability",
        ylabel="Observed home-win rate",
        title="Held-out 2025 MLB probability calibration",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
