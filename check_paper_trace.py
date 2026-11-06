"""check the baseline curve against the printed Figure 5B blue line

run the main project first then run python check_paper_trace.py
use --results if the main results are in a different folder"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    reference_folder = Path(__file__).resolve().parent / "paper_reference"
    reference = np.genfromtxt(reference_folder / "figure5B_trace.csv", delimiter=",", names=True)
    model = np.genfromtxt(args.results / "data/known_lambda_0.006_k_1.4.csv", delimiter=",", names=True)
    calibration = json.loads((reference_folder / "calibration.json").read_text())

    # compare at the visible source columns without shifting the model
    keep = (reference["time"] >= model["time"][0]) & (reference["time"] <= model["time"][-1])
    reference = reference[keep]
    predicted = np.interp(reference["time"], model["time"], model["T"])
    below = reference["T_lower"] - predicted
    above = predicted - reference["T_upper"]
    outside = np.maximum(np.maximum(below, above), 0)
    outside_pixels = outside * abs(calibration["y_scale"])
    fraction_inside = float(np.mean(outside_pixels == 0))
    print(f"source columns compared: {len(reference)}")
    print(f"inside the printed blue stroke: {100*fraction_inside:.2f} percent")
    print(f"largest distance outside the stroke: {outside_pixels.max():.3f} source pixels")
    print("the tick positions have about one source pixel of reading uncertainty")
    print("this checks the time series inset rather than the full five state model")


if __name__ == "__main__":
    main()
