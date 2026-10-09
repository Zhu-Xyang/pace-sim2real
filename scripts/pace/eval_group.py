"""Report only identified joints, using the truth saved with that experiment (no simulator)."""

import argparse
import csv
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Exact fitting run directory")
    parser.add_argument("--params", default="best_params.pt", help="Sampled best (default), or a mean_*.pt file")
    args = parser.parse_args()
    config = torch.load(args.run / "config.pt", map_location="cpu", weights_only=True)
    params = torch.load(args.run / args.params, map_location="cpu", weights_only=True)
    names = config["joint_order"]
    fit = config.get("fit", {})
    selected = fit.get("fitted_joints", list(range(len(names))))
    gt = config.get("gt")
    if gt is None or gt["joint_order"] != names:
        raise ValueError("Evaluation requires the recorded sim-to-sim truth in matching joint order")
    if fit.get("floor_test"):
        print("NOTE: floor test; this verifies replay consistency, not parameter recovery.")
    if fit.get("freeze_unfitted_gt"):
        print("NOTE: conditional sim-to-sim recovery; other groups were fixed at truth.")
    print(f"Parameters: {args.params}; fitted joints: {len(selected)}; delay: {fit.get('delay_mode')}")
    rows = []
    print(f"{'Joint':<28} {'Parameter':<10} {'GT':>10} {'ID':>10} {'Abs error':>12} {'Error %':>10}")
    for j in selected:
        for block, key in enumerate(("armature", "damping", "friction", "bias")):
            truth, identified = float(gt[key][j]), float(params[block * len(names) + j])
            error = identified - truth
            percent = 100 * error / truth if truth else None
            pct = f"{percent:+.2f}" if percent is not None else "N/A"
            print(f"{names[j]:<28} {key:<10} {truth:10.5f} {identified:10.5f} {error:+12.5g} {pct:>10}")
            rows.append([names[j], key, truth, identified, error, percent])
    effective = int(torch.round(params[-1]))
    dt = config.get("experiment", {}).get("physics_dt")
    print(f"Delay: raw={float(params[-1]):.6f}, effective={effective} steps, GT={gt['delay']:g} steps"
          + (f", effective={effective * dt * 1000:g} ms" if dt else ""))
    if any("TORSO_YAW" in names[j] for j in selected):
        print("NOTE: torso yaw bias may be unobservable in fixed-base encoder-only experiments; do not infer accuracy from the floor test.")
    output = args.run / "group_comparison.csv"
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Joint", "Parameter", "GT", "ID", "Signed_error", "Error_percent"])
        writer.writerows(rows)
        writer.writerow(["global", "delay_steps", gt["delay"], effective, effective - gt["delay"], ""])
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
