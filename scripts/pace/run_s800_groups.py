"""Run the three S800 experiments sequentially, with isolated datasets and logs."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from experiment_utils import recording_name

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=("legs", "torso", "arms"), default=["legs", "torso", "arms"])
    parser.add_argument("--delay_mode", choices=("command", "torque"), default="command")
    parser.add_argument("--delay_steps", type=int, default=3)
    parser.add_argument("--fix_delay", type=int, default=None)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--max_frequency", type=float, default=4.)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--freeze_unfitted_gt", action="store_true",
                        help="Explicitly use truth for unoptimized parameters in this sim-to-sim diagnostic")
    parser.add_argument("--smoke", action="store_true", help="1.2-second recordings, 8 environments, two fit iterations")
    parser.add_argument("--check_only", action="store_true", help="Only collect and verify full-length replay; skip optimization")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
    label = ("smoke_" if args.smoke else "experiment_") + stamp
    root = ROOT / "logs" / "pace" / "s800_groups" / label
    root.mkdir(parents=True)
    data_root = ROOT / "data" / "s800_sim" / label
    print(f"Experiment: {root}", flush=True)

    def run(script, arguments, logfile):
        cmd = [sys.executable, "-u", f"scripts/pace/{script}.py", *map(str, arguments)]
        print("Running: " + " ".join(cmd), flush=True)
        with logfile.open("w") as output:
            result = subprocess.run(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            print(logfile.read_text(errors="replace")[-6000:])
            raise SystemExit(f"Failed ({result.returncode}); see {logfile}")
        print(f"Completed; log: {logfile}", flush=True)

    for group in args.groups:
        dataset = data_root / recording_name(group, args.delay_mode)
        common = ["--headless", "--group", group, "--delay_mode", args.delay_mode]
        collect = [*common, "--no-plot", "--delay_steps", args.delay_steps, "--output", dataset,
                   "--max_frequency", args.max_frequency, "--duration", 1. if args.smoke else args.duration]
        if args.smoke:
            collect += ["--rest", .1, "--ramp", .1]
        run("data_collection", collect, root / f"{group}_collect.log")
        if not dataset.exists():
            raise SystemExit(f"Collector exited without producing {dataset}")
        fit_args = [*common, "--data", dataset, "--num_envs", 8 if args.smoke else args.num_envs]
        run("fit", [*fit_args, "--floor_test", "--log_dir", root / group / "floor"], root / f"{group}_floor.log")
        if "✓ PASS" not in (root / f"{group}_floor.log").read_text(errors="replace"):
            raise SystemExit(f"Floor test did not report PASS; see {root / (group + '_floor.log')}")
        if args.check_only:
            continue
        run_args = [*fit_args, "--max_iterations", 2 if args.smoke else args.max_iterations,
                    "--save_interval", 10, "--epsilon", 0, "--log_dir", root / group / "fit"]
        if args.freeze_unfitted_gt:
            run_args += ["--freeze_unfitted_gt"]
        if args.fix_delay is not None:
            run_args += ["--fix_delay", args.fix_delay]
        run("fit", run_args, root / f"{group}_fit.log")
        runs = [p for p in (root / group / "fit").iterdir() if p.is_dir()]
        latest = max(runs, key=lambda p: p.stat().st_mtime)
        run("eval_group", ["--run", latest], root / f"{group}_eval.log")
    print(f"All requested groups completed. Results: {root}", flush=True)


if __name__ == "__main__":
    main()
