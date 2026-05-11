import argparse
import os
import subprocess
import sys


def _read_cost(solution_file: str):
    if not os.path.isfile(solution_file):
        return None
    with open(solution_file) as fh:
        for line in fh:
            if line.startswith('COST ='):
                try:
                    return int(line.split('=', 1)[1].strip())
                except ValueError:
                    return None
    return None


def _solution_path(instance_path: str, method: str) -> str:
    name = os.path.splitext(os.path.basename(instance_path))[0]
    return os.path.join('solutions', method, name + '_solution.txt')


def main():
    parser = argparse.ArgumentParser(
        description="Run the optimiser repeatedly, warm-starting from the best known solution each time."
    )
    parser.add_argument("instance", help=".txt instance file")
    parser.add_argument("--method", default='alns', choices=['alns', 'greedy_gls'],
                        help="Solver method (default: alns)")
    parser.add_argument("--runs", type=int, default=0,
                        help="Max runs (0 = run until no improvement or Ctrl+C)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after this many consecutive runs with no improvement (default 3)")
    args = parser.parse_args()

    solution_file = _solution_path(args.instance, args.method)
    no_improve = 0
    run = 0

    try:
        while args.runs == 0 or run < args.runs:
            run += 1
            cost_before = _read_cost(solution_file)
            label = f"RUN {run}" if args.runs == 0 else f"RUN {run} / {args.runs}"
            print(f"\n{'#'*60}")
            print(f"  {label}  (best so far: {cost_before:,})" if cost_before else f"  {label}")
            print(f"{'#'*60}\n")

            result = subprocess.run([sys.executable, "main.py", args.instance,
                                     "--method", args.method])
            if result.returncode != 0:
                print(f"\nRun {run} exited with code {result.returncode}, stopping.")
                break

            cost_after = _read_cost(solution_file)
            if cost_after is not None and cost_before is not None and cost_after >= cost_before:
                no_improve += 1
                print(f"\n  No improvement ({no_improve}/{args.patience} consecutive).")
                if no_improve >= args.patience:
                    print(f"  Patience exhausted — stopping.")
                    break
            else:
                no_improve = 0

    except KeyboardInterrupt:
        print(f"\n\nStopped after {run} run(s).")


if __name__ == "__main__":
    main()
