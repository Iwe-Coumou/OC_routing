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
        description="Run the optimiser on all instances in the instances/ directory."
    )
    parser.add_argument('--dir', default='instances',
                        help='Directory containing .txt instance files (default: instances/)')
    parser.add_argument('--method', default='alns', choices=['alns', 'greedy_gls'],
                        help='Solver method (default: alns)')
    args = parser.parse_args()

    instances = sorted(
        os.path.join(args.dir, f)
        for f in os.listdir(args.dir)
        if f.endswith('.txt') and '_solution' not in f
    )

    if not instances:
        print(f"No instance files found in {args.dir}/")
        return

    print(f"Found {len(instances)} instances  |  method={args.method}\n")

    results = []
    for instance_path in instances:
        name = os.path.basename(instance_path)
        solution_path = _solution_path(instance_path, args.method)
        cost_before = _read_cost(solution_path)

        result = subprocess.run([sys.executable, 'main.py', instance_path,
                                 '--method', args.method])

        cost_after = _read_cost(solution_path)
        improved = (cost_before is None and cost_after is not None) or \
                   (cost_before is not None and cost_after is not None and cost_after < cost_before)
        results.append((name, cost_before, cost_after, result.returncode, improved))

        tag = 'IMPROVED' if improved else ('FAILED' if result.returncode != 0 else 'no change')
        before_str = f"{cost_before:,}" if cost_before is not None else '—'
        after_str  = f"{cost_after:,}"  if cost_after  is not None else '—'
        print(f"  {name:<35}  {before_str:>12} -> {after_str:>12}  [{tag}]")

    print(f"\n{'='*75}")
    print(f"  {'Instance':<35}  {'Cost':>12}  {'Status'}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*10}")
    for name, _, after, rc, improved in results:
        cost_str = f"{after:,}" if after is not None else 'no solution'
        status = 'FAILED' if rc != 0 else ('improved' if improved else 'no change')
        print(f"  {name:<35}  {cost_str:>12}  {status}")
    print(f"{'='*75}")

    solved = sum(1 for _, _, after, rc, _ in results if after is not None and rc == 0)
    improved = sum(1 for *_, imp in results if imp)
    print(f"\n  {solved}/{len(instances)} solved  |  {improved} improved this run")


if __name__ == '__main__':
    main()
