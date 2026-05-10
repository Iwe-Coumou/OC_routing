import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def _run_instance(instance_path: str) -> tuple:
    name = os.path.basename(instance_path)
    solution_path = instance_path.replace('.txt', '_solution.txt')
    cost_before = _read_cost(solution_path)

    result = subprocess.run([sys.executable, 'main.py', instance_path],
                            capture_output=False)

    cost_after = _read_cost(solution_path)
    improved = (cost_before is None and cost_after is not None) or \
               (cost_before is not None and cost_after is not None and cost_after < cost_before)
    return name, cost_before, cost_after, result.returncode, improved


def main():
    parser = argparse.ArgumentParser(
        description="Run the optimiser on all instances in the instances/ directory."
    )
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of instances to run in parallel (default 1). '
                             'Each instance itself uses parallel threads for fast routing, '
                             'so >2 workers may oversubscribe CPUs on smaller machines.')
    parser.add_argument('--dir', default='instances',
                        help='Directory containing .txt instance files (default: instances/)')
    args = parser.parse_args()

    instances = sorted(
        os.path.join(args.dir, f)
        for f in os.listdir(args.dir)
        if f.endswith('.txt') and not f.endswith('_solution.txt')
    )

    if not instances:
        print(f"No instance files found in {args.dir}/")
        return

    print(f"Found {len(instances)} instances  |  workers={args.workers}\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_instance, p): p for p in instances}
        for future in as_completed(futures):
            name, before, after, rc, improved = future.result()
            results.append((name, before, after, rc, improved))
            tag = 'IMPROVED' if improved else ('FAILED' if rc != 0 else 'no change')
            before_str = f"{before:,}" if before is not None else '—'
            after_str  = f"{after:,}"  if after  is not None else '—'
            print(f"  {name:<35}  {before_str:>12} -> {after_str:>12}  [{tag}]")

    results.sort(key=lambda r: r[0])
    print(f"\n{'='*75}")
    print(f"  {'Instance':<35}  {'Cost':>12}  {'Status'}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*10}")
    for name, before, after, rc, improved in results:
        cost_str = f"{after:,}" if after is not None else 'no solution'
        status = 'FAILED' if rc != 0 else ('improved' if improved else 'no change')
        print(f"  {name:<35}  {cost_str:>12}  {status}")
    print(f"{'='*75}")

    solved = sum(1 for _, _, after, rc, _ in results if after is not None and rc == 0)
    improved = sum(1 for *_, imp in results if imp)
    print(f"\n  {solved}/{len(instances)} solved  |  {improved} improved this run")


if __name__ == '__main__':
    main()
