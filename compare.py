import argparse
import os


def _read_solution(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    data = {}
    with open(path) as fh:
        for line in fh:
            for key in ('COST', 'MAX_NUMBER_OF_VEHICLES', 'NUMBER_OF_VEHICLE_DAYS',
                        'TOOL_USE', 'DISTANCE'):
                if line.startswith(key + ' ='):
                    val = line.split('=', 1)[1].strip()
                    if key == 'TOOL_USE':
                        data[key] = list(map(int, val.split()))
                    else:
                        try:
                            data[key] = int(val)
                        except ValueError:
                            data[key] = val
    return data if data else None


def main():
    parser = argparse.ArgumentParser(
        description="Compare solution costs across methods."
    )
    parser.add_argument('--methods', nargs='+',
                        help='Methods to compare (default: all subdirs of solutions/)')
    parser.add_argument('--detail', action='store_true',
                        help='Show per-component cost breakdown')
    args = parser.parse_args()

    solutions_dir = 'solutions'
    if args.methods:
        methods = args.methods
    else:
        if not os.path.isdir(solutions_dir):
            print("No solutions/ directory found.")
            return
        methods = sorted(
            d for d in os.listdir(solutions_dir)
            if os.path.isdir(os.path.join(solutions_dir, d))
        )

    if not methods:
        print("No method directories found in solutions/.")
        return

    # Collect all instance names across all methods
    instances = set()
    for method in methods:
        method_dir = os.path.join(solutions_dir, method)
        if not os.path.isdir(method_dir):
            continue
        for f in os.listdir(method_dir):
            if f.endswith('_solution.txt'):
                instances.add(f.replace('_solution.txt', ''))
    instances = sorted(instances)

    if not instances:
        print("No solution files found.")
        return

    # Load all solutions
    table = {}
    for instance in instances:
        table[instance] = {}
        for method in methods:
            path = os.path.join(solutions_dir, method, instance + '_solution.txt')
            table[instance][method] = _read_solution(path)

    # Print summary table
    col_w = 14
    header = f"  {'Instance':<35}"
    for method in methods:
        header += f"  {method:>{col_w}}"
    if len(methods) == 2:
        header += f"  {'delta':>{col_w}}  {'improvement':>11}"
    print(header)
    print('  ' + '-' * (35 + len(methods) * (col_w + 2) + (30 if len(methods) == 2 else 0)))

    for instance in instances:
        row = f"  {instance:<35}"
        costs = []
        for method in methods:
            sol = table[instance][method]
            if sol and 'COST' in sol:
                row += f"  {sol['COST']:>{col_w},}"
                costs.append(sol['COST'])
            else:
                row += f"  {'—':>{col_w}}"
                costs.append(None)

        if len(methods) == 2 and costs[0] is not None and costs[1] is not None:
            delta = costs[1] - costs[0]
            pct = 100 * delta / costs[0]
            row += f"  {delta:>+{col_w},}  {pct:>+10.2f}%"
        print(row)

    # Summary
    if len(methods) == 2:
        pairs = [
            (table[i][methods[0]]['COST'], table[i][methods[1]]['COST'])
            for i in instances
            if table[i][methods[0]] and table[i][methods[1]]
            and 'COST' in table[i][methods[0]] and 'COST' in table[i][methods[1]]
        ]
        if pairs:
            total_0 = sum(a for a, _ in pairs)
            total_1 = sum(b for _, b in pairs)
            total_delta = total_1 - total_0
            total_pct = 100 * total_delta / total_0
            n = len(pairs)
            print('  ' + '-' * (35 + len(methods) * (col_w + 2) + 30))
            row = f"  {'TOTAL (' + str(n) + ' instances)':<35}"
            row += f"  {total_0:>{col_w},}  {total_1:>{col_w},}"
            row += f"  {total_delta:>+{col_w},}  {total_pct:>+10.2f}%"
            print(row)

    # Detailed breakdown
    if args.detail:
        print(f"\n{'='*75}")
        print("  COST BREAKDOWN")
        print(f"{'='*75}")
        components = [('COST', 'total'), ('TOOL_USE', 'tools (units)'),
                      ('MAX_NUMBER_OF_VEHICLES', 'max vehicles'),
                      ('NUMBER_OF_VEHICLE_DAYS', 'vehicle-days'),
                      ('DISTANCE', 'distance')]
        for instance in instances:
            print(f"\n  {instance}")
            for method in methods:
                sol = table[instance][method]
                if sol is None:
                    print(f"    {method}: —")
                    continue
                parts = []
                for key, label in components:
                    if key in sol:
                        val = sol[key]
                        if isinstance(val, list):
                            parts.append(f"{label}=[{' '.join(map(str,val))}]")
                        else:
                            parts.append(f"{label}={val:,}")
                print(f"    {method}: {' | '.join(parts)}")


if __name__ == '__main__':
    main()
