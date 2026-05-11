import argparse
import logging
import os
from instance import Instance
from scheduling.state import build_schedule, build_schedule_single, validate_schedule, CONSTRUCTION_KEYS
from scheduling.cost import cost_breakdown, print_cost
from routing.export import write_solution, cost_from_routes, read_solution
from routing.solver import solve_routing
from optimiser.lns import route_lns

def _setup_logging(instance_name: str) -> None:
    logging.basicConfig(
        filename=f'logs/{instance_name}_schedule.log',
        filemode='w',
        level=logging.DEBUG,
        format='%(message)s'
    )
    opt_handler = logging.FileHandler(f'logs/{instance_name}_optimiser.log', mode='w')
    opt_handler.setFormatter(logging.Formatter('%(message)s'))
    opt_logger = logging.getLogger('optimiser')
    opt_logger.setLevel(logging.INFO)
    opt_logger.addHandler(opt_handler)
    opt_logger.propagate = False

GREEDY_METHODS = {f'greedy_{k}_gls': key for k, key in CONSTRUCTION_KEYS.items()}
METHODS = ['alns', 'greedy_gls'] + sorted(GREEDY_METHODS)


def _solution_path(instance_path: str, method: str) -> str:
    name = os.path.splitext(os.path.basename(instance_path))[0]
    return os.path.join('solutions', method, name + '_solution.txt')


def valid_txt(value: str):
    if not value.endswith(".txt"):
        raise argparse.ArgumentTypeError(f"{value} is not a .txt file")
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="txt file of the instance to be used.", type=valid_txt)
    parser.add_argument("--method", choices=METHODS, default='alns',
                        help="Solver method: 'alns' (default) or 'greedy_gls' (benchmark)")
    args = parser.parse_args()

    instance = Instance(args.instance)
    os.makedirs('logs', exist_ok=True)
    _setup_logging(instance.name.replace(' ', '_'))
    output_file = _solution_path(args.instance, args.method)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if args.method in GREEDY_METHODS:
        state = build_schedule_single(instance, GREEDY_METHODS[args.method])
        validate_schedule(state['scheduled'], instance)

        print(f"\n{'='*60}")
        print(f"  GREEDY SCHEDULE ({args.method})")
        print(f"{'='*60}")
        print_cost(cost_breakdown(state, instance))

        print(f"\n{'='*60}")
        print("  ROUTING (GLS)")
        print(f"{'='*60}")
        print("  computing fast routes...", flush=True)
        fast_routes = solve_routing(state, instance, fast=True)
        route_set = solve_routing(state, instance, fast=False, time_limit_seconds=30,
                                  initial_routes=fast_routes)

    elif args.method == 'greedy_gls':
        state = build_schedule(instance)
        validate_schedule(state['scheduled'], instance)

        print(f"\n{'='*60}")
        print("  GREEDY SCHEDULE (best construction)")
        print(f"{'='*60}")
        print_cost(cost_breakdown(state, instance))

        print(f"\n{'='*60}")
        print("  ROUTING (GLS)")
        print(f"{'='*60}")
        print("  computing fast routes...", flush=True)
        fast_routes = solve_routing(state, instance, fast=True)
        route_set = solve_routing(state, instance, fast=False, time_limit_seconds=30,
                                  initial_routes=fast_routes)

    else:  # alns
        initial_routes = None
        if os.path.isfile(output_file):
            try:
                state, initial_routes = read_solution(output_file, instance)
                print(f"\n{'='*60}")
                print("  WARM START")
                print(f"{'='*60}")
                print_cost(cost_from_routes(initial_routes, instance))
            except Exception as e:
                print(f"\n  Could not load existing solution ({e}), starting fresh.")
                initial_routes = None

        if initial_routes is None:
            state = build_schedule(instance)
            validate_schedule(state['scheduled'], instance)

            print(f"\n{'='*60}")
            print("  INITIAL SCHEDULE")
            print(f"{'='*60}")
            print_cost(cost_breakdown(state, instance))

        print(f"\n{'='*60}")
        print("  OPTIMISING")
        print(f"{'='*60}")
        route_set = route_lns(state, instance, iterations=500, patience=150,
                              initial_routes=initial_routes)

    print(f"\n{'='*60}")
    print("  FINAL COST")
    print(f"{'='*60}")
    routed_bd = cost_from_routes(route_set, instance)
    print_cost(routed_bd)

    new_cost = routed_bd['total']
    existing_cost = None
    if os.path.isfile(output_file):
        with open(output_file) as fh:
            for line in fh:
                if line.startswith('COST ='):
                    try:
                        existing_cost = int(line.split('=', 1)[1].strip())
                    except ValueError:
                        pass
                    break

    if existing_cost is None or new_cost < existing_cost:
        write_solution(route_set, instance, output_file)
        if existing_cost is None:
            print(f"\n  Solution written to {output_file}  (cost={new_cost:,})")
        else:
            print(f"\n  Solution written to {output_file}  (cost={new_cost:,}, improved from {existing_cost:,})")
    else:
        print(f"\n  Existing solution is better ({existing_cost:,} <= {new_cost:,}), not overwriting.")


if __name__ == "__main__":
    main()
