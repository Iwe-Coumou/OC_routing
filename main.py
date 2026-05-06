import argparse
import logging
import os
from instance import Instance
from scheduling import build_schedule, cost_breakdown, print_cost, validate_schedule, schedule_lns
from routing import write_solution, cost_from_routes, read_solution
from optimiser import route_lns

logging.basicConfig(
    filename='schedule.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(message)s'
)

_opt_handler = logging.FileHandler('optimiser.log', mode='w')
_opt_handler.setFormatter(logging.Formatter('%(message)s'))
_opt_logger = logging.getLogger('optimiser')
_opt_logger.setLevel(logging.INFO)
_opt_logger.addHandler(_opt_handler)
_opt_logger.propagate = False  # keep optimiser events out of schedule.log


def valid_txt(value: str) -> str:
    if not value.endswith(".txt"):
        raise argparse.ArgumentTypeError(f"{value} is not a .txt file")
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="txt file of the instance to be used.", type=valid_txt)
    args = parser.parse_args()

    instance = Instance(args.instance)
    output_file = args.instance.replace('.txt', '_solution.txt')

    # --- warm start from existing solution if available ---
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
        # --- cold start: build fresh schedule ---
        state = build_schedule(instance)
        if not validate_schedule(state['scheduled'], instance):
            raise ValueError("Initial schedule is not valid")

        print(f"\n{'='*60}")
        print("  INITIAL SCHEDULE")
        print(f"{'='*60}")
        print_cost(cost_breakdown(state, instance))

        print(f"\n{'='*60}")
        print("  SCHEDULING LNS")
        print(f"{'='*60}")
        schedule_lns(state, instance, iterations=1000, patience=250)
        print_cost(cost_breakdown(state, instance))

    # --- optimise ---
    print(f"\n{'='*60}")
    print("  OPTIMISING")
    print(f"{'='*60}")
    route_set = route_lns(state, instance, iterations=500, patience=500,
                          refine_time=0, initial_routes=initial_routes)

    # --- final cost ---
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
