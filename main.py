import argparse
import logging
import os
from instance import Instance
from scheduling import build_schedule, cost_breakdown, print_cost, validate_schedule
from scheduling.analysis import print_analysis, print_load_distribution
from routing import write_solution, cost_from_routes, solve_all_days
from optimiser import optimize

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

    # --- initial schedule ---
    state = build_schedule(instance)
    if not validate_schedule(state['scheduled'], instance):
        raise ValueError("Initial schedule is not valid")

    print(f"\n{'='*60}")
    print("  INITIAL SCHEDULE")
    print(f"{'='*60}")
    print_cost(cost_breakdown(state, instance))
    print_load_distribution(state, instance)
    print()
    print_analysis(state, instance)

    # --- optimise ---
    print(f"\n{'='*60}")
    print("  OPTIMISING")
    print(f"{'='*60}")
    route_set = optimize(state, instance)

    # --- final cost ---
    print(f"\n{'='*60}")
    print("  FINAL COST")
    print(f"{'='*60}")
    routed_bd = cost_from_routes(route_set, instance)
    print_cost(routed_bd)

    output_file = args.instance.replace('.txt', '_solution.txt')
    write_solution(route_set, instance, output_file)
    print(f"\n  Solution written to {output_file}")


if __name__ == "__main__":
    main()
