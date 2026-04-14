import argparse
import logging
import os
from instance import Instance
from scheduling import build_schedule, optimize_initial, cost_breakdown, print_cost, validate_schedule
from scheduling.analysis import print_analysis, print_load_distribution

logging.basicConfig(
    filename='schedule.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(message)s'
)


def valid_txt(value: str) -> str:
    if not value.endswith(".txt"):
        raise argparse.ArgumentTypeError(f"{value} is not a .txt file")
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="txt file of the instance to be used.", type=valid_txt)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--patience",   type=int, default=750)
    args = parser.parse_args()

    instance = Instance(args.instance)

    # --- initial schedule ---
    state = build_schedule(instance)
    if not validate_schedule(state['scheduled'], instance):
        raise ValueError("Initial schedule is not valid")
    initial_bd = cost_breakdown(state, instance)

    print(f"\n{'='*60}")
    print("  INITIAL SCHEDULE")
    print(f"{'='*60}")
    print_cost(initial_bd)
    print_load_distribution(state, instance)
    print()
    print_analysis(state, instance)

    # --- LNS ---
    print(f"\n{'='*60}")
    print("  LNS OPTIMISATION")
    print(f"{'='*60}")
    optimize_initial(state, instance, iterations=args.iterations, patience=args.patience)
    best_bd = cost_breakdown(state, instance)

    print(f"\n{'='*60}")
    print("  FINAL SCHEDULE")
    print(f"{'='*60}")
    print_cost(best_bd)
    print_load_distribution(state, instance)
    print()
    print_analysis(state, instance)

    # --- summary ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print_cost(initial_bd, label='Initial')
    print_cost(best_bd,    label='Best   ')
    pct = (initial_bd['total'] - best_bd['total']) / initial_bd['total'] * 100
    print(f"LNS improved by {pct:.1f}%")


if __name__ == "__main__":
    main()
