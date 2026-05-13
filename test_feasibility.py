import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.insert(0, os.path.dirname(__file__))

from instance import Instance
from scheduling.state import build_schedule, validate_schedule
from scheduling.feasibility import repair_feasibility
from scheduling.cost import cost_breakdown, print_cost


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'instances/B3.txt'
    instance = Instance(path)

    tool_summary = ', '.join(f'type {t.id}: {t.num_available} avail' for t in instance.tools)
    print(f"Instance : {instance.name}")
    print(f"Requests : {len(instance.requests)}")
    print(f"Days     : {instance.config.days}")
    print(f"Tools    : {tool_summary}")

    state = build_schedule(instance)
    n_before = sum(len(v) for v in state['unscheduled'].values())
    print(f"\nAfter greedy construction: {n_before} unscheduled request(s)")

    if n_before == 0:
        print("  Already fully scheduled — feasibility repair not needed.")
        validate_schedule(state['scheduled'], instance)
        print_cost(cost_breakdown(state, instance))
        return

    for machine_type, reqs in state['unscheduled'].items():
        if reqs:
            print(f"  type {machine_type}: {len(reqs)} unscheduled")

    print("\nRunning CP-SAT feasibility repair...")
    success = repair_feasibility(state, instance)

    n_after = sum(len(v) for v in state['unscheduled'].values())
    print(f"\nResult   : {'SUCCESS' if success else 'FAILED'}")
    print(f"Remaining unscheduled: {n_after}")

    if success:
        valid = validate_schedule(state['scheduled'], instance)
        print(f"Schedule valid: {valid}")
        print()
        print_cost(cost_breakdown(state, instance))
    else:
        print("\nInstance may be genuinely infeasible for the blocked tool type.")
        print("CP-SAT proved no valid delivery-day assignment exists within the time windows.")


if __name__ == '__main__':
    main()
