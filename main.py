import argparse
from instance import Instance
from initial_schedule import build_schedule, validate_schedule, optimize_rechain
import os

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
    #print_instance(instance)
    
    state = build_schedule(instance)
    print(f"Rechained: {optimize_rechain(state, instance)}")
    schedule = state['scheduled']
    validated = validate_schedule(schedule, instance)
    if not validated:
        raise ValueError("Schedule not valid")
    for entry in schedule:
        r = entry['request']
        chain = f" (chained from req {entry['chained_from']['request'].id})" if entry['chained_from'] else ""
        print(f"req={r.id} type={r.machine_type} loc={r.location_id} "
              f"delivery={entry['delivery_day']} pickup={entry['pickup_day']}{chain}")
    

def print_instance(instance: Instance) -> None:
    print(f"Dataset: {instance.dataset}")
    print(f"Name: {instance.name}")
    print()

    print("=== Config ===")
    for k, v in vars(instance.config).items():
        print(f"  {k}: {v}")
    print()

    print(f"=== Depot ===")
    print(f"  {instance.depot}")
    print()

    print(f"=== Tools ({len(instance.tools)}) ===")
    for tool in instance.tools:
        print(f"  {tool}")
    print()

    print(f"=== Coordinates ({len(instance.coordinates)}) ===")
    for i, coord in enumerate(instance.coordinates):
        print(f"  {i}: {coord}")
    print()

    print(f"=== Requests ({len(instance.requests)}) ===")
    for req in instance.requests:
        print(f"  {req}")
    print()

    print(f"=== Distance matrix ===")
    for row in instance.distance:
        print(row)

if __name__ == "__main__":
    main()
