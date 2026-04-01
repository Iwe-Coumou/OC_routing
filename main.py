import argparse
from instance import Instance
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
    print(f"  {instance.distance}")

if __name__ == "__main__":
    main()
