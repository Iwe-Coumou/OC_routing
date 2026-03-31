import argparse
import os
from instance import Instance

def valid_txt(value: str) -> str:
    if not value.endswith(".txt"):
        raise argparse.ArgumentTypeError(f"{value} is not a .txt file")
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return value

def read_instance(filename: str) -> Instance:
    with open(filename, "r") as f:
        instance_str = f.read()
    
    items = instance_str.split("\n\n")


    return None
    
