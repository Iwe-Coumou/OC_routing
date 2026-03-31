import pandas as pd
import argparse
import os
from instance import Instance

MATRIX_VALUES = ['TOOLS', 'COORDINATES', 'REQUESTS']


def valid_txt(value: str) -> str:
    if not value.endswith(".txt"):
        raise argparse.ArgumentTypeError(f"{value} is not a .txt file")
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return value

def _get_key_value(line: str) -> tuple[str, str]:
    key, value = line.split("=")
    return key.strip(), value.strip()

def _lines_to_matrix(lines: list[str]) -> pd.DataFrame:
    matrix = [[int(item.strip()) for item in line.split("\t") if item] for line in lines]
    return pd.DataFrame(matrix)

def _proccess_instance(file_str: str) -> dict:
    result = dict()
    lines = file_str.split("\n")
    for i in range(len(lines)):
        if "=" in lines[i]:
            key, value = _get_key_value(lines[i])

            if key in MATRIX_VALUES:
                size = int(value)
                result[key.lower()] = _lines_to_matrix(lines[i+1:i+size+1])
                i+=size
                continue
            else:
                try:
                    value = int(value)
                except:
                    pass
                result[key.lower()] = value

        if lines[i] == "DISTANCE":
            result["DISTANCE".lower()] = _lines_to_matrix(lines[i+1:])
            break

    return result


def read_instance(filename: str) -> Instance:
    with open(filename, "r") as f:
        instance_str = f.read().strip()
    
    items = "\n".join(instance_str.split("\n\n"))
    parsed = _proccess_instance(items)
    for k, v in parsed.items():
        print(f"{k}: {type(v)}")
    
    return Instance(**parsed)
    
