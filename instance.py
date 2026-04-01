import pandas as pd
from dataclasses import dataclass

MATRIX_VALUES = ['TOOLS', 'COORDINATES', 'REQUESTS']

@dataclass
class Config:
    days: int
    capacity: int
    max_trip_distance: int
    vehicle_cost: int
    vehicle_day_cost: int
    distance_cost: int

@dataclass
class Request:
    id: int
    location_id: int
    earliest: int
    latest: int
    duration: int
    machine_type: int
    num_machines: int

    def pickup_day(self, delivery_day: int) -> int:
        return delivery_day + self.duration

    def is_feasible(self, delivery_day: int) -> bool:
        return self.earliest <= delivery_day <= self.latest

@dataclass
class Tool:
    id: int
    size: int
    num_available: int
    cost: int

class Instance:
    def __init__(self, filename: str):
        self._load(filename)

    def _load(self, filename: str):
        with open(filename, "r") as f:
            raw = f.read().strip()

        items = "\n".join(raw.split("\n\n"))
        parsed = _process_instance(items)

        self.dataset = parsed.get("dataset")
        self.name = parsed.get("name")


        self.config = Config(
            days=parsed["days"],
            capacity=parsed["capacity"],
            max_trip_distance=parsed["max_trip_distance"],
            vehicle_cost=parsed["vehicle_cost"],
            vehicle_day_cost=parsed["vehicle_day_cost"],
            distance_cost=parsed["distance_cost"],
        )

        requests_df = parsed.get("requests")
        assert isinstance(requests_df, pd.DataFrame), "No requests found in instance file"
        self.requests = [
            Request(
                id=int(row[0]),
                location_id=int(row[1]),
                earliest=int(row[2]),
                latest=int(row[3]),
                duration=int(row[4]),
                machine_type=int(row[5]),
                num_machines=int(row[6]),
            )
            for _, row in requests_df.iterrows()
        ]

        tools_df = parsed.get("tools")
        assert isinstance(tools_df, pd.DataFrame), "No tools found in instance file"
        self.tools = [
            Tool(
                id=int(row[0]),
                size=int(row[1]),
                num_available=int(row[2]),
                cost=int(row[3]),
            )
            for _, row in tools_df.iterrows()
        ]

        coordinates_df = parsed.get("coordinates")
        assert isinstance(coordinates_df, pd.DataFrame), "No coordinates found in instance file"
        self.coordinates = [
            (int(row[1]), int(row[2]))
            for _, row in coordinates_df.iterrows()
        ]
        depot_id = parsed.get("depot_coordinate", 0)
        self.depot = self.coordinates[depot_id]

        distance_df = parsed.get("distance")
        self.distance = distance_df# keep as DataFrame, easy to index

    def _print(self):
        for k, v in vars(self).items():
            print(f"{k}: {v}")


def _get_key_value(line: str) -> tuple[str, str]:
    key, value = line.split("=")
    return key.strip(), value.strip()

def _lines_to_matrix(lines: list[str]) -> pd.DataFrame:
    matrix = [[int(item.strip()) for item in line.split("\t") if item] for line in lines]
    return pd.DataFrame(matrix)

def _process_instance(file_str: str) -> dict:
    result = dict()
    lines = file_str.split("\n")
    for i in range(len(lines)):
        if "=" in lines[i]:
            key, value = _get_key_value(lines[i])
            if key in MATRIX_VALUES:
                size = int(value)
                result[key.lower()] = _lines_to_matrix(lines[i+1:i+size+1])
                i += size
                continue
            else:
                try:
                    value = int(value)
                except ValueError:
                    pass
                result[key.lower()] = value
        if lines[i] == "DISTANCE":
            result["distance"] = _lines_to_matrix(lines[i+1:])
            break
    return result
