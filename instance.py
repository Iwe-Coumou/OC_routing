import pandas as pd
from dataclasses import dataclass
from InstanceCVRPTWUI import InstanceCVRPTWUI  # teacher's file

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
        

        raw = InstanceCVRPTWUI(filename)
        if not raw.isValid():
            raise ValueError(
                "Invalid instance file:\n" + "\n".join(raw.errorReport)
            )

        self.dataset = raw.Dataset
        self.name    = raw.Name


        self.config = Config(
            days              = raw.Days,
            capacity          = raw.Capacity,
            max_trip_distance = raw.MaxDistance,
            vehicle_cost      = raw.VehicleCost,
            vehicle_day_cost  = raw.VehicleDayCost,
            distance_cost     = raw.DistanceCost,
        )

        self.tools = [
            Tool(
                id            = t.ID,
                size          = t.weight,
                num_available = t.amount,
                cost          = t.cost,
            )
            for t in raw.Tools
        ]

        self.requests = [
            Request(
                id            = r.ID,
                location_id   = r.node,
                earliest      = r.fromDay,
                latest        = r.toDay,
                duration      = r.numDays,
                machine_type  = r.tool,
                num_machines  = r.toolCount,
            )
            for r in raw.Requests
        ]

        self.coordinates = [
            (c.X, c.Y) for c in raw.Coordinates
        ]
        self.depot = self.coordinates[raw.DepotCoordinate]
        self.depot_id = raw.DepotCoordinate

        raw.calculateDistances()
        if raw.ReadDistance is not None:
            self.distance = raw.ReadDistance
        elif raw.calcDistance is not None:
            self.distance = raw.calcDistance
        else:
            raise ValueError("Could not obtain distance matrix — no coordinates to compute from")

    def _print(self):
        for k, v in vars(self).items():
            print(f"{k}: {v}")
            
    def get_distance(self, loc_a: int, loc_b: int) -> int:
        return self.distance[loc_a][loc_b]

    def get_distance_from_depot(self, loc: int) -> int:
        return self.distance[self.depot_id][loc]