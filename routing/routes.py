from dataclasses import dataclass, field

# RouteSet: day -> list of non-empty vehicle routes for that day
RouteSet = dict  # dict[int, list[VehicleRoute]]


@dataclass
class Stop:
    request_id: int    # Request.id
    action: str        # 'delivery' or 'pickup'
    location_id: int   # customer location (used for distance lookups)
    load: int          # num_machines * tool.size (vehicle capacity consumed)
    machine_type: int  # tool type id (for per-type capacity dimensions)


@dataclass
class VehicleRoute:
    vehicle_id: int
    stops: list = field(default_factory=list)  # list[Stop]
    distance: int = 0
