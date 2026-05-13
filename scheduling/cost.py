import math
from collections import defaultdict
from instance import Instance


def compute_tool_cost(state: dict, instance: Instance) -> int:
    tool_by_type = {t.id: t for t in instance.tools}
    tool_cost = 0
    for machine_type, diff in state['loans'].items():
        pickups = state['pickups_per_day'][machine_type]
        peak = 0
        current = 0
        for day, delta in enumerate(diff):
            current += delta
            peak = max(peak, current + pickups[day])
        tool_cost += peak * tool_by_type[machine_type].cost
    return tool_cost


def day_tour_estimate(locs: list[int], instance: Instance) -> float:
    seen = set()
    unvisited = []
    for loc in locs:
        if loc not in seen:
            seen.add(loc)
            unvisited.append(loc)
    if not unvisited:
        return 0.0
    tour, current = 0, 0  # start at depot (location id 0)
    while unvisited:
        nearest = min(unvisited, key=lambda loc: instance.get_distance(current, loc))
        tour += instance.get_distance(current, nearest)
        current = nearest
        unvisited.remove(nearest)
    return float(tour + instance.get_distance(current, 0))  # return to depot


def estimate_vehicles_and_distance(state: dict, instance: Instance) -> tuple[dict, float]:
    tool_by_type = {t.id: t for t in instance.tools}

    load_per_day = defaultdict(int)
    locs_per_day = defaultdict(list)

    for e in state['scheduled']:
        req = e['request']
        load = req.num_machines * tool_by_type[req.machine_type].size
        d_day = e['delivery_day']
        p_day = e['pickup_day']
        loc = req.location_id

        locs_per_day[d_day].append(loc)
        locs_per_day[p_day].append(loc)

        load_per_day[d_day] += load
        load_per_day[p_day] += load

    cap = instance.config.capacity
    vehicles_per_day = {d: math.ceil(load / cap) for d, load in load_per_day.items()}
    total_distance = sum(day_tour_estimate(locs_per_day[d], instance) for d in vehicles_per_day)
    return vehicles_per_day, total_distance


def cost_breakdown(state: dict, instance: Instance) -> dict:
    tool_cost = compute_tool_cost(state, instance)

    vehicles_per_day, total_distance = estimate_vehicles_and_distance(state, instance)
    max_vehicles       = max(vehicles_per_day.values(), default=0)
    total_vehicle_days = sum(vehicles_per_day.values())

    vehicle_cost   = instance.config.vehicle_cost     * max_vehicles
    vehicle_d_cost = instance.config.vehicle_day_cost * total_vehicle_days
    distance_cost  = instance.config.distance_cost    * total_distance
    total          = tool_cost + vehicle_cost + vehicle_d_cost + distance_cost

    return {
        'tool':               tool_cost,
        'vehicle':            vehicle_cost,
        'vehicle_days':       vehicle_d_cost,
        'distance':           distance_cost,
        'total':              total,
        'max_vehicles':       max_vehicles,
        'vehicle_days_count': total_vehicle_days,
    }


def print_cost(breakdown: dict, label: str = '') -> None:
    b = breakdown
    prefix = f"{label}: " if label else ''
    print(
        f"{prefix}total={b['total']:>12.3e}  "
        f"| tools={b['tool']:>12,} "
        f"| vehicles={b['vehicle']:>10,} ({b['max_vehicles']} max) "
        f"| veh-days={b['vehicle_days']:>10,} ({b['vehicle_days_count']} routes) "
        f"| distance={b['distance']:>12,}"
    )
