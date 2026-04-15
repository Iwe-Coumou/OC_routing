import math
from collections import defaultdict
from instance import Instance


def compute_tool_cost(state: dict, instance: Instance) -> int:
    """Peak concurrent tool usage across the horizon, multiplied by per-tool cost."""
    tool_by_type = {t.id: t for t in instance.tools}
    tool_cost = 0
    for machine_type, diff in state['loans'].items():
        peak = 0
        current = 0
        for delta in diff:
            current += delta
            peak = max(peak, current)
        tool_cost += peak * tool_by_type[machine_type].cost
    return tool_cost


def day_distance_score(locs: list[int], instance: Instance) -> float:
    """Naive distance proxy for a set of locations on one day: mean distance to depot."""
    if not locs:
        return 0.0
    return sum(instance.get_distance_from_depot(l) for l in locs) / len(locs)


def estimate_vehicles_and_distance(state: dict, instance: Instance) -> tuple[dict, float]:
    """Return (vehicles_per_day, total_distance_score) estimated without actual routes."""
    tool_by_type = {t.id: t for t in instance.tools}

    load_per_day = defaultdict(int)
    locs_per_day = defaultdict(set)

    for e in state['scheduled']:
        req = e['request']
        load = req.num_machines * tool_by_type[req.machine_type].size
        d_day = e['delivery_day']
        p_day = e['pickup_day']
        loc = req.location_id

        locs_per_day[d_day].add(loc)
        locs_per_day[p_day].add(loc)

        load_per_day[d_day] += load
        load_per_day[p_day] += load

    cap = instance.config.capacity
    vehicles_per_day = {d: math.ceil(load / cap) for d, load in load_per_day.items()}
    total_distance = sum(day_distance_score(list(locs_per_day[d]), instance) for d in vehicles_per_day)
    return vehicles_per_day, total_distance


def cost_breakdown(state: dict, instance: Instance) -> dict:
    """Return each cost component as a dict."""
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


def compute_cost_estimate(state: dict, instance: Instance) -> float:
    """Total estimated cost."""
    return cost_breakdown(state, instance)['total']


def print_cost(breakdown: dict, label: str = '') -> None:
    """Print a formatted cost breakdown dict (from cost_breakdown())."""
    b = breakdown
    prefix = f"{label}: " if label else ''
    print(
        f"{prefix}total={b['total']:>14,.0f}  "
        f"| tools={b['tool']:>12,.0f} "
        f"| vehicles={b['vehicle']:>10,.0f} ({b['max_vehicles']} max) "
        f"| veh-days={b['vehicle_days']:>10,.0f} ({b['vehicle_days_count']} routes) "
        f"| distance={b['distance']:>12,.0f}"
    )
