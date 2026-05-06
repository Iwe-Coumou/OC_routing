"""Write a VeRoLog-format solution file from a RouteSet.

The format is validated by Validate.py (teacher's tool). The V-line computation
replicates the same logic used in Validate._calculateSolution so that the
given and calculated values always agree.
"""

from instance import Instance
from .routes import VehicleRoute, Stop


# ---------------------------------------------------------------------------
# V-line computation
# ---------------------------------------------------------------------------

def _compute_depot_visits(route: VehicleRoute, instance: Instance, req_lookup: dict) -> list:
    """Compute the depotVisits list for a single-trip vehicle route.

    Replicates the validator's depotVisits logic for a route of the form
    depot -> stops... -> depot (exactly one trip).

    Returns a list of two tool-delta vectors:
        [0] = what the vehicle loaded at the start depot (negative values)
        [1] = what the vehicle unloaded at the end depot (positive values)
    """
    n = len(instance.tools)
    current = [0] * n
    node_visits = []

    for stop in route.stops:
        req = req_lookup[stop.request_id]
        idx = req.machine_type - 1
        if stop.action == 'delivery':
            current[idx] -= req.num_machines
        else:
            current[idx] += req.num_machines
        node_visits.append(list(current))

    if not node_visits:
        return [[0] * n, [0] * n]

    # Replicate validator: compute bringTools and sumTools over all node visits
    bring = [0] * n
    for nv in node_visits:
        bring = [min(a, b) for a, b in zip(bring, nv)]

    # depotVisit[0] = bringTools added to initial depotVisits[-1] (which is [0]*n)
    depot_start = list(bring)
    # depotVisit[1] = nodeVisits[-1] - bringTools
    depot_end = [b - a for a, b in zip(bring, node_visits[-1])]

    return [depot_start, depot_end]


# ---------------------------------------------------------------------------
# Per-day aggregates
# ---------------------------------------------------------------------------

def _day_aggregates(vehicles: list, instance: Instance, req_lookup: dict) -> tuple:
    """Compute (calcStartDepot, calcFinishDepot) for a day.

    Follows the same accumulation loop as Validate._calculateSolution.

    Returns:
        (calc_start, calc_finish): each is a list[int] of length n_tools,
        representing net tool change for the day (delta, not absolute).
    """
    n = len(instance.tools)
    tool_size = [t.size for t in instance.tools]
    calc_start = [0] * n
    calc_finish = [0] * n

    for vehicle in vehicles:
        depot_visits = _compute_depot_visits(vehicle, instance, req_lookup)

        visit_total = [0] * n
        total_used_at_start = [0] * n
        for visit in depot_visits:
            visit_total = [a + b for a, b in zip(visit, visit_total)]
            total_used_at_start = [b - min(0, a) for a, b in zip(visit_total, total_used_at_start)]
            visit_total = [max(0, a) for a in visit_total]

        calc_start = [b - a for a, b in zip(total_used_at_start, calc_start)]
        calc_finish = [a + b for a, b in zip(visit_total, calc_finish)]

    return calc_start, calc_finish


# ---------------------------------------------------------------------------
# Absolute depot inventory
# ---------------------------------------------------------------------------

def _compute_depot_inventories(route_set: dict, instance: Instance, req_lookup: dict) -> tuple:
    """Return (tool_use, start_depots, finish_depots).

    tool_use: list[int] — peak concurrent tool loans per type (for header),
              computed from routes using the same algorithm as Validate.py
              _calculateSolution.
    start_depots: dict[int, list[int]] — absolute depot inventory at start of each day
    finish_depots: dict[int, list[int]] — absolute depot inventory at end of each day
    """
    n = len(instance.tools)

    # First pass: compute delta aggregates per day
    day_deltas = {}
    for day in sorted(route_set):
        cs, cf = _day_aggregates(route_set[day], instance, req_lookup)
        day_deltas[day] = (cs, cf)

    # Compute running toolStatus and toolUse (peak concurrent loans).
    # Replicates Validate._calculateSolution: peak measured after calcStartDepot
    # (morning deliveries) but before calcFinishDepot (evening pickups).
    tool_status = [0] * n
    tool_use = [0] * n
    for day in sorted(day_deltas):
        cs, cf = day_deltas[day]
        tool_status = [a + b for a, b in zip(tool_status, cs)]
        tool_use = [max(-a, b) for a, b in zip(tool_status, tool_use)]
        tool_status = [a + b for a, b in zip(tool_status, cf)]

    # Second pass: compute absolute depot inventories
    tool_status = list(tool_use)
    start_depots = {}
    finish_depots = {}
    for day in sorted(day_deltas):
        cs, cf = day_deltas[day]
        tool_status = [a + b for a, b in zip(tool_status, cs)]
        start_depots[day] = list(tool_status)
        tool_status = [a + b for a, b in zip(tool_status, cf)]
        finish_depots[day] = list(tool_status)

    return tool_use, start_depots, finish_depots


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def routing_stats(route_set: dict, instance: Instance) -> dict:
    """Return routing statistics for the current RouteSet.

    Returns:
        dict with keys: max_vehicles, vehicle_days, total_distance
    """
    max_vehicles = 0
    vehicle_days = 0
    total_distance = 0
    for routes in route_set.values():
        max_vehicles = max(max_vehicles, len(routes))
        vehicle_days += len(routes)
        total_distance += sum(r.distance for r in routes)
    return {
        'max_vehicles': max_vehicles,
        'vehicle_days': vehicle_days,
        'total_distance': total_distance,
    }


# ---------------------------------------------------------------------------
# Cost breakdown from routes
# ---------------------------------------------------------------------------

def cost_from_routes(route_set: dict, instance: Instance) -> dict:
    """Compute the full cost breakdown from routes, matching the validator exactly.

    Tool cost is derived from the route V-lines (same logic as Validate.py),
    not from the schedule state, so this value agrees with the COST line in
    the solution file.
    """
    req_lookup = {r.id: r for r in instance.requests}
    stats = routing_stats(route_set, instance)
    tool_use, _, _ = _compute_depot_inventories(route_set, instance, req_lookup)

    tool_cost     = sum(tool_use[i] * instance.tools[i].cost for i in range(len(instance.tools)))
    vehicle_cost  = stats['max_vehicles']  * instance.config.vehicle_cost
    veh_day_cost  = stats['vehicle_days']  * instance.config.vehicle_day_cost
    distance_cost = stats['total_distance'] * instance.config.distance_cost
    total         = tool_cost + vehicle_cost + veh_day_cost + distance_cost

    return {
        'tool':               tool_cost,
        'vehicle':            vehicle_cost,
        'vehicle_days':       veh_day_cost,
        'distance':           distance_cost,
        'total':              total,
        'max_vehicles':       stats['max_vehicles'],
        'vehicle_days_count': stats['vehicle_days'],
    }


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

def read_solution(path: str, instance: Instance) -> tuple[dict, dict]:
    """Parse a saved VeRoLog solution file and reconstruct (state, route_set).

    Reads the format written by write_solution(). Returns a fully committed
    scheduling state and a RouteSet, ready to use as a warm start for route_lns().
    Uses a local import of scheduling.state to avoid circular imports at module level.
    """
    from scheduling.state import build_state, commit_request

    req_by_id = {r.id: r for r in instance.requests}
    tool_by_type = {t.id: t for t in instance.tools}

    delivery_days: dict[int, int] = {}   # req_id -> delivery_day
    route_set: dict[int, list] = {}

    current_day = None
    veh_stops: dict[int, list] = {}
    veh_dist: dict[int, int] = {}

    def _finalise_day():
        if current_day is None:
            return
        routes = []
        for vnum in sorted(veh_stops):
            stops = veh_stops[vnum]
            if stops:
                routes.append(VehicleRoute(
                    vehicle_id=vnum,
                    stops=stops,
                    distance=veh_dist.get(vnum, 0),
                ))
        route_set[current_day] = routes

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            if line.startswith('DAY ='):
                _finalise_day()
                current_day = int(line.split('=', 1)[1].strip())
                veh_stops = {}
                veh_dist = {}
            elif '\tR\t' in line:
                parts = line.split('\t')
                vnum = int(parts[0])
                # format: vnum  R  0  [+delivery_id / -pickup_id ...]  0
                tokens = parts[3:-1]
                stops = []
                for tok in tokens:
                    rid = int(tok)
                    if rid > 0:
                        req = req_by_id[rid]
                        delivery_days[rid] = current_day
                        stops.append(Stop(
                            request_id=rid,
                            action='delivery',
                            location_id=req.location_id,
                            load=req.num_machines * tool_by_type[req.machine_type].size,
                            machine_type=req.machine_type,
                        ))
                    elif rid < 0:
                        req = req_by_id[-rid]
                        stops.append(Stop(
                            request_id=-rid,
                            action='pickup',
                            location_id=req.location_id,
                            load=req.num_machines * tool_by_type[req.machine_type].size,
                            machine_type=req.machine_type,
                        ))
                veh_stops[vnum] = stops
            elif '\tD\t' in line:
                parts = line.split('\t')
                veh_dist[int(parts[0])] = int(parts[2])

    _finalise_day()

    state = build_state(instance)
    for req_id, day in delivery_days.items():
        commit_request(state, instance, req_by_id[req_id], day)

    return state, route_set


def write_solution(route_set: dict, instance: Instance, output_path: str) -> None:
    """Write a VeRoLog-format solution file.

    Args:
        route_set: RouteSet from solve_routing().
        instance: Problem instance.
        output_path: Destination file path.
    """
    req_lookup = {r.id: r for r in instance.requests}
    stats = routing_stats(route_set, instance)
    tool_use, start_depots, finish_depots = _compute_depot_inventories(
        route_set, instance, req_lookup
    )

    # Compute total cost
    veh_cost = stats['max_vehicles'] * instance.config.vehicle_cost
    veh_day_cost = stats['vehicle_days'] * instance.config.vehicle_day_cost
    dist_cost = stats['total_distance'] * instance.config.distance_cost
    tool_cost = sum(tool_use[i] * instance.tools[i].cost for i in range(len(instance.tools)))
    total_cost = veh_cost + veh_day_cost + dist_cost + tool_cost

    with open(output_path, 'w') as f:
        # Header
        f.write(f'DATASET = VeRoLog solver challenge 2017\n')
        f.write(f'NAME = {instance.name}\n')
        f.write('\n')
        f.write(f'MAX_NUMBER_OF_VEHICLES = {stats["max_vehicles"]}\n')
        f.write(f'NUMBER_OF_VEHICLE_DAYS = {stats["vehicle_days"]}\n')
        f.write(f'TOOL_USE = {" ".join(str(x) for x in tool_use)}\n')
        f.write(f'DISTANCE = {stats["total_distance"]}\n')
        f.write(f'COST = {total_cost}\n')
        f.write('\n')

        for day in sorted(route_set):
            routes = route_set[day]
            f.write(f'DAY = {day}\n')
            f.write(f'NUMBER_OF_VEHICLES = {len(routes)}\n')
            f.write(f'START_DEPOT = {" ".join(str(x) for x in start_depots[day])}\n')
            f.write(f'FINISH_DEPOT = {" ".join(str(x) for x in finish_depots[day])}\n')

            for veh_num, route in enumerate(routes, start=1):
                # R line: depot, +req_id for delivery, -req_id for pickup, depot
                route_ids = []
                for stop in route.stops:
                    route_ids.append(stop.request_id if stop.action == 'delivery' else -stop.request_id)
                r_line = '\t'.join(str(x) for x in [veh_num, 'R', 0] + route_ids + [0])
                f.write(r_line + '\n')

                # V lines
                depot_visits = _compute_depot_visits(route, instance, req_lookup)
                for visit_num, dv in enumerate(depot_visits, start=1):
                    v_line = '\t'.join(str(x) for x in [veh_num, 'V', visit_num] + dv)
                    f.write(v_line + '\n')

                # D line
                f.write(f'{veh_num}\tD\t{route.distance}\n')

            f.write('\n')
