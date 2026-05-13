from instance import Instance
from .solver import VehicleRoute, Stop


def _single_trip_bring_end(stops: list, n: int, req_lookup: dict) -> tuple:
    """Return (bring, end) depot-visit vectors for one trip.

    bring[i] < 0 means |bring[i]| tools of kind i were loaded from the depot.
    end[i] > 0 means end[i] tools of kind i were returned to the depot.
    """
    if not stops:
        return [0] * n, [0] * n
    current = [0] * n
    node_visits = []
    for stop in stops:
        req = req_lookup[stop.request_id]
        idx = req.machine_type - 1
        if stop.action == 'delivery':
            current[idx] -= req.num_machines
        else:
            current[idx] += req.num_machines
        node_visits.append(list(current))
    bring = [0] * n
    for nv in node_visits:
        bring = [min(a, b) for a, b in zip(bring, nv)]
    end = [b - a for a, b in zip(bring, node_visits[-1])]
    return bring, end


def _compute_depot_visits(route: VehicleRoute, instance: Instance, req_lookup: dict) -> list:
    n = len(instance.tools)
    trips = route.trips if route.trips else [route.stops]
    trip_bes = [_single_trip_bring_end(trip, n, req_lookup) for trip in trips]

    # First depot visit: load tools for trip 1
    v_lines = [trip_bes[0][0]]
    # Intermediate depot visits: return from previous trip + load for next trip
    for i in range(1, len(trip_bes)):
        prev_end = trip_bes[i - 1][1]
        curr_bring = trip_bes[i][0]
        v_lines.append([prev_end[j] + curr_bring[j] for j in range(n)])
    # Final depot visit: return tools from last trip
    v_lines.append(trip_bes[-1][1])
    return v_lines


def _day_aggregates(vehicles: list, instance: Instance, req_lookup: dict) -> tuple:
    n = len(instance.tools)
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


def _compute_depot_inventories(route_set: dict, instance: Instance, req_lookup: dict) -> tuple:
    n = len(instance.tools)

    day_deltas = {}
    for day in sorted(route_set):
        cs, cf = _day_aggregates(route_set[day], instance, req_lookup)
        day_deltas[day] = (cs, cf)

    # Replicates Validate._calculateSolution: peak measured after calcStartDepot
    # (morning deliveries) but before calcFinishDepot (evening pickups).
    tool_status = [0] * n
    tool_use = [0] * n
    for day in sorted(day_deltas):
        cs, cf = day_deltas[day]
        tool_status = [a + b for a, b in zip(tool_status, cs)]
        tool_use = [max(-a, b) for a, b in zip(tool_status, tool_use)]
        tool_status = [a + b for a, b in zip(tool_status, cf)]

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


def routing_stats(route_set: dict, instance: Instance) -> dict:
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


def cost_from_routes(route_set: dict, instance: Instance) -> dict:
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


def read_solution(path: str, instance: Instance) -> tuple:
    from scheduling.state import build_state, commit_request

    req_by_id = {r.id: r for r in instance.requests}
    tool_by_type = {t.id: t for t in instance.tools}

    delivery_days = {}
    route_set = {}

    current_day = None
    veh_stops = {}
    veh_trips = {}
    veh_dist = {}

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
                    trips=veh_trips.get(vnum, []),
                ))
        route_set[current_day] = routes

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            if line.startswith('DAY ='):
                _finalise_day()
                current_day = int(line.split('=', 1)[1].strip())
                veh_stops = {}
                veh_trips = {}
                veh_dist = {}
            elif '\tR\t' in line:
                parts = line.split('\t')
                vnum = int(parts[0])
                tokens = [int(t) for t in parts[2:]]

                trips_stops = []
                current_trip = []
                for tok in tokens:
                    if tok == 0:
                        if current_trip:
                            trips_stops.append(current_trip)
                            current_trip = []
                    else:
                        rid = tok
                        if rid > 0:
                            req = req_by_id[rid]
                            delivery_days[rid] = current_day
                            stop = Stop(
                                request_id=rid, action='delivery',
                                location_id=req.location_id,
                                load=req.num_machines * tool_by_type[req.machine_type].size,
                                machine_type=req.machine_type,
                            )
                        else:
                            req = req_by_id[-rid]
                            stop = Stop(
                                request_id=-rid, action='pickup',
                                location_id=req.location_id,
                                load=req.num_machines * tool_by_type[req.machine_type].size,
                                machine_type=req.machine_type,
                            )
                        current_trip.append(stop)

                all_stops = [s for trip in trips_stops for s in trip]
                veh_stops[vnum] = all_stops
                if len(trips_stops) > 1:
                    veh_trips[vnum] = trips_stops

            elif '\tD\t' in line:
                parts = line.split('\t')
                veh_dist[int(parts[0])] = int(parts[2])

    _finalise_day()

    state = build_state(instance)
    for req_id, day in delivery_days.items():
        commit_request(state, instance, req_by_id[req_id], day)

    return state, route_set


def write_solution(route_set: dict, instance: Instance, output_path: str) -> None:
    req_lookup = {r.id: r for r in instance.requests}
    stats = routing_stats(route_set, instance)
    tool_use, start_depots, finish_depots = _compute_depot_inventories(
        route_set, instance, req_lookup
    )

    veh_cost = stats['max_vehicles'] * instance.config.vehicle_cost
    veh_day_cost = stats['vehicle_days'] * instance.config.vehicle_day_cost
    dist_cost = stats['total_distance'] * instance.config.distance_cost
    tool_cost = sum(tool_use[i] * instance.tools[i].cost for i in range(len(instance.tools)))
    total_cost = veh_cost + veh_day_cost + dist_cost + tool_cost

    with open(output_path, 'w') as f:
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
                if route.trips:
                    tokens = [0]
                    for trip in route.trips:
                        for stop in trip:
                            tokens.append(stop.request_id if stop.action == 'delivery' else -stop.request_id)
                        tokens.append(0)
                    r_line = '\t'.join(str(x) for x in [veh_num, 'R'] + tokens)
                else:
                    route_ids = [stop.request_id if stop.action == 'delivery' else -stop.request_id
                                 for stop in route.stops]
                    r_line = '\t'.join(str(x) for x in [veh_num, 'R', 0] + route_ids + [0])
                f.write(r_line + '\n')

                depot_visits = _compute_depot_visits(route, instance, req_lookup)
                for visit_num, dv in enumerate(depot_visits, start=1):
                    v_line = '\t'.join(str(x) for x in [veh_num, 'V', visit_num] + dv)
                    f.write(v_line + '\n')

                f.write(f'{veh_num}\tD\t{route.distance}\n')

            f.write('\n')
