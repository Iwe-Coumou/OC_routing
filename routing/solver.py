try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except ImportError as exc:
    raise ImportError(
        "Google OR-Tools is required for routing. Install it with: pip install ortools"
    ) from exc
import math
from dataclasses import dataclass, field
from tqdm import tqdm

from collections import defaultdict
from instance import Instance


@dataclass
class Stop:
    request_id: int
    action: str        # 'delivery' or 'pickup'
    location_id: int
    load: int
    machine_type: int


@dataclass
class VehicleRoute:
    vehicle_id: int
    stops: list = field(default_factory=list)
    distance: int = 0


def _tasks_by_day(state: dict, instance: Instance) -> dict:
    tool_by_type = {t.id: t for t in instance.tools}
    result = defaultdict(list)
    for e in state['scheduled']:
        req = e['request']
        load = req.num_machines * tool_by_type[req.machine_type].size
        result[e['delivery_day']].append({
            'type': 'delivery', 'request_id': req.id,
            'location': req.location_id, 'load': load,
        })
        result[e['pickup_day']].append({
            'type': 'pickup', 'request_id': req.id,
            'location': req.location_id, 'load': load,
        })
    return result


def build_daily_stops(state: dict, instance: Instance) -> dict:
    req_lookup = {r.id: r for r in instance.requests}
    raw = _tasks_by_day(state, instance)
    result = {}
    for day, tasks in raw.items():
        if not tasks:
            continue
        stops = [
            Stop(
                request_id=t['request_id'],
                action=t['type'],
                location_id=t['location'],
                load=t['load'],
                machine_type=req_lookup[t['request_id']].machine_type,
            )
            for t in tasks
        ]
        result[day] = stops
    return result


def _compute_route_distance(stops, instance):
    if not stops:
        return 0
    locs = [0] + [s.location_id for s in stops] + [0]
    return sum(instance.get_distance(locs[i], locs[i + 1]) for i in range(len(locs) - 1))


def _build_and_solve(stops: list, instance: Instance, num_vehicles: int,
                     time_limit_seconds: int, fast: bool,
                     initial_routes: list = None):
    capacity = instance.config.capacity
    num_nodes = len(stops) + 1
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    loc_ids = [0] + [s.location_id for s in stops]
    n = len(loc_ids)
    d_cost = instance.config.distance_cost
    scaled_matrix = [
        [instance.get_distance(loc_ids[i], loc_ids[j]) * d_cost for j in range(n)]
        for i in range(n)
    ]
    raw_matrix = [
        [instance.get_distance(loc_ids[i], loc_ids[j]) for j in range(n)]
        for i in range(n)
    ]

    scaled_transit_index = routing.RegisterTransitMatrix(scaled_matrix)
    raw_transit_index    = routing.RegisterTransitMatrix(raw_matrix)

    routing.SetArcCostEvaluatorOfAllVehicles(scaled_transit_index)
    routing.SetFixedCostOfAllVehicles(instance.config.vehicle_day_cost)

    types_present = {s.machine_type for s in stops}
    type_dims = {}
    solver = routing.solver()

    for tool in instance.tools:
        if tool.id not in types_present:
            continue

        def make_demand_cb(t):
            def cb(from_index):
                node = manager.IndexToNode(from_index)
                if node == 0:
                    return 0
                s = stops[node - 1]
                if s.machine_type != t:
                    return 0
                return -s.load if s.action == 'delivery' else s.load
            return cb

        cb_idx = routing.RegisterUnaryTransitCallback(make_demand_cb(tool.id))
        dim_name = f'Cap_{tool.id}'
        routing.AddDimensionWithVehicleCapacity(
            cb_idx, 0, [capacity] * num_vehicles, False, dim_name,
        )
        type_dims[tool.id] = routing.GetDimensionOrDie(dim_name)

    all_dims = list(type_dims.values())
    if len(all_dims) > 1:
        for node in range(1, num_nodes):
            idx = manager.NodeToIndex(node)
            solver.Add(solver.Sum([d.CumulVar(idx) for d in all_dims]) <= capacity)
        for v in range(num_vehicles):
            for idx in [routing.Start(v), routing.End(v)]:
                solver.Add(solver.Sum([d.CumulVar(idx) for d in all_dims]) <= capacity)

    routing.AddDimension(
        raw_transit_index, 0, instance.config.max_trip_distance, True, 'Distance',
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    if fast:
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS
        )
    else:
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.guided_local_search_lambda_coefficient = 0.1
        search_params.time_limit.seconds = time_limit_seconds

    if initial_routes and not fast:
        stop_to_node = {(s.request_id, s.action): i + 1 for i, s in enumerate(stops)}
        routes_as_nodes = []
        for route in initial_routes:
            nodes = []
            for stop in route.stops:
                k = (stop.request_id, stop.action)
                if k in stop_to_node:
                    nodes.append(stop_to_node[k])
            routes_as_nodes.append(nodes)
        while len(routes_as_nodes) < num_vehicles:
            routes_as_nodes.append([])
        try:
            hint = routing.ReadAssignmentFromRoutes(routes_as_nodes, True)
            if hint:
                solution = routing.SolveFromAssignmentWithParameters(hint, search_params)
            else:
                solution = routing.SolveWithParameters(search_params)
        except Exception:
            solution = routing.SolveWithParameters(search_params)
    else:
        solution = routing.SolveWithParameters(search_params)

    if solution is None:
        return None

    vehicle_routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route_stops = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route_stops.append(stops[node - 1])
            index = solution.Value(routing.NextVar(index))
        if route_stops:
            dist = _compute_route_distance(route_stops, instance)
            vehicle_routes.append(VehicleRoute(vehicle_id=vehicle_id, stops=route_stops, distance=dist))

    return vehicle_routes


def solve_day(
    day: int,
    stops: list,
    instance: Instance,
    time_limit_seconds: int = 30,
    fast: bool = False,
    initial_routes: list = None,
) -> list:
    if not stops:
        return []

    capacity = instance.config.capacity
    delivery_load = sum(s.load for s in stops if s.action == 'delivery')
    pickup_load   = sum(s.load for s in stops if s.action == 'pickup')
    min_cap = max(1, math.ceil(max(delivery_load, pickup_load) / capacity))
    tight_n = min(len(stops), max(min_cap + 3, min_cap * 2))

    if initial_routes and len(initial_routes) > tight_n:
        tight_n = min(len(stops), len(initial_routes))

    result = _build_and_solve(stops, instance, tight_n, time_limit_seconds, fast, initial_routes)
    if result is None and tight_n < len(stops):
        result = _build_and_solve(stops, instance, len(stops), time_limit_seconds, fast)

    return result or []


def solve_all_days(
    daily_stops: dict,
    instance: Instance,
    time_limit_seconds: int = 30,
    fast: bool = False,
    initial_routes: dict = None,
) -> dict:
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    days = sorted(daily_stops)
    route_set = {}

    if fast:
        max_workers = min(len(days), os.cpu_count() or 4)
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for day in days:
                f = pool.submit(solve_day, day, daily_stops[day], instance,
                                time_limit_seconds, True, None)
                futures[f] = day
            for f in as_completed(futures):
                day = futures[f]
                routes = f.result()
                if routes:
                    route_set[day] = routes
    else:
        with tqdm(days, desc='Routing', unit='day', leave=True) as bar:
            for day in bar:
                stops = daily_stops[day]
                bar.set_postfix(day=day, stops=len(stops))
                day_hint = initial_routes.get(day) if initial_routes else None
                routes = solve_day(day, stops, instance, time_limit_seconds,
                                   fast=False, initial_routes=day_hint)
                if routes:
                    route_set[day] = routes

    return route_set


def solve_routing(
    state: dict,
    instance: Instance,
    time_limit_seconds: int = 30,
    fast: bool = False,
    initial_routes: dict = None,
) -> dict:
    daily_stops = build_daily_stops(state, instance)
    return solve_all_days(daily_stops, instance, time_limit_seconds,
                          fast=fast, initial_routes=initial_routes)


def solve_routing_incremental(
    state: dict,
    instance: Instance,
    changed_days: set,
    current_routes: dict,
) -> dict:
    daily_stops = build_daily_stops(state, instance)

    route_set = {day: routes for day, routes in current_routes.items()
                 if day not in changed_days}
    route_set = {day: routes for day, routes in route_set.items()
                 if day in daily_stops}

    dirty_stops = {day: daily_stops[day] for day in changed_days if day in daily_stops}
    if dirty_stops:
        new_routes = solve_all_days(dirty_stops, instance, fast=True)
        route_set.update(new_routes)

    return route_set
