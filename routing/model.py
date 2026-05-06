try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except ImportError as exc:
    raise ImportError(
        "Google OR-Tools is required for routing. Install it with: pip install ortools"
    ) from exc
import math
from tqdm import tqdm

from instance import Instance
from .routes import Stop, VehicleRoute


def _compute_route_distance(stops: list, instance: Instance) -> int:
    """Total distance for a single vehicle route: depot -> stops -> depot."""
    if not stops:
        return 0
    locs = [0] + [s.location_id for s in stops] + [0]
    return sum(instance.get_distance(locs[i], locs[i + 1]) for i in range(len(locs) - 1))


def _build_and_solve(stops: list, instance: Instance, num_vehicles: int,
                     time_limit_seconds: int, fast: bool,
                     initial_routes: list = None):
    """Build and solve an OR-Tools CVRP model for one day.

    Returns list[VehicleRoute] on success, None if OR-Tools finds no solution.
    """
    capacity = instance.config.capacity
    num_nodes = len(stops) + 1
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Precompute distance matrices once (avoids n² Python→C++ callbacks per solve).
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
        # Better initial solution → GLS converges faster
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.guided_local_search_lambda_coefficient = 0.1
        search_params.time_limit.seconds = time_limit_seconds

    # Warm-start GLS from a previously known good solution (fast=False only).
    if initial_routes and not fast:
        stop_to_node = {(s.request_id, s.action): i + 1 for i, s in enumerate(stops)}
        routes_as_nodes = []
        for route in initial_routes:
            nodes = [stop_to_node[k]
                     for stop in route.stops
                     if (k := (stop.request_id, stop.action)) in stop_to_node]
            routes_as_nodes.append(nodes)
        while len(routes_as_nodes) < num_vehicles:
            routes_as_nodes.append([])
        try:
            hint = routing.ReadAssignmentFromRoutes(routes_as_nodes, True)
            solution = routing.SolveFromAssignmentWithParameters(hint, search_params) if hint else \
                       routing.SolveWithParameters(search_params)
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
    """Solve CVRP for a single day.

    Tries a tight fleet-size bound first (based on load / capacity), then falls
    back to one vehicle per stop if OR-Tools finds the tighter model infeasible
    (e.g. when the max_trip_distance constraint forces single-stop routes).

    Args:
        day: Day number (used only for logging).
        stops: list[Stop] for this day.
        instance: Problem instance.
        time_limit_seconds: OR-Tools solver time budget per day (ignored when fast=True).
        fast: If True, use PATH_CHEAPEST_ARC only — completes in seconds per day.
        initial_routes: Optional list[VehicleRoute] to warm-start GLS (ignored when fast=True).

    Returns:
        list[VehicleRoute] — only non-empty routes included.
    """
    if not stops:
        return []

    capacity = instance.config.capacity

    delivery_load = sum(s.load for s in stops if s.action == 'delivery')
    pickup_load   = sum(s.load for s in stops if s.action == 'pickup')
    min_cap = max(1, math.ceil(max(delivery_load, pickup_load) / capacity))
    tight_n = min(len(stops), max(min_cap + 3, min_cap * 2))

    # Ensure the model has enough vehicles to accommodate the warm-start hint.
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
    """Solve CVRP independently for every day that has stops.

    Uses a ThreadPoolExecutor for parallel solving when fast=True (OR-Tools C++
    solver releases the GIL during SolveWithParameters). Falls back to sequential
    for fast=False GLS solves to avoid thread safety concerns.

    Args:
        daily_stops: dict[int, list[Stop]] from tasks.build_daily_stops().
        instance: Problem instance.
        time_limit_seconds: Solver time budget per day (ignored when fast=True).
        fast: If True, use PATH_CHEAPEST_ARC only (no GLS, no time limit).
        initial_routes: Optional RouteSet to warm-start GLS per day (ignored when fast=True).

    Returns:
        RouteSet: dict[int, list[VehicleRoute]]
    """
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
