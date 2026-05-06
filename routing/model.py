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
                     time_limit_seconds: int, fast: bool):
    """Build and solve an OR-Tools CVRP model for one day.

    Returns list[VehicleRoute] on success, None if OR-Tools finds no solution.
    """
    capacity = instance.config.capacity
    num_nodes = len(stops) + 1
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def _loc(node):
        return 0 if node == 0 else stops[node - 1].location_id

    def scaled_distance_callback(from_index, to_index):
        return (instance.get_distance(_loc(manager.IndexToNode(from_index)),
                                      _loc(manager.IndexToNode(to_index)))
                * instance.config.distance_cost)

    def raw_distance_callback(from_index, to_index):
        return instance.get_distance(_loc(manager.IndexToNode(from_index)),
                                     _loc(manager.IndexToNode(to_index)))

    scaled_transit_index = routing.RegisterTransitCallback(scaled_distance_callback)
    raw_transit_index    = routing.RegisterTransitCallback(raw_distance_callback)

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
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if not fast:
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = time_limit_seconds

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

    Returns:
        list[VehicleRoute] — only non-empty routes included.
    """
    if not stops:
        return []

    capacity = instance.config.capacity

    # Tight fleet bound: need at least enough vehicles to carry all deliveries
    # (start full) or all pickups (end full) — whichever direction binds.
    delivery_load = sum(s.load for s in stops if s.action == 'delivery')
    pickup_load   = sum(s.load for s in stops if s.action == 'pickup')
    min_cap = max(1, math.ceil(max(delivery_load, pickup_load) / capacity))
    # Buffer for distance-constraint splits and routing complexity.
    tight_n = min(len(stops), max(min_cap + 3, min_cap * 2))

    result = _build_and_solve(stops, instance, tight_n, time_limit_seconds, fast)
    if result is None and tight_n < len(stops):
        # Tight bound was infeasible — retry with full fleet
        result = _build_and_solve(stops, instance, len(stops), time_limit_seconds, fast)

    return result or []

    # Node 0 = depot, nodes 1..k = stops
    num_nodes = len(stops) + 1
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Two distance callbacks: one scaled for the objective (monetary units),
    # one unscaled for the max_trip_distance constraint (raw distance units).
    def _loc(node):
        return 0 if node == 0 else stops[node - 1].location_id

    def scaled_distance_callback(from_index, to_index):
        return (instance.get_distance(_loc(manager.IndexToNode(from_index)),
                                      _loc(manager.IndexToNode(to_index)))
                * instance.config.distance_cost)

    def raw_distance_callback(from_index, to_index):
        return instance.get_distance(_loc(manager.IndexToNode(from_index)),
                                     _loc(manager.IndexToNode(to_index)))

    scaled_transit_index = routing.RegisterTransitCallback(scaled_distance_callback)
    raw_transit_index    = routing.RegisterTransitCallback(raw_distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(scaled_transit_index)

    # Fixed cost per vehicle used: vehicle_day_cost (cost of running a route).
    # OR-Tools only pays this for vehicles that serve at least one stop.
    routing.SetFixedCostOfAllVehicles(instance.config.vehicle_day_cost)

    # Per-type capacity dimensions.
    #
    # A single "net load" dimension fails for mixed-type routes because a
    # pickup of type A can cancel a delivery of type B in the cumulative,
    # even though both types are physically on the vehicle simultaneously.
    # Instead we give each tool type its own dimension:
    #   delivery of type t  -> cumul_t decreases (tools dropped off)
    #   pickup   of type t  -> cumul_t increases (tools collected)
    #   stop of other type  -> cumul_t unchanged (demand = 0)
    # fix_start_cumul_to_zero=False: OR-Tools sets each start_cumul_t to the
    # minimum pre-load of type t needed for the chosen stop ordering.
    # cumul_t >= 0 enforces "can't deliver type-t tools you don't have".
    #
    # The joint constraint  sum(cumul_t)  <= capacity  is then added
    # explicitly at every node, correctly enforcing that all tool sizes share
    # the single vehicle capacity regardless of type.

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
            cb_idx,
            0,
            [capacity] * num_vehicles,
            False,
            dim_name,
        )
        type_dims[tool.id] = routing.GetDimensionOrDie(dim_name)

    # Joint constraint: physical load across all types must not exceed capacity.
    all_dims = list(type_dims.values())
    if len(all_dims) > 1:
        for node in range(1, num_nodes):
            idx = manager.NodeToIndex(node)
            solver.Add(solver.Sum([d.CumulVar(idx) for d in all_dims]) <= capacity)
        for v in range(num_vehicles):
            for idx in [routing.Start(v), routing.End(v)]:
                solver.Add(solver.Sum([d.CumulVar(idx) for d in all_dims]) <= capacity)

    # Max trip distance dimension — uses unscaled distances so the limit is
    # correctly compared against instance.config.max_trip_distance (raw units).
    routing.AddDimension(
        raw_transit_index,
        0,
        instance.config.max_trip_distance,
        True,
        'Distance',
    )

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if not fast:
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = time_limit_seconds

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        return []

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


def solve_all_days(
    daily_stops: dict,
    instance: Instance,
    time_limit_seconds: int = 30,
    fast: bool = False,
) -> dict:
    """Solve CVRP independently for every day that has stops.

    Args:
        daily_stops: dict[int, list[Stop]] from tasks.build_daily_stops().
        instance: Problem instance.
        time_limit_seconds: Solver time budget per day (ignored when fast=True).
        fast: If True, use PATH_CHEAPEST_ARC only (no GLS, no time limit).

    Returns:
        RouteSet: dict[int, list[VehicleRoute]]
    """
    route_set = {}
    days = sorted(daily_stops)
    with tqdm(days, desc='Routing', unit='day', disable=fast) as bar:
        for day in bar:
            stops = daily_stops[day]
            if not fast:
                bar.set_postfix(day=day, stops=len(stops))
            routes = solve_day(day, stops, instance, time_limit_seconds, fast=fast)
            if routes:
                route_set[day] = routes
    return route_set
