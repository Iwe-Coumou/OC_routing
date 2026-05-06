import random
from collections import defaultdict
from instance import Instance, Request


def break_tool_cost(state: dict, instance: Instance) -> list[Request]:
    """Return all requests on loan during the peak tool-cost day.

    Finds the (tool_type, day) pair with the highest weighted concurrent loan
    count, then returns every request on loan that day sorted by cost contribution.
    Returning the full set (not top-k) gives repair maximum freedom to flatten
    the loan peak.
    """
    tool_by_id = {t.id: t for t in instance.tools}

    best_weighted_peak, peak_tool_id, peak_day = 0, None, None
    for t in instance.tools:
        diff = state['loans'].get(t.id, [0] * (instance.config.days + 2))
        pickups = state['pickups_per_day'].get(t.id, [0] * (instance.config.days + 2))
        running = 0
        for day in range(1, instance.config.days + 1):
            running += diff[day]
            concurrent = running + pickups[day]
            weighted = concurrent * t.cost
            if weighted > best_weighted_peak:
                best_weighted_peak = weighted
                peak_tool_id = t.id
                peak_day = day

    if peak_tool_id is None:
        return []

    tool = tool_by_id[peak_tool_id]
    candidates = [
        e for e in state['scheduled']
        if e['request'].machine_type == peak_tool_id
        and e['delivery_day'] <= peak_day < e['pickup_day']
    ]
    candidates.sort(key=lambda e: e['request'].num_machines * tool.cost, reverse=True)
    return [e['request'] for e in candidates]


def _stop_detour(route, stop_idx: int, depot_id: int, instance: Instance) -> int:
    """Marginal distance of stop stop_idx: dist(prev,stop)+dist(stop,next)-dist(prev,next)."""
    stops = route.stops
    prev_loc = stops[stop_idx - 1].location_id if stop_idx > 0 else depot_id
    curr_loc = stops[stop_idx].location_id
    next_loc = stops[stop_idx + 1].location_id if stop_idx < len(stops) - 1 else depot_id
    return (instance.get_distance(prev_loc, curr_loc)
            + instance.get_distance(curr_loc, next_loc)
            - instance.get_distance(prev_loc, next_loc))


def break_vehicle_cost(state: dict, instance: Instance, route_set: dict) -> list[Request]:
    """Return all requests with a stop on the peak vehicle day.

    Clears the entire peak day so repair can freely redistribute its requests
    across other days, enabling structural vehicle-count reduction rather than
    just marginal load shuffling.
    """
    if not route_set:
        return []

    peak_day = max(route_set, key=lambda d: len(route_set[d]))

    tool_by_type = {t.id: t for t in instance.tools}
    req_by_id = {r.id: r for r in instance.requests}

    seen: dict[int, int] = {}  # request_id -> load
    for route in route_set[peak_day]:
        for stop in route.stops:
            if stop.request_id not in seen:
                req = req_by_id[stop.request_id]
                seen[req.id] = req.num_machines * tool_by_type[req.machine_type].size

    candidates = [(load, req_by_id[rid]) for rid, load in seen.items()]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [req for _, req in candidates]


def break_vehicle_day_cost(state: dict, instance: Instance, route_set: dict,
                           k: int | None = None) -> list[Request]:
    """Return requests most responsible for total vehicle-day cost.

    Scores each request by route_distance / n_stops for its delivery and pickup routes.
    A high score means the request rides a sparse route — the strongest driver of
    unnecessary vehicle-days.
    """
    # Map request_id -> {action -> route} from actual routes
    req_route: dict[int, dict] = {}
    for routes in route_set.values():
        for route in routes:
            for stop in route.stops:
                if stop.request_id not in req_route:
                    req_route[stop.request_id] = {}
                req_route[stop.request_id][stop.action] = route

    scored = []
    for e in state['scheduled']:
        req = e['request']
        routes = req_route.get(req.id, {})
        score = 0.0
        for action in ('delivery', 'pickup'):
            route = routes.get(action)
            if route is not None:
                score += route.distance / max(len(route.stops), 1)
        scored.append((score, req))

    scored.sort(key=lambda x: x[0], reverse=True)
    k = k if k is not None else max(1, len(scored) // 5)
    return [req for _, req in scored[:k]]


def break_distance_cost(state: dict, instance: Instance, route_set: dict,
                        k: int | None = None) -> list[Request]:
    """Return requests most responsible for distance cost.

    Scores each request by its actual detour: dist(prev,stop)+dist(stop,next)-dist(prev,next)
    summed over its delivery and pickup stops. This is the exact marginal distance that
    removing the request from its current routes would save.
    """
    depot_id = instance.depot_id
    req_by_id = {r.id: r for r in instance.requests}

    req_detour: dict[int, int] = defaultdict(int)
    for routes in route_set.values():
        for route in routes:
            for i, stop in enumerate(route.stops):
                req_detour[stop.request_id] += _stop_detour(route, i, depot_id, instance)

    scored = []
    for e in state['scheduled']:
        req = e['request']
        scored.append((req_detour.get(req.id, 0), req))

    scored.sort(key=lambda x: x[0], reverse=True)
    k = k if k is not None else max(1, len(scored) // 5)
    return [req for _, req in scored[:k]]


def break_worst_day(state: dict, instance: Instance, route_set: dict) -> list[Request]:
    """Return all requests with a stop on the highest-distance routing day.

    Clears the entire worst day so repair can freely redistribute its requests,
    enabling structural route improvements rather than marginal detour reduction.
    """
    if not route_set:
        return []

    worst_day = max(route_set, key=lambda d: sum(r.distance for r in route_set[d]))
    depot_id = instance.depot_id

    req_detour: dict[int, int] = defaultdict(int)
    for route in route_set[worst_day]:
        for i, stop in enumerate(route.stops):
            req_detour[stop.request_id] += _stop_detour(route, i, depot_id, instance)

    candidates = [
        e for e in state['scheduled']
        if e['delivery_day'] == worst_day or e['pickup_day'] == worst_day
    ]
    candidates.sort(key=lambda e: req_detour.get(e['request'].id, 0), reverse=True)
    return [e['request'] for e in candidates]


def break_geographic(state: dict, instance: Instance, k: int) -> list[Request]:
    """Return k requests nearest (by location) to a random seed request.

    Selects a random scheduled request as seed, then returns the k closest
    scheduled requests by location distance (seed included). Destroying a
    geographic cluster enables the repair to find better joint routings.
    """
    if not state['scheduled']:
        return []
    seed_loc = random.choice(state['scheduled'])['request'].location_id
    candidates = sorted(
        state['scheduled'],
        key=lambda e: instance.get_distance(seed_loc, e['request'].location_id),
    )
    return [e['request'] for e in candidates[:k]]
