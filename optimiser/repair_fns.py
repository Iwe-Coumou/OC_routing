import math
import random
import logging
from collections import defaultdict
from instance import Instance
from scheduling.state import commit_request, is_feasible

log = logging.getLogger(__name__)


def _next_request(state: dict, instance: Instance, blocked: set):
    """Return the unscheduled request with the fewest currently feasible days (MRV).

    Re-evaluated on every call so that capacity changes from prior commits are
    reflected. Requests in `blocked` are skipped — they have already been found
    to have no feasible day and can't be helped by further commits.
    Ties broken by window width then EDD.
    """
    best_req = None
    best_key = (float('inf'),) * 3
    for reqs in state['unscheduled'].values():
        for req in reqs:
            if req.id in blocked:
                continue
            count = sum(
                1 for d in range(req.earliest, req.latest + 1)
                if is_feasible(state, instance, req, d, d + req.duration)
            )
            key = (count, req.latest - req.earliest, req.latest)
            if key < best_key:
                best_key = key
                best_req = req
    return best_req


def _first_feasible_day(state: dict, instance: Instance, req) -> int | None:
    """Earliest feasible delivery day in the request's time window."""
    for d in range(req.earliest, req.latest + 1):
        if is_feasible(state, instance, req, d, d + req.duration):
            return d
    return None


def _cheapest_insertion_cost(loc: int, routes: list, depot_id: int, instance) -> int:
    """Minimum additional distance to insert loc into any position across all routes on a day."""
    if not routes:
        return 2 * instance.get_distance(depot_id, loc)
    best = float('inf')
    for route in routes:
        stops = route.stops
        n = len(stops)
        for i in range(n + 1):
            prev = stops[i - 1].location_id if i > 0 else depot_id
            nxt = stops[i].location_id if i < n else depot_id
            cost = (instance.get_distance(prev, loc)
                    + instance.get_distance(loc, nxt)
                    - instance.get_distance(prev, nxt))
            if cost < best:
                best = cost
    return max(0, best)


def repair_tool_cost(state: dict, instance: Instance, epsilon: float = 0.0,
                     current_routes: dict | None = None) -> None:
    """Place unscheduled requests to minimise peak concurrent tool loans.

    For each request (MRV order), picks the feasible delivery day that minimises the
    resulting maximum concurrent loan count across the planning horizon, spreading
    deliveries to flatten the loan peak for each tool type.
    """
    n = instance.config.days + 2
    blocked: set = set()

    while True:
        req = _next_request(state, instance, blocked)
        if req is None:
            break

        diff = state['loans'].get(req.machine_type, [0] * n)
        pickups = state['pickups_per_day'].get(req.machine_type, [0] * n)

        current_peaks: list[int] = []
        running = 0
        for day in range(n):
            running += diff[day]
            current_peaks.append(running + pickups[day])

        prefix_max = [0] * n
        prefix_max[0] = current_peaks[0]
        for i in range(1, n):
            prefix_max[i] = max(prefix_max[i - 1], current_peaks[i])
        suffix_max = [0] * n
        suffix_max[-1] = current_peaks[-1]
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(suffix_max[i + 1], current_peaks[i])

        best_day, best_peak = None, float('inf')
        feasible_days = []
        for d in range(req.earliest, req.latest + 1):
            p = d + req.duration
            if not is_feasible(state, instance, req, d, p):
                continue
            feasible_days.append(d)
            window_peak = max(current_peaks[d: p + 1]) + req.num_machines
            outside_peak = max(
                prefix_max[d - 1] if d > 0 else 0,
                suffix_max[p + 1] if p + 1 < n else 0,
            )
            new_peak = max(window_peak, outside_peak)
            if new_peak < best_peak:
                best_peak = new_peak
                best_day = d
        if epsilon > 0 and feasible_days and random.random() < epsilon:
            best_day = random.choice(feasible_days)

        if best_day is None:
            best_day = _first_feasible_day(state, instance, req)
        if best_day is not None:
            commit_request(state, instance, req, best_day)
        else:
            log.warning(f"repair_tool_cost: req={req.id} has no feasible day — leaving unscheduled")
            blocked.add(req.id)


def repair_vehicle_cost(state: dict, instance: Instance, epsilon: float = 0.0,
                        current_routes: dict | None = None) -> None:
    """Place unscheduled requests to minimise the peak daily vehicle count.

    For each request (MRV order), picks the feasible day where adding this request
    results in the lowest max-vehicles-on-any-day. Ties are broken by preferring the
    day with the most existing load (consolidation into running vehicles).
    """
    tool_by_type = {t.id: t for t in instance.tools}
    cap = instance.config.capacity
    blocked: set = set()

    while True:
        req = _next_request(state, instance, blocked)
        if req is None:
            break

        req_load = req.num_machines * tool_by_type[req.machine_type].size

        load: dict[int, int] = defaultdict(int)
        for e in state['scheduled']:
            r = e['request']
            l = r.num_machines * tool_by_type[r.machine_type].size
            load[e['delivery_day']] += l
            load[e['pickup_day']] += l

        current_max = max((math.ceil(l / cap) for l in load.values()), default=0)

        best_day = None
        best_peak = float('inf')
        best_secondary = float('-inf')
        feasible_days = []

        for d in range(req.earliest, req.latest + 1):
            p = d + req.duration
            if not is_feasible(state, instance, req, d, p):
                continue
            feasible_days.append(d)
            new_veh_d = math.ceil((load.get(d, 0) + req_load) / cap)
            new_veh_p = math.ceil((load.get(p, 0) + req_load) / cap)
            projected_max = max(current_max, new_veh_d, new_veh_p)
            secondary = load.get(d, 0) + load.get(p, 0)
            if projected_max < best_peak or (projected_max == best_peak and secondary > best_secondary):
                best_peak = projected_max
                best_secondary = secondary
                best_day = d
        if epsilon > 0 and feasible_days and random.random() < epsilon:
            best_day = random.choice(feasible_days)

        if best_day is None:
            best_day = _first_feasible_day(state, instance, req)
        if best_day is not None:
            commit_request(state, instance, req, best_day)
        else:
            log.warning(f"repair_vehicle_cost: req={req.id} has no feasible day — leaving unscheduled")
            blocked.add(req.id)


def repair_vehicle_day_cost(state: dict, instance: Instance, epsilon: float = 0.0,
                            current_routes: dict | None = None) -> None:
    """Place unscheduled requests to minimise total vehicle-days.

    For each request (MRV order), picks the feasible day pair (delivery, pickup) where
    the combined existing load is highest — consolidating onto already-busy days fills
    running vehicles rather than spawning new trips.
    """
    tool_by_type = {t.id: t for t in instance.tools}
    blocked: set = set()

    while True:
        req = _next_request(state, instance, blocked)
        if req is None:
            break

        load: dict[int, int] = defaultdict(int)
        for e in state['scheduled']:
            r = e['request']
            l = r.num_machines * tool_by_type[r.machine_type].size
            load[e['delivery_day']] += l
            load[e['pickup_day']] += l

        best_day = None
        best_score = float('-inf')
        feasible_days = []

        for d in range(req.earliest, req.latest + 1):
            p = d + req.duration
            if not is_feasible(state, instance, req, d, p):
                continue
            feasible_days.append(d)
            score = load.get(d, 0) + load.get(p, 0)
            if score > best_score:
                best_score = score
                best_day = d
        if epsilon > 0 and feasible_days and random.random() < epsilon:
            best_day = random.choice(feasible_days)

        if best_day is None:
            best_day = _first_feasible_day(state, instance, req)
        if best_day is not None:
            commit_request(state, instance, req, best_day)
        else:
            log.warning(f"repair_vehicle_day_cost: req={req.id} has no feasible day — leaving unscheduled")
            blocked.add(req.id)


def repair_distance_cost(state: dict, instance: Instance, epsilon: float = 0.0,
                         current_routes: dict | None = None) -> None:
    """Place unscheduled requests to minimise distance cost.

    When current_routes is provided (optimiser context), scores each candidate day
    by the cheapest insertion cost into the actual routes for that day — the exact
    marginal distance the new stop adds. Falls back to nearest-stop estimate when
    routes are unavailable (scheduling context).
    """
    depot_id = instance.depot_id
    blocked: set = set()

    while True:
        req = _next_request(state, instance, blocked)
        if req is None:
            break

        depot_dist = instance.get_distance(depot_id, req.location_id)

        locs_per_day: dict[int, list[int]] = defaultdict(list)
        if current_routes is None:
            for e in state['scheduled']:
                locs_per_day[e['delivery_day']].append(e['request'].location_id)
                locs_per_day[e['pickup_day']].append(e['request'].location_id)

        best_day = None
        best_score = float('inf')
        feasible_days = []

        for d in range(req.earliest, req.latest + 1):
            p = d + req.duration
            if not is_feasible(state, instance, req, d, p):
                continue
            feasible_days.append(d)

            if current_routes is not None:
                score = (
                    _cheapest_insertion_cost(req.location_id, current_routes.get(d, []), depot_id, instance)
                    + _cheapest_insertion_cost(req.location_id, current_routes.get(p, []), depot_id, instance)
                )
            else:
                d_locs = locs_per_day.get(d, [])
                nearest_d = min(
                    (instance.get_distance(req.location_id, l) for l in d_locs),
                    default=depot_dist,
                )
                p_locs = locs_per_day.get(p, [])
                nearest_p = min(
                    (instance.get_distance(req.location_id, l) for l in p_locs),
                    default=depot_dist,
                )
                score = nearest_d + nearest_p

            if score < best_score:
                best_score = score
                best_day = d

        if epsilon > 0 and feasible_days and random.random() < epsilon:
            best_day = random.choice(feasible_days)

        if best_day is None:
            best_day = _first_feasible_day(state, instance, req)
        if best_day is not None:
            commit_request(state, instance, req, best_day)
        else:
            log.warning(f"repair_distance_cost: req={req.id} has no feasible day — leaving unscheduled")
            blocked.add(req.id)
