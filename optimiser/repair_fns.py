import math
import random
import logging
from collections import defaultdict
from instance import Instance
from scheduling.state import commit_request, is_feasible, _first_feasible_day

log = logging.getLogger(__name__)


def _sorted_candidates(state: dict, instance: Instance) -> list:
    result = []
    for reqs in state['unscheduled'].values():
        for req in reqs:
            count = sum(
                1 for d in range(req.earliest, req.latest + 1)
                if is_feasible(state, instance, req, d, d + req.duration)
            )
            result.append((count, req.latest - req.earliest, req.latest, req.id, req))
    result.sort()
    return [req for _, _, _, _, req in result]


def _cheapest_insertion_cost(loc: int, routes: list, depot_id: int, instance) -> int:
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
    n = instance.config.days + 2

    for req in _sorted_candidates(state, instance):
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


def repair_vehicle_cost(state: dict, instance: Instance, epsilon: float = 0.0,
                        current_routes: dict | None = None) -> None:
    tool_by_type = {t.id: t for t in instance.tools}
    cap = instance.config.capacity

    for req in _sorted_candidates(state, instance):
        req_load = req.num_machines * tool_by_type[req.machine_type].size

        load: dict[int, int] = defaultdict(int)
        for e in state['scheduled']:
            r = e['request']
            load[e['delivery_day']] += r.num_machines * tool_by_type[r.machine_type].size
            load[e['pickup_day']]   += r.num_machines * tool_by_type[r.machine_type].size

        current_max = max((math.ceil(l / cap) for l in load.values()), default=0)

        best_day = None
        best_vehicles = float('inf')
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
            if projected_max < best_vehicles or (projected_max == best_vehicles and secondary > best_secondary):
                best_vehicles = projected_max
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


def repair_vehicle_day_cost(state: dict, instance: Instance, epsilon: float = 0.0,
                            current_routes: dict | None = None) -> None:
    tool_by_type = {t.id: t for t in instance.tools}

    for req in _sorted_candidates(state, instance):
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


def repair_distance_cost(state: dict, instance: Instance, epsilon: float = 0.0,
                         current_routes: dict | None = None) -> None:
    depot_id = instance.depot_id

    for req in _sorted_candidates(state, instance):
        depot_dist = instance.get_distance(depot_id, req.location_id)

        locs_per_day: dict[int, list[int]] = defaultdict(list)
        if current_routes is None:
            for e in state['scheduled']:
                locs_per_day[e['delivery_day']].append(e['request'].location_id)
                locs_per_day[e['pickup_day']].append(e['request'].location_id)

        feasible_days = [
            d for d in range(req.earliest, req.latest + 1)
            if is_feasible(state, instance, req, d, d + req.duration)
        ]

        if not feasible_days:
            log.warning(f"repair_distance_cost: req={req.id} has no feasible day — leaving unscheduled")
            continue

        if epsilon > 0 and random.random() < epsilon:
            commit_request(state, instance, req, random.choice(feasible_days))
            continue

        best_day = None
        best_insert_cost = float('inf')

        for d in feasible_days:
            p = d + req.duration
            if current_routes is not None:
                insert_cost = (
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
                insert_cost = nearest_d + nearest_p

            if insert_cost < best_insert_cost:
                best_insert_cost = insert_cost
                best_day = d

        if best_day is None:
            best_day = feasible_days[0]
        commit_request(state, instance, req, best_day)
