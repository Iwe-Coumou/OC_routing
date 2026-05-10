import math
import random
import logging
from tqdm import tqdm
from scheduling.state import snapshot, restore, uncommit_request, place_unscheduled
from routing.solver import solve_routing, solve_routing_incremental, VehicleRoute
from routing.export import cost_from_routes
from .break_fns import (
    break_tool_cost,
    break_vehicle_cost,
    break_vehicle_day_cost,
    break_distance_cost,
    break_worst_day,
    break_geographic,
)
from .repair_fns import (
    repair_tool_cost,
    repair_vehicle_cost,
    repair_vehicle_day_cost,
    repair_distance_cost,
)

log = logging.getLogger(__name__)

# ALNS weight update constants
_ALNS_REWARD    = 1.5    # accepted improvement
_ALNS_SA_REWARD = 1.05   # accepted via SA (non-improving)
_ALNS_PENALTY   = 0.90   # rejected
_ALNS_W_MIN     = 0.1
_ALNS_W_MAX     = 10.0

_OP_KEYS = ['tool', 'vehicle', 'vehicle_days', 'distance', 'worst_day']

_BREAK_REPAIR = {
    'tool':         (break_tool_cost,        repair_tool_cost),
    'vehicle':      (break_vehicle_cost,     repair_vehicle_cost),
    'vehicle_days': (break_vehicle_day_cost, repair_vehicle_day_cost),
    'distance':     (break_distance_cost,    repair_distance_cost),
    'worst_day':    (break_worst_day,        repair_distance_cost),
}
_NEEDS_ROUTES = {'vehicle', 'vehicle_days', 'distance', 'worst_day'}
_DAY_TARGETED = {'tool', 'vehicle', 'worst_day'}  # destroy full day, no k limit

_OP_BREAKDOWN = {
    'tool':         'tool',
    'vehicle':      'vehicle',
    'vehicle_days': 'vehicle_days',
    'distance':     'distance',
    'worst_day':    'distance',
}

_COST_OP_PROB = 0.20   # fraction of iterations using cost-targeted destroy
_EPSILON      = 0.25   # repair randomness (uniform random from feasible days)
_GEO_PROB     = 0.30   # fraction of random-destroy iterations using geographic cluster
_SA_T0_FRAC   = 0.02   # SA initial temperature as fraction of initial cost
_SA_ALPHA     = 0.998  # SA cooling rate — reaches ~37% of T0 at iter 500


def _routes_without(routes: dict, req_ids: set) -> dict:
    result = {}
    for day, day_routes in routes.items():
        clean = []
        for route in day_routes:
            filtered = [s for s in route.stops if s.request_id not in req_ids]
            if filtered:
                clean.append(VehicleRoute(
                    vehicle_id=route.vehicle_id,
                    stops=filtered,
                    distance=route.distance,
                ))
        result[day] = clean
    return result


def _get_days_for_requests(state: dict, req_ids: set) -> set:
    days = set()
    for e in state['scheduled']:
        if e['request'].id in req_ids:
            days.add(e['delivery_day'])
            days.add(e['pickup_day'])
    return days


def route_lns(
    state: dict,
    instance,
    iterations: int = 500,
    patience: int = 500,
    initial_routes: dict | None = None,
) -> dict:
    if initial_routes is not None:
        current_routes = initial_routes
    else:
        print("  computing initial routing cost...", flush=True)
        current_routes = solve_routing(state, instance, fast=True)
    best_cost = cost_from_routes(current_routes, instance)['total']
    best_snap = snapshot(state)
    breakdown = cost_from_routes(current_routes, instance)
    print(f"  initial routed cost: {best_cost:.3e}", flush=True)

    T0 = _SA_T0_FRAC * best_cost
    T = T0
    current_cost = best_cost

    max_destroy = max(30, len(state['scheduled']) // 8)
    alns_w = {k: 1.0 for k in _OP_KEYS}

    no_improve = 0
    total_improvements = 0
    total_sa_accepts = 0

    log.info(
        f"=== OPTIMISE START  instance={instance.name}  "
        f"initial={best_cost:.3e}  "
        f"tool={breakdown['tool']:.3e}  vehicle={breakdown['vehicle']:.3e}  "
        f"veh_days={breakdown['vehicle_days']:.3e}  distance={breakdown['distance']:.3e} ==="
    )

    stop_reason = 'iterations'
    pbar = tqdm(range(iterations), desc="route_lns", unit="iter")
    for iteration in pbar:
        snap = snapshot(state)

        if random.random() < _COST_OP_PROB:
            combined = [alns_w[key] * max(breakdown[_OP_BREAKDOWN[key]], 1) for key in _OP_KEYS]
            driver = random.choices(_OP_KEYS, weights=combined, k=1)[0]
            break_fn, repair_fn = _BREAK_REPAIR[driver]

            if driver in _DAY_TARGETED:
                if driver in _NEEDS_ROUTES:
                    targets = break_fn(state, instance, current_routes)
                else:
                    targets = break_fn(state, instance)
            else:
                k = min(max_destroy, max(1, int(len(state['scheduled']) * random.uniform(0.1, 0.25))))
                if driver in _NEEDS_ROUTES:
                    targets = break_fn(state, instance, current_routes, k)
                else:
                    targets = break_fn(state, instance, k)

            target_req_ids = {r.id for r in targets}
            changed_days = _get_days_for_requests(state, target_req_ids)
            for req in targets:
                uncommit_request(state, req)

            repair_routes = _routes_without(current_routes, target_req_ids)
            repair_fn(state, instance, epsilon=_EPSILON, current_routes=repair_routes)

            used_edd_fallback = any(v for v in state['unscheduled'].values())
            if used_edd_fallback:
                place_unscheduled(state, instance)

            op_label = f"cost:{driver[:3].upper()}"
            k_str = "all" if driver in _DAY_TARGETED else str(k)
            op_detail = f"op=cost driver={driver} k={k_str}" + (" +edd_fallback" if used_edd_fallback else "")

        else:
            k = min(max_destroy, max(1, int(len(state['scheduled']) * random.uniform(0.1, 0.3))))

            if random.random() < _GEO_PROB:
                targets = break_geographic(state, instance, k)
                op_label = "geo"
                break_label = "geo"
            else:
                sampled = random.sample(state['scheduled'], k)
                targets = [e['request'] for e in sampled]
                op_label = "rand"
                break_label = "rand"

            target_req_ids = {r.id for r in targets}
            changed_days = _get_days_for_requests(state, target_req_ids)
            for req in targets:
                uncommit_request(state, req)

            rand_driver = random.choices(_OP_KEYS, weights=[alns_w[key] for key in _OP_KEYS], k=1)[0]
            _, rand_repair = _BREAK_REPAIR[rand_driver]
            repair_routes = _routes_without(current_routes, target_req_ids)
            rand_repair(state, instance, epsilon=0.5, current_routes=repair_routes)
            driver = rand_driver
            if any(v for v in state['unscheduled'].values()):
                place_unscheduled(state, instance)

            op_detail = f"op={break_label} k={k} repair={rand_driver}"

        unscheduled_count = sum(len(v) for v in state['unscheduled'].values())
        if unscheduled_count > 0:
            log.warning(f"iter={iteration:4d}  {op_detail}  REJECT (unscheduled={unscheduled_count})")
            restore(state, instance, snap)
            alns_w[driver] = max(_ALNS_W_MIN, alns_w[driver] * _ALNS_PENALTY)
            no_improve += 1
            pbar.set_postfix(best=f"{best_cost:.3e}", impr=total_improvements,
                             stale=no_improve, op=op_label)
            if no_improve >= patience:
                stop_reason = 'patience'
                break
            continue

        for e in state['scheduled']:
            if e['request'].id in target_req_ids:
                changed_days.add(e['delivery_day'])
                changed_days.add(e['pickup_day'])

        candidate_routes = solve_routing_incremental(
            state, instance, changed_days, current_routes
        )
        candidate_cost = cost_from_routes(candidate_routes, instance)['total']
        delta = candidate_cost - current_cost
        T = max(T * _SA_ALPHA, 1e-6)

        if delta < 0:
            accept = True
        elif T > 1e-6 and random.random() < math.exp(-delta / T):
            accept = True
        else:
            accept = False

        if accept:
            current_cost = candidate_cost
            current_routes = candidate_routes

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_snap = snapshot(state)
                breakdown = cost_from_routes(candidate_routes, instance)
                no_improve = 0
                total_improvements += 1
                alns_w[driver] = min(_ALNS_W_MAX, alns_w[driver] * _ALNS_REWARD)
                log.info(
                    f"iter={iteration:4d}  {op_detail}  "
                    f"candidate={candidate_cost:.3e}  delta={delta:+.3e}  ACCEPT (improve)  "
                    f"[tool={breakdown['tool']:.3e}  vehicle={breakdown['vehicle']:.3e}  "
                    f"veh_days={breakdown['vehicle_days']:.3e}  distance={breakdown['distance']:.3e}]"
                )
            else:
                no_improve += 1
                total_sa_accepts += 1
                alns_w[driver] = min(_ALNS_W_MAX, alns_w[driver] * _ALNS_SA_REWARD)
                log.info(
                    f"iter={iteration:4d}  {op_detail}  "
                    f"candidate={candidate_cost:.3e}  delta={delta:+.3e}  ACCEPT (SA)"
                )
        else:
            restore(state, instance, snap)
            no_improve += 1
            alns_w[driver] = max(_ALNS_W_MIN, alns_w[driver] * _ALNS_PENALTY)
            log.info(
                f"iter={iteration:4d}  {op_detail}  "
                f"candidate={candidate_cost:.3e}  delta={delta:+.3e}  reject"
            )

        pbar.set_postfix(best=f"{best_cost:.3e}", impr=total_improvements,
                         stale=no_improve, op=op_label, T=f"{T:.1e}")

        if no_improve >= patience:
            stop_reason = 'patience'
            break

    restore(state, instance, best_snap)
    log.info(
        f"=== OPTIMISE END  best={best_cost:.3e}  improvements={total_improvements}  "
        f"sa_accepts={total_sa_accepts}  iterations={iteration + 1}  stopped={stop_reason} ==="
    )
    print("  computing final routes (quality mode)...", flush=True)
    clean_fast_routes = solve_routing(state, instance, fast=True)
    final_routes = solve_routing(state, instance, fast=False, time_limit_seconds=30,
                                 initial_routes=clean_fast_routes)
    final_cost = cost_from_routes(final_routes, instance)['total']
    print(f"  final routed cost: {final_cost:.3e}  (LNS best was {best_cost:.3e})", flush=True)
    return final_routes
