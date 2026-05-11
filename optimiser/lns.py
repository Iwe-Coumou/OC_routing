import math
import random
import logging
from tqdm import tqdm
from scheduling.state import snapshot, restore, uncommit_request, place_unscheduled
from scheduling.cost import cost_breakdown
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

# Break operators: cost-targeted + random/geo
_BREAK_KEYS      = ['tool', 'vehicle', 'vehicle_days', 'distance', 'worst_day', 'random', 'geo']
_BREAK_FNS       = {
    'tool':         break_tool_cost,
    'vehicle':      break_vehicle_cost,
    'vehicle_days': break_vehicle_day_cost,
    'distance':     break_distance_cost,
    'worst_day':    break_worst_day,
}
_NEEDS_ROUTES    = {'vehicle', 'vehicle_days', 'distance', 'worst_day'}
_DAY_TARGETED    = {'tool', 'vehicle', 'worst_day'}
_BREAK_DRIVER    = {        # which breakdown component weights each cost break op
    'tool':         'tool',
    'vehicle':      'vehicle',
    'vehicle_days': 'vehicle_days',
    'distance':     'distance',
    'worst_day':    'distance',
}
_COST_BREAK_KEYS = [k for k in _BREAK_KEYS if k in _BREAK_DRIVER]

# Repair operators (decoupled from break)
_REPAIR_KEYS = ['tool', 'vehicle', 'vehicle_days', 'distance']
_REPAIR_FNS  = {
    'tool':         repair_tool_cost,
    'vehicle':      repair_vehicle_cost,
    'vehicle_days': repair_vehicle_day_cost,
    'distance':     repair_distance_cost,
}

_RANDOM_INIT_W  = 0.625  # random+geo start at ~20% combined across 7 break ops
_EPSILON        = 0.25   # repair randomness
_SA_T0_FRAC     = 0.02
_SA_ALPHA       = 0.998
_MAX_RESTARTS   = 3      # weight+temperature resets before stopping
_REHEAT_FRAC    = 0.5    # T after reheat as fraction of T0


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
    best_clean_cost = cost_from_routes(current_routes, instance)['total']
    best_snap = snapshot(state)
    best_routes = current_routes
    breakdown = cost_from_routes(current_routes, instance)
    print(f"  initial routed cost: {best_clean_cost:.3e}", flush=True)
    n_init_unscheduled = sum(len(v) for v in state['unscheduled'].values())

    T0 = _SA_T0_FRAC * best_clean_cost
    unscheduled_penalty = int(T0)
    T = T0
    best_cost = best_clean_cost + n_init_unscheduled * unscheduled_penalty
    current_cost = best_cost

    best_feasible_cost   = float('inf')
    best_feasible_snap   = None
    best_feasible_routes = None
    if n_init_unscheduled == 0:
        best_feasible_cost   = best_clean_cost
        best_feasible_snap   = best_snap
        best_feasible_routes = current_routes

    max_destroy = max(30, len(state['scheduled']) // 8)
    k_scale = 1.0

    alns_w_break  = {k: 1.0 for k in _COST_BREAK_KEYS}
    alns_w_break['random'] = _RANDOM_INIT_W
    alns_w_break['geo']    = _RANDOM_INIT_W
    alns_w_repair = {k: 1.0 for k in _REPAIR_KEYS}

    no_improve = 0
    total_improvements = 0
    total_sa_accepts = 0
    restarts = 0
    dirty_days: set[int] = set()  # days changed by accepted infeasible states, not yet routed

    log.info(
        f"=== OPTIMISE START  instance={instance.name}  "
        f"initial={best_clean_cost:.3e}  unscheduled={n_init_unscheduled}  "
        f"tool={breakdown['tool']:.3e}  vehicle={breakdown['vehicle']:.3e}  "
        f"veh_days={breakdown['vehicle_days']:.3e}  distance={breakdown['distance']:.3e} ==="
    )

    stop_reason = 'iterations'
    pbar = tqdm(range(iterations), desc="route_lns", unit="iter")
    for iteration in pbar:
        snap = snapshot(state)

        # --- select break operator ---
        # cost-targeted ops weighted by normalised breakdown; random/geo by raw weight
        cost_bvals = [breakdown[_BREAK_DRIVER[k]] for k in _COST_BREAK_KEYS]
        total_bd = sum(cost_bvals) or 1
        break_weights = [alns_w_break[k] * cost_bvals[i] / total_bd
                         for i, k in enumerate(_COST_BREAK_KEYS)]
        break_weights += [alns_w_break['random'], alns_w_break['geo']]
        break_op = random.choices(_BREAK_KEYS, weights=break_weights, k=1)[0]

        # --- select repair operator ---
        repair_op = random.choices(_REPAIR_KEYS,
                                   weights=[alns_w_repair[k] for k in _REPAIR_KEYS], k=1)[0]
        repair_fn = _REPAIR_FNS[repair_op]

        # --- execute break ---
        if break_op == 'random':
            k = min(max_destroy, max(1, int(len(state['scheduled']) * k_scale * random.uniform(0.1, 0.3))))
            targets = [e['request'] for e in random.sample(state['scheduled'], k)]
            k_str = str(k)
        elif break_op == 'geo':
            k = min(max_destroy, max(1, int(len(state['scheduled']) * k_scale * random.uniform(0.1, 0.3))))
            targets = break_geographic(state, instance, k)
            k_str = str(k)
        elif break_op in _DAY_TARGETED:
            if break_op in _NEEDS_ROUTES:
                targets = _BREAK_FNS[break_op](state, instance, current_routes)
            else:
                targets = _BREAK_FNS[break_op](state, instance)
            targets = targets[:max_destroy]
            k_str = str(len(targets))
        else:
            k = min(max_destroy, max(1, int(len(state['scheduled']) * k_scale * random.uniform(0.1, 0.25))))
            if break_op in _NEEDS_ROUTES:
                targets = _BREAK_FNS[break_op](state, instance, current_routes, k)
            else:
                targets = _BREAK_FNS[break_op](state, instance, k)
            k_str = str(k)

        target_req_ids = {r.id for r in targets}
        changed_days = _get_days_for_requests(state, target_req_ids)
        for req in targets:
            uncommit_request(state, req)

        # --- execute repair ---
        repair_routes = _routes_without(current_routes, target_req_ids)
        repair_fn(state, instance, epsilon=_EPSILON, current_routes=repair_routes)
        if any(v for v in state['unscheduled'].values()):
            place_unscheduled(state, instance)

        op_label  = f"{break_op[:4]}/{repair_op[:4]}"
        op_detail = f"break={break_op} repair={repair_op} k={k_str}"

        # --- evaluate ---
        unscheduled_count = sum(len(v) for v in state['unscheduled'].values())

        # Hard-reject fast path: instances that started fully scheduled should never
        # explore infeasible states — skip routing entirely and restore immediately.
        if unscheduled_count > 0 and n_init_unscheduled == 0:
            k_scale = max(0.1, k_scale * 0.9)
            restore(state, instance, snap)
            no_improve += 1
            T = max(T * _SA_ALPHA, 1e-6)
            alns_w_break[break_op]   = max(_ALNS_W_MIN, alns_w_break[break_op]   * _ALNS_PENALTY)
            alns_w_repair[repair_op] = max(_ALNS_W_MIN, alns_w_repair[repair_op] * _ALNS_PENALTY)
            log.info(f"iter={iteration:4d}  {op_detail}  REJECT (unscheduled={unscheduled_count})")
            pbar.set_postfix(best=f"{best_feasible_cost:.3e}", impr=total_improvements,
                             stale=no_improve, op=op_label, T=f"{T:.1e}", ks=f"{k_scale:.2f}")
            if no_improve >= patience:
                if restarts >= _MAX_RESTARTS:
                    stop_reason = 'patience'
                    break
                restarts += 1
                no_improve = 0
                T = T0 * _REHEAT_FRAC
                alns_w_break  = {k: 1.0 for k in _COST_BREAK_KEYS}
                alns_w_break['random'] = _RANDOM_INIT_W
                alns_w_break['geo']    = _RANDOM_INIT_W
                alns_w_repair = {k: 1.0 for k in _REPAIR_KEYS}
                log.info(f"iter={iteration:4d}  RESTART {restarts}/{_MAX_RESTARTS}  T={T:.3e}")
            continue

        if unscheduled_count > 0:
            k_scale = max(0.1, k_scale * 0.9)
        else:
            k_scale = min(1.0, k_scale * 1.02)

        for e in state['scheduled']:
            if e['request'].id in target_req_ids:
                changed_days.add(e['delivery_day'])
                changed_days.add(e['pickup_day'])

        # Penalty-mode estimate path: skip routing for infeasible states; use the
        # scheduling cost estimate instead. Dirty days are carried forward and
        # flushed into the next real routing call once the state is feasible.
        if unscheduled_count > 0:  # n_init_unscheduled > 0 guaranteed (hard-reject above handles == 0)
            sched_estimate = cost_breakdown(state, instance)['total']
            candidate_cost = sched_estimate + unscheduled_count * unscheduled_penalty
            delta = candidate_cost - current_cost
            T = max(T * _SA_ALPHA, 1e-6)
            if delta < 0 or (T > 1e-6 and random.random() < math.exp(-delta / T)):
                current_cost = candidate_cost
                dirty_days |= changed_days
                no_improve += 1
                alns_w_break[break_op]   = min(_ALNS_W_MAX, alns_w_break[break_op]   * _ALNS_SA_REWARD)
                alns_w_repair[repair_op] = min(_ALNS_W_MAX, alns_w_repair[repair_op] * _ALNS_SA_REWARD)
                log.info(
                    f"iter={iteration:4d}  {op_detail}  "
                    f"estimate={sched_estimate:.3e}  unscheduled={unscheduled_count}  delta={delta:+.3e}  ACCEPT (est)"
                )
            else:
                restore(state, instance, snap)
                no_improve += 1
                alns_w_break[break_op]   = max(_ALNS_W_MIN, alns_w_break[break_op]   * _ALNS_PENALTY)
                alns_w_repair[repair_op] = max(_ALNS_W_MIN, alns_w_repair[repair_op] * _ALNS_PENALTY)
                log.info(
                    f"iter={iteration:4d}  {op_detail}  "
                    f"estimate={sched_estimate:.3e}  unscheduled={unscheduled_count}  delta={delta:+.3e}  reject (est)"
                )
            pbar.set_postfix(best=f"{best_feasible_cost:.3e}", impr=total_improvements,
                             stale=no_improve, op=op_label, T=f"{T:.1e}", ks=f"{k_scale:.2f}")
            if no_improve >= patience:
                if restarts >= _MAX_RESTARTS:
                    stop_reason = 'patience'
                    break
                restarts += 1
                no_improve = 0
                T = T0 * _REHEAT_FRAC
                alns_w_break  = {k: 1.0 for k in _COST_BREAK_KEYS}
                alns_w_break['random'] = _RANDOM_INIT_W
                alns_w_break['geo']    = _RANDOM_INIT_W
                alns_w_repair = {k: 1.0 for k in _REPAIR_KEYS}
                log.info(f"iter={iteration:4d}  RESTART {restarts}/{_MAX_RESTARTS}  T={T:.3e}")
            continue

        # Feasible path: route changed days plus any days dirtied by prior infeasible acceptances.
        all_changed = changed_days | dirty_days
        dirty_days = set()
        candidate_routes = solve_routing_incremental(
            state, instance, all_changed, current_routes
        )
        routed_cost = cost_from_routes(candidate_routes, instance)['total']
        candidate_cost = routed_cost + unscheduled_count * unscheduled_penalty
        delta = candidate_cost - current_cost
        T = max(T * _SA_ALPHA, 1e-6)

        if delta < 0 or (T > 1e-6 and random.random() < math.exp(-delta / T)):
            accept = True
        else:
            accept = False

        if accept:
            current_cost = candidate_cost
            current_routes = candidate_routes

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_clean_cost = routed_cost
                best_snap = snapshot(state)
                best_routes = candidate_routes
                breakdown = cost_from_routes(candidate_routes, instance)
                no_improve = 0
                total_improvements += 1
                alns_w_break[break_op]   = min(_ALNS_W_MAX, alns_w_break[break_op]   * _ALNS_REWARD)
                alns_w_repair[repair_op] = min(_ALNS_W_MAX, alns_w_repair[repair_op] * _ALNS_REWARD)
                log.info(
                    f"iter={iteration:4d}  {op_detail}  "
                    f"candidate={routed_cost:.3e}  unscheduled={unscheduled_count}  delta={delta:+.3e}  ACCEPT (improve)  "
                    f"[tool={breakdown['tool']:.3e}  vehicle={breakdown['vehicle']:.3e}  "
                    f"veh_days={breakdown['vehicle_days']:.3e}  distance={breakdown['distance']:.3e}]"
                )
            else:
                no_improve += 1
                total_sa_accepts += 1
                alns_w_break[break_op]   = min(_ALNS_W_MAX, alns_w_break[break_op]   * _ALNS_SA_REWARD)
                alns_w_repair[repair_op] = min(_ALNS_W_MAX, alns_w_repair[repair_op] * _ALNS_SA_REWARD)
                log.info(
                    f"iter={iteration:4d}  {op_detail}  "
                    f"candidate={routed_cost:.3e}  unscheduled={unscheduled_count}  delta={delta:+.3e}  ACCEPT (SA)"
                )
        else:
            restore(state, instance, snap)
            no_improve += 1
            alns_w_break[break_op]   = max(_ALNS_W_MIN, alns_w_break[break_op]   * _ALNS_PENALTY)
            alns_w_repair[repair_op] = max(_ALNS_W_MIN, alns_w_repair[repair_op] * _ALNS_PENALTY)
            log.info(
                f"iter={iteration:4d}  {op_detail}  "
                f"candidate={routed_cost:.3e}  unscheduled={unscheduled_count}  delta={delta:+.3e}  reject"
            )

        if accept and unscheduled_count == 0 and routed_cost < best_feasible_cost:
            best_feasible_cost   = routed_cost
            best_feasible_snap   = snapshot(state)
            best_feasible_routes = candidate_routes

        pbar.set_postfix(best=f"{best_feasible_cost:.3e}", impr=total_improvements,
                         stale=no_improve, op=op_label, T=f"{T:.1e}", ks=f"{k_scale:.2f}")

        if no_improve >= patience:
            if restarts >= _MAX_RESTARTS:
                stop_reason = 'patience'
                break
            restarts += 1
            no_improve = 0
            T = T0 * _REHEAT_FRAC
            alns_w_break  = {k: 1.0 for k in _COST_BREAK_KEYS}
            alns_w_break['random'] = _RANDOM_INIT_W
            alns_w_break['geo']    = _RANDOM_INIT_W
            alns_w_repair = {k: 1.0 for k in _REPAIR_KEYS}
            log.info(f"iter={iteration:4d}  RESTART {restarts}/{_MAX_RESTARTS}  T={T:.3e}")

    log.info(
        f"=== OPTIMISE END  best_feasible={best_feasible_cost:.3e}  best_penalized={best_clean_cost:.3e}  "
        f"improvements={total_improvements}  sa_accepts={total_sa_accepts}  "
        f"iterations={iteration + 1}  stopped={stop_reason}  "
        f"restarts={restarts}  final_k_scale={k_scale:.3f} ==="
    )
    log.info(f"final break weights:  { {k: f'{v:.2f}' for k, v in alns_w_break.items()} }")
    log.info(f"final repair weights: { {k: f'{v:.2f}' for k, v in alns_w_repair.items()} }")

    if best_feasible_snap is None:
        log.warning("=== NO FEASIBLE SOLUTION FOUND — all states had unscheduled requests ===")
        print("  ERROR: ALNS could not find a fully-scheduled solution.", flush=True)
        return None

    restore(state, instance, best_feasible_snap)
    print("  computing final routes (quality mode)...", flush=True)
    final_routes = solve_routing(state, instance, fast=False, time_limit_seconds=15,
                                 initial_routes=best_feasible_routes)
    final_cost = cost_from_routes(final_routes, instance)['total']
    print(f"  final routed cost: {final_cost:.3e}  (LNS best feasible was {best_feasible_cost:.3e})", flush=True)
    return final_routes
