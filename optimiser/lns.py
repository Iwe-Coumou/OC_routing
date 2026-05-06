import random
import logging
from tqdm import tqdm
from scheduling.state import snapshot, restore, uncommit_request
from scheduling.greedy_edd import place_unscheduled
from routing import solve_routing
from routing.export import cost_from_routes
from .break_fns import (
    break_tool_cost,
    break_vehicle_cost,
    break_vehicle_day_cost,
    break_distance_cost,
)
from .repair_fns import (
    repair_tool_cost,
    repair_vehicle_cost,
    repair_vehicle_day_cost,
    repair_distance_cost,
)

log = logging.getLogger(__name__)

_BREAK_REPAIR = {
    'tool':         (break_tool_cost,        repair_tool_cost),
    'vehicle':      (break_vehicle_cost,     repair_vehicle_cost),
    'vehicle_days': (break_vehicle_day_cost, repair_vehicle_day_cost),
    'distance':     (break_distance_cost,    repair_distance_cost),
}
_NEEDS_ROUTES = {'vehicle', 'vehicle_days', 'distance'}


def optimize(
    state: dict,
    instance,
    iterations: int = 300,
    patience: int = 150,
    cost_op_prob: float = 0.2,
    epsilon: float = 0.25,
    max_destroy: int = 30,
) -> dict:
    """LNS optimiser using true routed cost as the acceptance criterion.

    Primary operator (1 - cost_op_prob): random destroy + EDD repair.
    Secondary operator (cost_op_prob): cost-targeted break + matching repair with
    ε-greedy day selection; falls back to EDD if repair leaves anything unscheduled.

    After every destroy+repair cycle, solve_routing(fast=True) is called to get
    the true cost for acceptance comparison — no estimated-cost proxies.

    Returns the RouteSet corresponding to the best accepted schedule.
    """
    print("  computing initial routing cost...", flush=True)
    current_routes = solve_routing(state, instance, fast=True)
    best_cost = cost_from_routes(current_routes, instance)['total']
    best_snap = snapshot(state)
    breakdown = cost_from_routes(current_routes, instance)
    print(f"  initial routed cost: {best_cost:.3e}", flush=True)

    no_improve = 0
    total_improvements = 0

    log.info(
        f"=== OPTIMISE START  instance={instance.name}  "
        f"initial={best_cost:.3e}  "
        f"tool={breakdown['tool']:.3e}  vehicle={breakdown['vehicle']:.3e}  "
        f"veh_days={breakdown['vehicle_days']:.3e}  distance={breakdown['distance']:.3e} ==="
    )

    stop_reason = 'iterations'
    pbar = tqdm(range(iterations), desc="LNS", unit="iter")
    for iteration in pbar:
        snap = snapshot(state)

        if random.random() < cost_op_prob:
            # --- cost-targeted operator ---
            driver = max(
                ['tool', 'vehicle', 'vehicle_days', 'distance'],
                key=lambda k: breakdown[k],
            )
            break_fn, repair_fn = _BREAK_REPAIR[driver]
            k = min(max_destroy, max(1, int(len(state['scheduled']) * random.uniform(0.1, 0.25))))

            if driver in _NEEDS_ROUTES:
                targets = break_fn(state, instance, current_routes, k)
            else:
                targets = break_fn(state, instance, k)

            for req in targets:
                uncommit_request(state, req)

            repair_fn(state, instance, epsilon=epsilon)

            used_edd_fallback = any(v for v in state['unscheduled'].values())
            if used_edd_fallback:
                place_unscheduled(state, instance)

            op_label = f"cost:{driver[:3].upper()}"
            op_detail = f"op=cost driver={driver} k={k}" + (" +edd_fallback" if used_edd_fallback else "")
        else:
            # --- random operator ---
            # Use a cost-targeted repair at high epsilon instead of pure EDD.
            # EDD always picks the earliest feasible day, which tends to put requests
            # back exactly where they were (freed capacity on their original days),
            # producing no schedule change and thus no routing improvement.
            k = min(max_destroy, max(1, int(len(state['scheduled']) * random.uniform(0.1, 0.3))))
            targets = random.sample(state['scheduled'], k)
            for e in targets:
                uncommit_request(state, e['request'])

            rand_driver = random.choice(['tool', 'vehicle', 'vehicle_days', 'distance'])
            _, rand_repair = _BREAK_REPAIR[rand_driver]
            rand_repair(state, instance, epsilon=0.5)
            if any(v for v in state['unscheduled'].values()):
                place_unscheduled(state, instance)

            op_label = "rand"
            op_detail = f"op=rand k={k} repair={rand_driver}"

        unscheduled_count = sum(len(v) for v in state['unscheduled'].values())
        if unscheduled_count > 0:
            log.warning(
                f"iter={iteration:4d}  {op_detail}  REJECT (unscheduled={unscheduled_count})"
            )
            restore(state, instance, snap)
            no_improve += 1
            pbar.set_postfix(best=f"{best_cost:.3e}", impr=total_improvements,
                             stale=no_improve, op=op_label)
            if no_improve >= patience:
                stop_reason = 'patience'
                break
            continue

        candidate_routes = solve_routing(state, instance, fast=True)
        candidate_cost = cost_from_routes(candidate_routes, instance)['total']
        delta = candidate_cost - best_cost

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_snap = snapshot(state)
            current_routes = candidate_routes
            breakdown = cost_from_routes(current_routes, instance)
            no_improve = 0
            total_improvements += 1
            log.info(
                f"iter={iteration:4d}  {op_detail}  "
                f"candidate={candidate_cost:.3e}  delta={delta:+.3e}  ACCEPT  "
                f"[tool={breakdown['tool']:.3e}  vehicle={breakdown['vehicle']:.3e}  "
                f"veh_days={breakdown['vehicle_days']:.3e}  distance={breakdown['distance']:.3e}]"
            )
        else:
            restore(state, instance, snap)
            no_improve += 1
            log.info(
                f"iter={iteration:4d}  {op_detail}  "
                f"candidate={candidate_cost:.3e}  delta={delta:+.3e}  reject (routing)"
            )

        pbar.set_postfix(best=f"{best_cost:.3e}", impr=total_improvements,
                         stale=no_improve, op=op_label)

        if no_improve >= patience:
            stop_reason = 'patience'
            break

    restore(state, instance, best_snap)
    log.info(
        f"=== OPTIMISE END  best={best_cost:.3e}  improvements={total_improvements}  "
        f"iterations={iteration + 1}  stopped={stop_reason} ==="
    )
    print("  computing final routes (quality mode)...", flush=True)
    final_routes = solve_routing(state, instance, fast=False, time_limit_seconds=10, initial_routes=current_routes)
    final_cost = cost_from_routes(final_routes, instance)['total']
    print(f"  final routed cost: {final_cost:.3e}  (LNS best was {best_cost:.3e})", flush=True)
    return final_routes
