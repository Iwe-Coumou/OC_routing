from instance import Instance, Request, Config, Tool
import bisect
import logging
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(
    filename='schedule.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(message)s'
)
log = logging.getLogger(__name__)

def build_state(instance: Instance) -> dict:
    state_dict = dict()
    
    pool = defaultdict(lambda: defaultdict(int))
    state_dict['pool'] = pool
    
    loans = defaultdict(lambda: [0]*(instance.config.days+2))
    state_dict['loans'] = loans
    
    stops_per_day = defaultdict(int)
    state_dict['stops_per_day'] = stops_per_day
    
    state_dict['scheduled'] = []
    
    unscheduled = defaultdict(list)
    for req in sorted(instance.requests, key=lambda r: r.latest):
        unscheduled[req.machine_type].append(req)
    state_dict['unscheduled'] = unscheduled
    
    return state_dict
    
def print_state(state: dict) -> None:
    print("=== POOL ===")
    for machine_type, days in state['pool'].items():
        for day, count in sorted(days.items()):
            if count > 0:
                print(f"  type={machine_type} day={day}: {count} units")

    print("=== STOPS PER DAY ===")
    for day, count in sorted(state['stops_per_day'].items()):
        print(f"  day={day}: {count} stops")

    print("=== SCHEDULED ===")
    for entry in state['scheduled']:
        r = entry['request']
        chain = f" (chained from req {entry['chained_from']['request'].id})" if entry['chained_from'] else ""
        print(f"  req={r.id} type={r.machine_type} loc={r.location_id} "
              f"delivery={entry['delivery_day']} pickup={entry['pickup_day']}{chain}")

    print("=== UNSCHEDULED ===")
    for machine_type, reqs in state['unscheduled'].items():
        ids = [r.id for r in reqs]
        print(f"  type={machine_type}: {ids}")
        
def commit_request(state: dict, instance: Instance, request: Request, delivery_day: int, chained_from=None) -> None:
    pickup_day = delivery_day + request.duration
    if not is_feasible(state, instance, request, delivery_day, pickup_day, chained_from):
        raise ValueError(f"Commiting request ({request.id}) on day {delivery_day} is not feasible")

    pool_consumed = 0
    if chained_from:
        pool_consumed = min(state['pool'][request.machine_type][delivery_day], request.num_machines)
        state['pool'][request.machine_type][delivery_day] -= pool_consumed
    state['pool'][request.machine_type][pickup_day] += request.num_machines

    state['loans'][request.machine_type][delivery_day] += request.num_machines
    state['loans'][request.machine_type][pickup_day] -= request.num_machines

    state['stops_per_day'][delivery_day] += 1
    state['stops_per_day'][pickup_day] += 1

    state['scheduled'].append({
        'request': request,
        'delivery_day': delivery_day,
        'pickup_day': pickup_day,
        'chained_from': chained_from,
        'pool_consumed': pool_consumed,
    })

    state['unscheduled'][request.machine_type].remove(request)
    chain_str = f" chained_from=req{chained_from['request'].id}" if chained_from else ""
    log.debug(f"COMMIT req={request.id} type={request.machine_type} delivery={delivery_day} pickup={pickup_day}{chain_str}")
    
def uncommit_request(state: dict, request: Request) -> None:
    entry = next((e for e in state['scheduled'] if e['request'] is request), None)
    if entry is None:
        raise ValueError(f"Request ({request.id}) is not scheduled, cannot uncommit")

    delivery_day = entry['delivery_day']
    pickup_day = entry['pickup_day']

    state['pool'][request.machine_type][pickup_day] -= request.num_machines
    if entry['pool_consumed']:
        state['pool'][request.machine_type][delivery_day] += entry['pool_consumed']

    state['loans'][request.machine_type][delivery_day] -= request.num_machines
    state['loans'][request.machine_type][pickup_day] += request.num_machines

    state['stops_per_day'][delivery_day] -= 1
    state['stops_per_day'][pickup_day] -= 1

    state['scheduled'].remove(entry)

    # re-insert into unscheduled preserving latest-ascending order
    reqs = state['unscheduled'][request.machine_type]
    bisect.insort(reqs, request, key=lambda r: r.latest)
    log.debug(f"UNCOMMIT req={request.id} type={request.machine_type} delivery={delivery_day} pickup={pickup_day}")

def is_feasible(state, instance, request, delivery_day, pickup_day, chained_from=None) -> bool:
    if not request.earliest <= delivery_day <= request.latest:
        log.debug(f"INFEASIBLE req={request.id} window [{request.earliest},{request.latest}] delivery={delivery_day}")
        return False

    if not pickup_day <= instance.config.days:
        log.debug(f"INFEASIBLE req={request.id} pickup_day={pickup_day} > days={instance.config.days}")
        return False

    if chained_from:
        if chained_from['pickup_day'] != delivery_day:
            log.debug(f"INFEASIBLE req={request.id} chain mismatch: source pickup={chained_from['pickup_day']} != delivery={delivery_day}")
            return False
        if state['pool'][request.machine_type][delivery_day] <= 0:
            log.debug(f"INFEASIBLE req={request.id} pool[{request.machine_type}][{delivery_day}]=0, nothing to chain")
            return False

    return True

def try_backwards_extend(state: dict, instance: Instance) -> bool:
    changed = False
    for machine_type, days in state['pool'].items():
        for day, count in list(days.items()):
            if count <= 0:
                continue
            for req in list(state['unscheduled'][machine_type]):
                if req.earliest <= day <= req.latest and count > 0:
                    source_entry = next(e for e in state['scheduled']
                                        if e['pickup_day'] == day and e['request'].machine_type == machine_type)
                    log.debug(f"BACKWARDS_EXTEND req={req.id} onto pool day={day} type={machine_type}")
                    commit_request(state, instance, req, day, chained_from=source_entry)
                    changed = True
                    break  # pool count may have changed, re-scan from outer loop
    return changed

def try_forward_chain(state: dict, instance: Instance, request: Request) -> bool:
    earliest_pickup = request.earliest + request.duration
    latest_pickup = request.latest + request.duration

    candidate = None
    chain_day = None
    for req in state['unscheduled'][request.machine_type]:
        if req is request:
            continue
        overlap_day = max(earliest_pickup, req.earliest)
        if (overlap_day <= min(latest_pickup, req.latest)
                and overlap_day <= instance.config.days
                and state['pool'][request.machine_type][overlap_day] + request.num_machines > 0):
            candidate = req
            chain_day = overlap_day
            break

    if candidate is None:
        return False

    # delivery_day must be chosen so that pickup_day == chain_day
    delivery_day = chain_day - request.duration
    log.debug(f"FORWARD_CHAIN req={request.id} -> candidate=req{candidate.id} chain_day={chain_day}")
    commit_request(state, instance, request, delivery_day)
    source_entry = state['scheduled'][-1]
    commit_request(state, instance, candidate, chain_day, chained_from=source_entry)
    return True

def fallback(state: dict, instance: Instance, request: Request) -> None:
    log.debug(f"FALLBACK req={request.id} type={request.machine_type} delivery={request.earliest}")
    commit_request(state, instance, request, request.earliest)

def next_unscheduled(state: dict):
    best = None
    for reqs in state['unscheduled'].values():
        if reqs and (best is None or reqs[0].latest < best.latest or
                     (reqs[0].latest == best.latest and reqs[0].earliest < best.earliest)):
            best = reqs[0]
    return best

def build_schedule(instance: Instance) -> list:
    state = build_state(instance)

    while True:
        changed = try_backwards_extend(state, instance)
        if changed:
            continue

        request = next_unscheduled(state)
        if request is None:
            break

        chained = try_forward_chain(state, instance, request)
        if not chained:
            fallback(state, instance, request)

    return state

def compute_cost(state: dict, instance: Instance) -> float:
    tool_by_type = {t.id: t for t in instance.tools}

    # peak machine usage cost per type
    tool_cost = 0
    for machine_type, diff in state['loans'].items():
        peak = 0
        current = 0
        for delta in diff:
            current += delta
            peak = max(peak, current)
        tool_cost += peak * tool_by_type[machine_type].cost

    # fallback penalty: each unchained request costs one extra vehicle-day
    num_fallbacks = sum(1 for e in state['scheduled'] if e['chained_from'] is None)
    fallback_penalty = num_fallbacks * instance.config.vehicle_day_cost

    return tool_cost + fallback_penalty

def repair(state: dict, instance: Instance) -> None:
    """Re-schedule all requests currently in unscheduled using the greedy."""
    while True:
        changed = try_backwards_extend(state, instance)
        if changed:
            continue
        request = next_unscheduled(state)
        if request is None:
            break
        if not try_forward_chain(state, instance, request):
            fallback(state, instance, request)

def snapshot(state: dict) -> list:
    """Save current schedule as (request, delivery_day, chained_from_id)."""
    return [
        (e['request'], e['delivery_day'], e['chained_from']['request'].id if e['chained_from'] else None, e['pool_consumed'])
        for e in state['scheduled']
    ]

def restore(state: dict, instance: Instance, snap: list) -> None:
    """Restore a snapshot, reconstructing chained_from links by request id."""
    for entry in list(state['scheduled']):
        uncommit_request(state, entry['request'])
    committed = {}
    for req, delivery_day, chained_from_id, _ in sorted(snap, key=lambda x: x[1]):
        chained_from = committed.get(chained_from_id)
        if chained_from and state['pool'][req.machine_type][delivery_day] <= 0:
            chained_from = None
        log.debug(f"RESTORE req={req.id} delivery={delivery_day} chained_from_id={chained_from_id} pool={state['pool'][req.machine_type][delivery_day]}")
        commit_request(state, instance, req, delivery_day, chained_from=chained_from)
        committed[req.id] = state['scheduled'][-1]

def optimize_lns(state: dict, instance: Instance, iterations: int = 100, destroy_fraction: float = 0.2) -> float:
    """LNS: destroy a random subset of requests and repair. Returns best cost achieved."""
    import random

    best_cost = compute_cost(state, instance)
    best_snap = snapshot(state)
    log.debug(f"LNS start cost={best_cost}")

    pbar = tqdm(range(iterations), desc="LNS", unit="iter")
    for i in pbar:
        sampled = random.sample(state['scheduled'], max(1, int(len(state['scheduled']) * destroy_fraction)))
        seed_ids = {e['request'].id for e in sampled}
        seed_reqs = [e['request'] for e in sampled]
        # dechain any children whose source is being destroyed
        for entry in state['scheduled']:
            if entry['chained_from'] is not None and entry['chained_from']['request'].id in seed_ids:
                state['pool'][entry['request'].machine_type][entry['delivery_day']] += entry['pool_consumed']
                entry['chained_from'] = None
                entry['pool_consumed'] = 0
                log.debug(f"DECHAIN req={entry['request'].id} delivery={entry['delivery_day']}")
        for req in seed_reqs:
            uncommit_request(state, req)

        repair(state, instance)

        cost = compute_cost(state, instance)
        log.debug(f"LNS iter={i} cost={cost} best={best_cost}")

        if cost < best_cost:
            best_cost = cost
            best_snap = snapshot(state)
            log.debug(f"LNS improvement at iter={i} cost={best_cost}")
            pbar.set_postfix(best=best_cost, improved=True)
        else:
            restore(state, instance, best_snap)
            pbar.set_postfix(best=best_cost, improved=False)

    return best_cost

def optimize_rechain(state: dict, instance: Instance) -> int:
    """Try to rechain fallback-scheduled requests. Returns number of requests rechained."""
    rechained = 0
    fallbacks = [e for e in state['scheduled'] if e['chained_from'] is None]

    for entry in fallbacks:
        request = entry['request']
        uncommit_request(state, request)
        chained = try_forward_chain(state, instance, request)
        if chained:
            rechained += 1
        else:
            # restore original day
            commit_request(state, instance, request, entry['delivery_day'])

    return rechained

def validate_schedule(scheduled: list, instance: Instance) -> bool:
    valid = True

    # check all requests are scheduled exactly once
    scheduled_ids = [e['request'].id for e in scheduled]
    expected_ids = [r.id for r in instance.requests]
    if sorted(scheduled_ids) != sorted(expected_ids):
        print(f"FAIL: scheduled requests {sorted(scheduled_ids)} != expected {sorted(expected_ids)}")
        valid = False

    # check window and pickup day constraints per entry
    for entry in scheduled:
        r = entry['request']
        d = entry['delivery_day']
        p = entry['pickup_day']

        if not (r.earliest <= d <= r.latest):
            print(f"FAIL: req={r.id} delivery_day={d} outside window [{r.earliest}, {r.latest}]")
            valid = False

        if p != d + r.duration:
            print(f"FAIL: req={r.id} pickup_day={p} != delivery_day={d} + duration={r.duration}")
            valid = False

        if p > instance.config.days:
            print(f"FAIL: req={r.id} pickup_day={p} exceeds horizon {instance.config.days}")
            valid = False

        if entry['chained_from'] and entry['chained_from']['pickup_day'] != d:
            print(f"FAIL: req={r.id} delivery_day={d} != chained source pickup_day={entry['chained_from']['pickup_day']}")
            valid = False

    # check peak machine usage per type never exceeds num_available
    tool_by_type = {t.id: t for t in instance.tools}
    loans = defaultdict(lambda: [0] * (instance.config.days + 2))
    for entry in scheduled:
        r = entry['request']
        loans[r.machine_type][entry['delivery_day']] += r.num_machines
        loans[r.machine_type][entry['pickup_day']] -= r.num_machines

    for machine_type, diff in loans.items():
        current = 0
        for day, delta in enumerate(diff):
            current += delta
            limit = tool_by_type[machine_type].num_available
            if current > limit:
                print(f"FAIL: type={machine_type} day={day} concurrent use={current} exceeds available={limit}")
                valid = False

    if valid:
        print("OK: schedule is valid")
    return valid
