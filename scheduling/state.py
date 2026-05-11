import logging
from collections import defaultdict
from instance import Instance, Request

log = logging.getLogger(__name__)


def build_state(instance: Instance) -> dict:
    state_dict = dict()

    loans = defaultdict(lambda: [0] * (instance.config.days + 2))
    state_dict['loans'] = loans

    # Per-day pickup counts kept separately so is_feasible() can check the
    # within-day peak (after deliveries, before pickups), matching Validate._calculateSolution.
    pickups_per_day = defaultdict(lambda: [0] * (instance.config.days + 2))
    state_dict['pickups_per_day'] = pickups_per_day

    state_dict['scheduled'] = []

    unscheduled = defaultdict(list)
    for req in sorted(instance.requests, key=lambda r: r.latest):
        unscheduled[req.machine_type].append(req)
    state_dict['unscheduled'] = unscheduled

    return state_dict


def is_feasible(state, instance, request, delivery_day, pickup_day) -> bool:
    if not request.earliest <= delivery_day <= request.latest:
        log.debug(f"INFEASIBLE req={request.id} window [{request.earliest},{request.latest}] delivery={delivery_day}")
        return False

    if not pickup_day <= instance.config.days:
        log.debug(f"INFEASIBLE req={request.id} pickup_day={pickup_day} > days={instance.config.days}")
        return False

    tool = next(t for t in instance.tools if t.id == request.machine_type)
    diff = state['loans'][request.machine_type]
    pickups = state['pickups_per_day'][request.machine_type]
    running = 0
    # Loop includes pickup_day: tools are still at the customer before pickup (morning peak).
    for day in range(pickup_day + 1):
        running += diff[day]
        if day >= delivery_day:
            # Within-day peak: add back pickups[day] to get the before-pickup count.
            peak = running + pickups[day] + request.num_machines
            if peak > tool.num_available:
                log.debug(f"INFEASIBLE req={request.id} peak on day={day}: {peak} > available={tool.num_available}")
                return False

    return True


def commit_request(state: dict, instance: Instance, request: Request, delivery_day: int) -> None:
    pickup_day = delivery_day + request.duration
    if not is_feasible(state, instance, request, delivery_day, pickup_day):
        raise ValueError(f"Commiting request ({request.id}) on day {delivery_day} is not feasible")

    state['loans'][request.machine_type][delivery_day] += request.num_machines
    state['loans'][request.machine_type][pickup_day] -= request.num_machines
    state['pickups_per_day'][request.machine_type][pickup_day] += request.num_machines

    state['scheduled'].append({
        'request': request,
        'delivery_day': delivery_day,
        'pickup_day': pickup_day,
    })

    state['unscheduled'][request.machine_type].remove(request)
    log.debug(f"COMMIT req={request.id} type={request.machine_type} delivery={delivery_day} pickup={pickup_day}")


def uncommit_request(state: dict, request: Request) -> None:
    idx = next((i for i, e in enumerate(state['scheduled']) if e['request'] is request), None)
    if idx is None:
        raise ValueError(f"Request ({request.id}) is not scheduled, cannot uncommit")

    entry = state['scheduled'][idx]
    delivery_day = entry['delivery_day']
    pickup_day = entry['pickup_day']

    state['loans'][request.machine_type][delivery_day] -= request.num_machines
    state['loans'][request.machine_type][pickup_day] += request.num_machines
    state['pickups_per_day'][request.machine_type][pickup_day] -= request.num_machines

    del state['scheduled'][idx]

    reqs = state['unscheduled'][request.machine_type]
    reqs.append(request)
    reqs.sort(key=lambda r: r.latest)
    log.debug(f"UNCOMMIT req={request.id} type={request.machine_type} delivery={delivery_day} pickup={pickup_day}")


def snapshot(state: dict) -> list:
    return [(e['request'], e['delivery_day']) for e in state['scheduled']]


def restore(state: dict, instance: Instance, snap: list) -> None:
    state['scheduled'].clear()
    for v in state['loans'].values():
        v[:] = [0] * len(v)
    state['pickups_per_day'].clear()

    state['unscheduled'].clear()
    for req in sorted(instance.requests, key=lambda r: r.latest):
        state['unscheduled'][req.machine_type].append(req)

    for req, delivery_day in snap:
        log.debug(f"RESTORE req={req.id} delivery={delivery_day}")
        commit_request(state, instance, req, delivery_day)


def _first_feasible_day(state: dict, instance: Instance, request: Request) -> int | None:
    for d in range(request.earliest, request.latest + 1):
        if is_feasible(state, instance, request, d, d + request.duration):
            return d
    return None


def place_unscheduled(state: dict, instance: Instance, key=None) -> None:
    requests = [r for reqs in state['unscheduled'].values() for r in reqs]
    key = key or (lambda r: (r.latest, r.num_machines * r.duration))
    requests.sort(key=key)

    for request in requests:
        day = _first_feasible_day(state, instance, request)
        if day is None:
            log.warning(f"place_unscheduled: req={request.id} has no feasible day — leaving unscheduled")
            continue
        commit_request(state, instance, request, day)


def place_unscheduled_mrv(state: dict, instance: Instance) -> None:
    """Dynamic MRV ordering: at each step place the most constrained unscheduled request."""
    while True:
        best_req = None
        best_key = (float('inf'), float('inf'), float('inf'))
        for reqs in state['unscheduled'].values():
            for req in reqs:
                count = sum(
                    1 for d in range(req.earliest, req.latest + 1)
                    if is_feasible(state, instance, req, d, d + req.duration)
                )
                key = (count, req.latest - req.earliest, req.latest)
                if key < best_key:
                    best_key = key
                    best_req = req
        if best_req is None:
            break
        day = _first_feasible_day(state, instance, best_req)
        if day is None:
            log.warning(f"MRV req={best_req.id} has no feasible day — leaving unscheduled")
            break
        commit_request(state, instance, best_req, day)


def repair_by_ejection_chain(state: dict, instance: Instance) -> None:
    """Place unscheduled requests via an ejection chain.

    For each unscheduled request (most-constrained first), try direct
    placement. If blocked, eject a same-type scheduled request to make room
    and let it re-enter the unscheduled pool for the next iteration.
    Each request can be ejected at most once, guaranteeing termination in
    O(n) outer iterations.
    """
    displaced: set = set()  # ejected this call — cannot be ejected again
    failed: set = set()     # no ejectee found — skip on future iterations

    changed = True
    while changed:
        changed = False

        candidates = []
        for reqs in state['unscheduled'].values():
            for req in reqs:
                if req.id in failed:
                    continue
                count = sum(
                    1 for d in range(req.earliest, req.latest + 1)
                    if is_feasible(state, instance, req, d, d + req.duration)
                )
                candidates.append((count, req.latest - req.earliest, req.latest, req.id, req))

        if not candidates:
            break

        candidates.sort()

        for _, _, _, _, u in candidates:
            day = _first_feasible_day(state, instance, u)
            if day is not None:
                commit_request(state, instance, u, day)
                changed = True
                break

            placed = False
            for entry in list(state['scheduled']):
                s = entry['request']
                if s.machine_type != u.machine_type or s.id in displaced:
                    continue
                s_day = entry['delivery_day']
                uncommit_request(state, s)
                u_day = _first_feasible_day(state, instance, u)
                if u_day is not None:
                    commit_request(state, instance, u, u_day)
                    s_new_day = _first_feasible_day(state, instance, s)
                    if s_new_day is not None:
                        commit_request(state, instance, s, s_new_day)
                        displaced.add(s.id)
                        placed = True
                        changed = True
                        break
                    uncommit_request(state, u)
                commit_request(state, instance, s, s_day)

            if not placed:
                failed.add(u.id)

            if changed:
                break


CONSTRUCTION_KEYS = {
    'edd':   lambda r: (r.latest, r.num_machines * r.duration),
    'tight': lambda r: (r.latest - r.earliest, r.latest),
    'heavy': lambda r: (-r.num_machines, r.latest),
    'late':  lambda r: (-r.earliest, r.latest),
}

_ORDERINGS = list(CONSTRUCTION_KEYS.values())


def build_schedule(instance: Instance) -> dict:
    from scheduling.cost import cost_breakdown

    best_state = None
    best_cost = float('inf')

    for key in _ORDERINGS:
        state = build_state(instance)
        place_unscheduled(state, instance, key=key)
        n_unscheduled = sum(len(v) for v in state['unscheduled'].values())
        if n_unscheduled > 0:
            repair_by_ejection_chain(state, instance)
            n_unscheduled = sum(len(v) for v in state['unscheduled'].values())
        if n_unscheduled > 0:
            continue
        cost = cost_breakdown(state, instance)['total']
        if cost < best_cost:
            best_cost = cost
            best_state = state

    # MRV + ejection chain
    mrv_state = build_state(instance)
    place_unscheduled_mrv(mrv_state, instance)
    n_mrv = sum(len(v) for v in mrv_state['unscheduled'].values())
    if n_mrv > 0:
        repair_by_ejection_chain(mrv_state, instance)
        n_mrv = sum(len(v) for v in mrv_state['unscheduled'].values())
    if n_mrv == 0:
        cost = cost_breakdown(mrv_state, instance)['total']
        if cost < best_cost:
            best_state = mrv_state
    elif best_state is None:
        best_state = mrv_state

    return best_state


def build_schedule_single(instance: Instance, key) -> dict:
    state = build_state(instance)
    place_unscheduled(state, instance, key=key)
    n = sum(len(v) for v in state['unscheduled'].values())
    if n > 0:
        repair_by_ejection_chain(state, instance)
        n = sum(len(v) for v in state['unscheduled'].values())
    if n > 0:
        log.warning(f"build_schedule_single: {n} requests could not be placed")
    return state


def validate_schedule(scheduled: list, instance: Instance) -> bool:
    valid = True

    scheduled_ids = [e['request'].id for e in scheduled]
    expected_ids  = [r.id for r in instance.requests]
    if sorted(scheduled_ids) != sorted(expected_ids):
        print(f"FAIL: scheduled requests {sorted(scheduled_ids)} != expected {sorted(expected_ids)}")
        valid = False

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

    tool_by_type = {t.id: t for t in instance.tools}
    loans = defaultdict(lambda: [0] * (instance.config.days + 2))
    pickups = defaultdict(lambda: [0] * (instance.config.days + 2))
    for entry in scheduled:
        r = entry['request']
        loans[r.machine_type][entry['delivery_day']] += r.num_machines
        loans[r.machine_type][entry['pickup_day']]   -= r.num_machines
        pickups[r.machine_type][entry['pickup_day']] += r.num_machines

    for machine_type, diff in loans.items():
        current = 0
        limit = tool_by_type[machine_type].num_available
        for day, delta in enumerate(diff):
            current += delta
            # Within-day peak: after deliveries, before pickups.
            peak = current + pickups[machine_type][day]
            if peak > limit:
                print(f"FAIL: type={machine_type} day={day} peak use={peak} exceeds available={limit}")
                valid = False

    if valid:
        log.debug("OK: schedule is valid")
    return valid
