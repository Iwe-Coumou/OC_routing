from instance import Instance, Request
import bisect
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


def build_state(instance: Instance) -> dict:
    state_dict = dict()

    loans = defaultdict(lambda: [0] * (instance.config.days + 2))
    state_dict['loans'] = loans

    # Per-day pickup counts per machine type.  Kept separately so is_feasible()
    # can check the within-day peak (after deliveries, before pickups), which is
    # what Validate._calculateSolution measures.
    pickups_per_day = defaultdict(lambda: [0] * (instance.config.days + 2))
    state_dict['pickups_per_day'] = pickups_per_day

    stops_per_day = defaultdict(int)
    state_dict['stops_per_day'] = stops_per_day

    state_dict['scheduled'] = []

    unscheduled = defaultdict(list)
    for req in sorted(instance.requests, key=lambda r: r.latest):
        unscheduled[req.machine_type].append(req)
    state_dict['unscheduled'] = unscheduled

    return state_dict


def print_state(state: dict) -> None:
    print("=== STOPS PER DAY ===")
    for day, count in sorted(state['stops_per_day'].items()):
        print(f"  day={day}: {count} stops")

    print("=== SCHEDULED ===")
    for entry in state['scheduled']:
        r = entry['request']
        print(f"  req={r.id} type={r.machine_type} loc={r.location_id} "
              f"delivery={entry['delivery_day']} pickup={entry['pickup_day']}")

    print("=== UNSCHEDULED ===")
    for machine_type, reqs in state['unscheduled'].items():
        ids = [r.id for r in reqs]
        print(f"  type={machine_type}: {ids}")


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
    # Loop includes pickup_day: on that day the request's tools are still at
    # the customer before the vehicle arrives (morning peak).
    for day in range(pickup_day + 1):
        running += diff[day]
        if day >= delivery_day:
            # Within-day peak: running is end-of-day net (pickups subtracted),
            # so adding pickups[day] back gives the before-pickup count.
            # The new request's machines are at customers from delivery_day
            # through (and including) pickup_day.
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

    state['stops_per_day'][delivery_day] += 1
    state['stops_per_day'][pickup_day] += 1

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

    state['stops_per_day'][delivery_day] -= 1
    state['stops_per_day'][pickup_day] -= 1

    del state['scheduled'][idx]

    # re-insert into unscheduled preserving latest-ascending order
    reqs = state['unscheduled'][request.machine_type]
    bisect.insort(reqs, request, key=lambda r: r.latest)
    log.debug(f"UNCOMMIT req={request.id} type={request.machine_type} delivery={delivery_day} pickup={pickup_day}")


def snapshot(state: dict) -> list:
    """Save current schedule as (request, delivery_day)."""
    return [(e['request'], e['delivery_day']) for e in state['scheduled']]


def restore(state: dict, instance: Instance, snap: list) -> None:
    """Restore a snapshot.

    Resets state directly instead of calling uncommit_request n times,
    avoiding the O(n²) list.remove cost of the naive approach.
    """
    state['scheduled'].clear()
    state['stops_per_day'].clear()
    for v in state['loans'].values():
        v[:] = [0] * len(v)
    state['pickups_per_day'].clear()

    state['unscheduled'].clear()
    for req in sorted(instance.requests, key=lambda r: r.latest):
        state['unscheduled'][req.machine_type].append(req)

    for req, delivery_day in snap:
        log.debug(f"RESTORE req={req.id} delivery={delivery_day}")
        commit_request(state, instance, req, delivery_day)
