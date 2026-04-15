from instance import Instance, Request
import bisect
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


def build_state(instance: Instance) -> dict:
    state_dict = dict()

    pool = defaultdict(lambda: defaultdict(int))
    state_dict['pool'] = pool

    loans = defaultdict(lambda: [0] * (instance.config.days + 2))
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

    current_loans = sum(state['loans'][request.machine_type][:delivery_day + 1])
    tool = next(t for t in instance.tools if t.id == request.machine_type)
    if current_loans + request.num_machines > tool.num_available:
        log.debug(f"INFEASIBLE req={request.id} loans exceed available: current={current_loans} + needed={request.num_machines} > available={tool.num_available}")
        return False

    return True


def commit_request(state: dict, instance: Instance, request: Request, delivery_day: int) -> None:
    pickup_day = delivery_day + request.duration
    if not is_feasible(state, instance, request, delivery_day, pickup_day):
        raise ValueError(f"Commiting request ({request.id}) on day {delivery_day} is not feasible")

    state['pool'][request.machine_type][pickup_day] += request.num_machines

    state['loans'][request.machine_type][delivery_day] += request.num_machines
    state['loans'][request.machine_type][pickup_day] -= request.num_machines

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
    entry = next((e for e in state['scheduled'] if e['request'] is request), None)
    if entry is None:
        raise ValueError(f"Request ({request.id}) is not scheduled, cannot uncommit")

    delivery_day = entry['delivery_day']
    pickup_day = entry['pickup_day']

    state['pool'][request.machine_type][pickup_day] -= request.num_machines

    state['loans'][request.machine_type][delivery_day] -= request.num_machines
    state['loans'][request.machine_type][pickup_day] += request.num_machines

    state['stops_per_day'][delivery_day] -= 1
    state['stops_per_day'][pickup_day] -= 1

    state['scheduled'].remove(entry)

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
    state['pool'].clear()
    state['stops_per_day'].clear()
    for v in state['loans'].values():
        v[:] = [0] * len(v)

    state['unscheduled'].clear()
    for req in sorted(instance.requests, key=lambda r: r.latest):
        state['unscheduled'][req.machine_type].append(req)

    for req, delivery_day in snap:
        log.debug(f"RESTORE req={req.id} delivery={delivery_day}")
        commit_request(state, instance, req, delivery_day)
