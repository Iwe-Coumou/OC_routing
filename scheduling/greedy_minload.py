import logging
from instance import Instance, Request
from .state import build_state, commit_request, is_feasible

log = logging.getLogger(__name__)


def _next_unscheduled(state: dict):
    best = None
    for reqs in state['unscheduled'].values():
        if reqs and (best is None or reqs[0].latest < best.latest or
                     (reqs[0].latest == best.latest and reqs[0].earliest < best.earliest)):
            best = reqs[0]
    return best


def _best_delivery_day(state: dict, instance: Instance, request: Request) -> int | None:
    """Return the feasible delivery day that minimizes concurrent tool usage.

    For each candidate day d, the concurrent usage is:
        sum(loans[machine_type][:d+1])  (= machines currently out on loan on day d)
    Choosing the minimum delivers when fewest machines are already out,
    directly targeting a lower peak.
    """
    best_day = None
    best_net = float('inf')

    for d in range(request.earliest, request.latest + 1):
        pickup_day = d + request.duration
        if not is_feasible(state, instance, request, d, pickup_day):
            continue
        net = sum(state['loans'][request.machine_type][:d + 1])
        if net < best_net:
            best_net = net
            best_day = d

    return best_day


def _register_implicit_chains(state: dict) -> None:
    """After scheduling, record implicit chains where one entry's pickup_day
    matches another's delivery_day for the same tool type.

    This lets the LNS destroy_chain operator and repair's pool tracking
    work correctly on minload schedules.
    """
    pickup_index = {}
    for e in state['scheduled']:
        key = (e['request'].machine_type, e['pickup_day'])
        pickup_index[key] = e

    for e in state['scheduled']:
        r = e['request']
        source = pickup_index.get((r.machine_type, e['delivery_day']))
        if source and source is not e:
            e['chained_from'] = source
            consumed = min(state['pool'][r.machine_type][e['delivery_day']], r.num_machines)
            e['pool_consumed'] = consumed
            state['pool'][r.machine_type][e['delivery_day']] -= consumed
            log.debug(f"IMPLICIT_CHAIN req={r.id} chained_from=req{source['request'].id} day={e['delivery_day']}")


def repair(state: dict, instance: Instance) -> None:
    """Re-schedule all unscheduled requests using minload scoring.

    Processes requests in deadline order and places each on the delivery day
    that currently has the lowest concurrent tool usage for its type.
    """
    while True:
        request = _next_unscheduled(state)
        if request is None:
            break

        day = _best_delivery_day(state, instance, request)
        if day is None:
            log.debug(f"MINLOAD REPAIR no best day for req={request.id}, falling back to earliest feasible")
            for d in range(request.earliest, request.latest + 1):
                pickup_day = d + request.duration
                if is_feasible(state, instance, request, d, pickup_day):
                    day = d
                    break

        if day is not None:
            commit_request(state, instance, request, day)
        else:
            log.debug(f"MINLOAD REPAIR req={request.id} has no feasible day at all, leaving unscheduled")
            state['unscheduled'][request.machine_type].remove(request)


def build_schedule(instance: Instance) -> dict:
    """Greedy scheduler: process requests in deadline order, place each on the
    delivery day that currently has the lowest net machine load for its type."""
    state = build_state(instance)

    while True:
        request = _next_unscheduled(state)
        if request is None:
            break

        day = _best_delivery_day(state, instance, request)
        if day is None:
            log.debug(f"MINLOAD no feasible day for req={request.id}, falling back to earliest feasible")
            for d in range(request.earliest, request.latest + 1):
                pickup_day = d + request.duration
                if is_feasible(state, instance, request, d, pickup_day):
                    day = d
                    break

        if day is not None:
            commit_request(state, instance, request, day)
        else:
            log.debug(f"MINLOAD req={request.id} has no feasible day at all, leaving unscheduled")
            state['unscheduled'][request.machine_type].remove(request)

    _register_implicit_chains(state)
    return state
