import logging
from instance import Instance, Request
from .state import commit_request

log = logging.getLogger(__name__)


def next_unscheduled(state: dict):
    best = None
    for reqs in state['unscheduled'].values():
        if reqs and (best is None or reqs[0].latest < best.latest or
                     (reqs[0].latest == best.latest and reqs[0].earliest < best.earliest)):
            best = reqs[0]
    return best


def fallback(state: dict, instance: Instance, request: Request) -> None:
    log.debug(f"FALLBACK req={request.id} type={request.machine_type} delivery={request.earliest}")
    commit_request(state, instance, request, request.earliest)


def try_backwards_extend(state: dict, instance: Instance) -> bool:
    changed = False
    for machine_type, days in state['pool'].items():
        for day, count in list(days.items()):
            if count <= 0:
                continue
            for req in list(state['unscheduled'][machine_type]):
                if req.earliest <= day <= req.latest:
                    source_entry = next(e for e in state['scheduled']
                                        if e['pickup_day'] == day and e['request'].machine_type == machine_type)
                    log.debug(f"BACKWARDS_EXTEND req={req.id} onto pool day={day} type={machine_type}")
                    commit_request(state, instance, req, day, chained_from=source_entry)
                    changed = True
                    break
    return changed


def try_forward_chain(state: dict, instance: Instance, request: Request) -> bool:
    earliest_pickup = request.earliest + request.duration
    latest_pickup   = request.latest   + request.duration

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

    delivery_day = chain_day - request.duration
    log.debug(f"FORWARD_CHAIN req={request.id} -> candidate=req{candidate.id} chain_day={chain_day}")
    commit_request(state, instance, request, delivery_day)
    source_entry = state['scheduled'][-1]
    commit_request(state, instance, candidate, chain_day, chained_from=source_entry)
    return True


def repair(state: dict, instance: Instance) -> None:
    """Re-schedule all unscheduled requests using the chain-aware greedy."""
    while True:
        changed = try_backwards_extend(state, instance)
        if changed:
            continue
        request = next_unscheduled(state)
        if request is None:
            break
        if not try_forward_chain(state, instance, request):
            fallback(state, instance, request)
