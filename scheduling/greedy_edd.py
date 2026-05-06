import logging
import random
from instance import Instance, Request
from .state import build_state, commit_request, is_feasible

log = logging.getLogger(__name__)


def _first_feasible_day(state: dict, instance: Instance, request: Request) -> int | None:
    """Return the earliest feasible delivery day in the request's time window."""
    for d in range(request.earliest, request.latest + 1):
        if is_feasible(state, instance, request, d, d + request.duration):
            return d
    return None


def place_unscheduled(state: dict, instance: Instance, key=None, randomize=False) -> None:
    """Place unscheduled requests into the schedule.

    Ordering: earliest deadline first by default; alternate insertion orders can
    be used during LNS to explore different schedule structures.

    Called by build_schedule for initial construction and by LNS after each
    destroy step.  Any request that has no feasible day is left in unscheduled
    and logged; the unscheduled penalty in compute_cost_estimate ensures LNS
    never accepts such a state as an improvement.
    """
    requests = [r for reqs in state['unscheduled'].values() for r in reqs]
    if randomize:
        random.shuffle(requests)
    else:
        key = key or (lambda r: (r.latest, r.num_machines * r.duration))
        requests.sort(key=key)

    for request in requests:
        day = _first_feasible_day(state, instance, request)
        if day is None:
            log.warning(f"EDD req={request.id} has no feasible day — leaving unscheduled")
            continue
        commit_request(state, instance, request, day)


def build_schedule(instance: Instance) -> dict:
    """Greedy scheduler: EDD ordering + earliest-feasible-day placement.

    Sorts requests by earliest deadline (smallest latest-day first), breaking
    ties by smallest resource demand (num_machines * duration).  Each request
    is placed on its earliest feasible delivery day.

    This ordering produces a complete, capacity-feasible initial schedule on
    all benchmark instances without any repair or force-placement.
    """
    state = build_state(instance)
    place_unscheduled(state, instance)
    return state
