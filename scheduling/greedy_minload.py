import bisect
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
    """Return the feasible delivery day that minimises the worst-case peak load
    across the entire loan window [delivery_day, pickup_day].

    Scoring by worst peak (rather than net load at delivery day) prevents the
    greedy from choosing days whose pickup end is already crowded, which is the
    root cause of requests being left unscheduled by the initial greedy pass.
    """
    best_day = None
    best_peak = float('inf')

    diff = state['loans'][request.machine_type]
    pickups = state['pickups_per_day'][request.machine_type]

    for d in range(request.earliest, request.latest + 1):
        pickup_day = d + request.duration
        if not is_feasible(state, instance, request, d, pickup_day):
            continue

        # Compute worst peak across [d, pickup_day] with this request included.
        # This mirrors exactly what is_feasible checks, so the greedy stays
        # aligned with the feasibility criterion.
        running = 0
        worst = 0
        for day in range(pickup_day + 1):
            running += diff[day]
            if day >= d:
                worst = max(worst, running + pickups[day] + request.num_machines)

        if worst < best_peak:
            best_peak = worst
            best_day = d

    return best_day



def repair(state: dict, instance: Instance) -> None:
    """Re-schedule all unscheduled requests using minload scoring.

    Iterates until no further progress can be made.  Requests that cannot be
    placed in the current pass are left in unscheduled so that a later LNS
    iteration (which destroys other requests) can free up capacity for them.
    """
    while True:
        progress = False
        skipped = []
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
                progress = True
            else:
                log.debug(f"MINLOAD REPAIR req={request.id} has no feasible day, skipping for now")
                state['unscheduled'][request.machine_type].remove(request)
                skipped.append(request)

        # Re-insert skipped requests for the next pass.
        for req in skipped:
            bisect.insort(state['unscheduled'][req.machine_type], req, key=lambda r: r.latest)

        if not progress:
            break  # No new requests were placed; stop to avoid infinite loop.


def build_schedule(instance: Instance) -> dict:
    """Greedy scheduler: process requests in deadline order, place each on the
    delivery day that currently has the lowest net machine load for its type.

    Requests that cannot be placed in the initial pass are left in unscheduled
    so that LNS repair can recover them after destroying blocking requests.
    """
    state = build_state(instance)

    # Iterate until no further progress (same termination logic as repair).
    while True:
        progress = False
        skipped = []
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
                progress = True
            else:
                log.debug(f"MINLOAD req={request.id} has no feasible day, skipping")
                state['unscheduled'][request.machine_type].remove(request)
                skipped.append(request)

        for req in skipped:
            bisect.insort(state['unscheduled'][req.machine_type], req, key=lambda r: r.latest)

        if not progress:
            break

    return state
