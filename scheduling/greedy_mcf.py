import logging
from instance import Instance, Request
from .state import build_state, commit_request, is_feasible

log = logging.getLogger(__name__)



def _best_delivery_day(state: dict, instance: Instance, request: Request) -> int | None:
    """Return the feasible delivery day that minimises the worst-case peak load
    across the entire loan window [delivery_day, pickup_day].

    Scores each candidate day by max(running + pickups[day] + request.num_machines)
    over [delivery_day, pickup_day], which is exactly what is_feasible enforces.
    The greedy therefore avoids creating pickup-day congestion from the outset.
    """
    best_day = None
    best_peak = float('inf')

    diff = state['loans'][request.machine_type]
    pickups = state['pickups_per_day'][request.machine_type]

    for d in range(request.earliest, request.latest + 1):
        pickup_day = d + request.duration
        if not is_feasible(state, instance, request, d, pickup_day):
            continue

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


def place_unscheduled(state: dict, instance: Instance) -> None:
    """Place all unscheduled requests into the schedule.

    Collects all unscheduled requests, sorts them in MCF order (tightest window
    first, largest demand on ties), then places each one on the delivery day that
    minimises the worst-case peak across its loan window.

    Called by build_schedule for initial construction and by LNS after each
    destroy step to re-insert the requests that were removed.
    """
    requests = [r for reqs in state['unscheduled'].values() for r in reqs]
    requests.sort(key=lambda r: (r.latest - r.earliest, -(r.num_machines * r.duration)))

    for request in requests:
        day = _best_delivery_day(state, instance, request)
        if day is None:
            log.warning(f"MCF req={request.id} has no feasible day")
            continue
        commit_request(state, instance, request, day)


def build_schedule(instance: Instance) -> dict:
    """Greedy scheduler: Most-Constrained-First ordering + worst-peak minload placement.

    Requests with the tightest time windows (and, on ties, the largest resource
    demand) are placed first, when capacity is most available.  This makes a
    complete initial schedule far more likely than EDF ordering, because the
    hardest-to-fit requests get first pick of capacity rather than finding it
    already consumed by flexible requests.

    Placement scoring minimises the worst peak across the full loan window
    [delivery_day, pickup_day], keeping the schedule aligned with is_feasible
    and reducing congestion that LNS would otherwise have to fix.
    """
    state = build_state(instance)
    place_unscheduled(state, instance)
    return state
