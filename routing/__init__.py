from instance import Instance
from .tasks import build_daily_stops
from .model import solve_all_days
from .export import routing_stats, write_solution, cost_from_routes


def solve_routing(
    state: dict,
    instance: Instance,
    time_limit_seconds: int = 30,
    fast: bool = False,
) -> dict:
    """Solve CVRP routing for all days in the schedule.

    Takes the optimised schedule state and finds vehicle routes for each day
    using Google OR-Tools. Capacity is modelled as load on vehicle: deliveries
    reduce load (tools dropped off), pickups increase load (tools collected).
    OR-Tools sets the start load per vehicle to the minimum pre-load needed
    for the chosen stop ordering, enforced by keeping cumul in [0, capacity].

    Returns an in-memory RouteSet — no file I/O, safe to call in a loop.

    Args:
        state: Schedule state dict from scheduling.build_schedule() /
               scheduling.optimize_initial().
        instance: Problem instance.
        time_limit_seconds: OR-Tools solver budget per day (ignored when fast=True).
        fast: If True, use PATH_CHEAPEST_ARC only — completes in seconds across
              all days, suitable as a cost signal inside an optimiser loop.
              If False (default), run GLS for final solution quality.

    Returns:
        RouteSet: dict[int, list[VehicleRoute]]
    """
    daily_stops = build_daily_stops(state, instance)
    return solve_all_days(daily_stops, instance, time_limit_seconds, fast=fast)
