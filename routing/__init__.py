from instance import Instance
from .tasks import build_daily_stops
from .model import solve_all_days
from .export import routing_stats, write_solution, cost_from_routes, read_solution


def solve_routing(
    state: dict,
    instance: Instance,
    time_limit_seconds: int = 30,
    fast: bool = False,
    initial_routes: dict = None,
) -> dict:
    """Solve CVRP routing for all days in the schedule.

    Returns an in-memory RouteSet — no file I/O, safe to call in a loop.

    Args:
        state: Schedule state dict.
        instance: Problem instance.
        time_limit_seconds: OR-Tools solver budget per day (ignored when fast=True).
        fast: If True, use PATH_CHEAPEST_ARC only (parallel threads, seconds per run).
              If False, run GLS for final solution quality.
        initial_routes: Optional RouteSet to warm-start GLS. Ignored when fast=True.

    Returns:
        RouteSet: dict[int, list[VehicleRoute]]
    """
    daily_stops = build_daily_stops(state, instance)
    return solve_all_days(daily_stops, instance, time_limit_seconds,
                          fast=fast, initial_routes=initial_routes)


def solve_routing_incremental(
    state: dict,
    instance: Instance,
    changed_days: set,
    current_routes: dict,
) -> dict:
    """Re-route only the days whose stop sets changed; carry over the rest.

    Args:
        state: Current schedule state dict.
        instance: Problem instance.
        changed_days: Set of day integers that had stops added or removed.
        current_routes: Full RouteSet from the previous iteration.

    Returns:
        RouteSet: dict[int, list[VehicleRoute]] — full set for all days.
    """
    daily_stops = build_daily_stops(state, instance)

    # Carry over unchanged days without re-solving.
    route_set = {day: routes for day, routes in current_routes.items()
                 if day not in changed_days}

    # Drop days that are no longer in daily_stops (no stops left after destroy).
    route_set = {day: routes for day, routes in route_set.items()
                 if day in daily_stops}

    # Build the subset to re-solve.
    dirty_stops = {day: daily_stops[day] for day in changed_days if day in daily_stops}
    if dirty_stops:
        new_routes = solve_all_days(dirty_stops, instance, fast=True)
        route_set.update(new_routes)

    return route_set
