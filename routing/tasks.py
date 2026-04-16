from instance import Instance
from scheduling.analysis import tasks_by_day
from .routes import Stop


def build_daily_stops(state: dict, instance: Instance) -> dict:
    """Return per-day stop lists derived from the schedule state.

    Returns:
        dict[int, list[Stop]]  — days with no stops excluded.
    """
    req_lookup = {r.id: r for r in instance.requests}
    raw = tasks_by_day(state, instance)
    result = {}
    for day, tasks in raw.items():
        if not tasks:
            continue
        stops = [
            Stop(
                request_id=t['request_id'],
                action=t['type'],
                location_id=t['location'],
                load=t['load'],
                machine_type=req_lookup[t['request_id']].machine_type,
            )
            for t in tasks
        ]
        result[day] = stops
    return result
