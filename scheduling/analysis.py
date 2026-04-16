from collections import defaultdict
from instance import Instance
from .cost import estimate_vehicles_and_distance


def tasks_by_day(state: dict, instance: Instance) -> dict:
    """Return per-day task lists. Each task is a dict with type, location, load, request_id."""
    tool_by_type = {t.id: t for t in instance.tools}
    result = defaultdict(list)
    for e in state['scheduled']:
        req = e['request']
        load = req.num_machines * tool_by_type[req.machine_type].size
        result[e['delivery_day']].append({
            'type': 'delivery', 'request_id': req.id,
            'location': req.location_id, 'load': load,
        })
        result[e['pickup_day']].append({
            'type': 'pickup', 'request_id': req.id,
            'location': req.location_id, 'load': load,
        })
    return result



def tool_peak_usage(state: dict, instance: Instance) -> dict:
    """Return peak concurrent usage per tool type.

    Peak = after-deliveries before-pickups, matching Validate._calculateSolution.
    """
    result = {}
    for t in instance.tools:
        diff    = state['loans'].get(t.id, [0] * (instance.config.days + 2))
        pickups = state['pickups_per_day'].get(t.id, [0] * (instance.config.days + 2))
        peak, current = 0, 0
        for day, delta in enumerate(diff):
            current += delta
            peak = max(peak, current + pickups[day])
        result[t.id] = {'peak': peak, 'available': t.num_available, 'size': t.size, 'cost': t.cost}
    return result



def print_daily_breakdown(state: dict, instance: Instance) -> None:
    tasks = tasks_by_day(state, instance)
    vehicles_per_day, _ = estimate_vehicles_and_distance(state, instance)

    print(f"{'Day':>4}  {'Vehicles':>9}  {'Tasks':>6}  {'Load':>6}  {'Deliveries':>11}  {'Pickups':>8}")
    print("-" * 60)
    for d in range(1, instance.config.days + 1):
        day_tasks = tasks.get(d, [])
        vehicles  = vehicles_per_day.get(d, 0)
        n_del     = sum(1 for t in day_tasks if t['type'] == 'delivery')
        n_pick    = sum(1 for t in day_tasks if t['type'] == 'pickup')
        total_load = sum(t['load'] for t in day_tasks)
        print(f"{d:>4}  {vehicles:>9}  {len(day_tasks):>6}  {total_load:>6}  {n_del:>11}  {n_pick:>8}")


def print_tool_usage(state: dict, instance: Instance) -> None:
    usage = tool_peak_usage(state, instance)
    print(f"{'Tool':>5}  {'Peak':>5}  {'Available':>10}  {'Utilisation':>12}  {'Size':>5}  {'Cost':>12}")
    print("-" * 55)
    for tool_id, u in usage.items():
        util = u['peak'] / u['available'] * 100 if u['available'] else 0
        print(f"{tool_id:>5}  {u['peak']:>5}  {u['available']:>10}  {util:>11.0f}%  {u['size']:>5}  {u['cost']:>12,}")


def print_load_distribution(state: dict, instance: Instance, bar_width: int = 40) -> None:
    """For each tool type, print concurrent loan count per day as a bar chart."""
    for t in instance.tools:
        diff    = state['loans'].get(t.id, [0] * (instance.config.days + 2))
        pickups = state['pickups_per_day'].get(t.id, [0] * (instance.config.days + 2))
        concurrent = []
        current = 0
        for d in range(1, instance.config.days + 1):
            current += diff[d]
            concurrent.append(current + pickups[d])

        peak = max(concurrent, default=0)
        if peak == 0:
            continue

        print(f"\n  Tool {t.id}  (available={t.num_available}, peak={peak}, cost/unit={t.cost:,})")
        print(f"  {'Day':>4}  {'In use':>6}  bar")
        print(f"  {'-'*4}  {'-'*6}  {'-'*bar_width}")
        for d, count in enumerate(concurrent, start=1):
            filled = round(count / t.num_available * bar_width)
            bar = '#' * filled + '.' * (bar_width - filled)
            peak_marker = " <-- peak" if count == peak else ""
            print(f"  {d:>4}  {count:>6}  {bar}{peak_marker}")


def print_analysis(state: dict, instance: Instance) -> None:
    print(f"=== Schedule Analysis: {instance.name} ===")
    print(f"Horizon: {instance.config.days} days  |  Capacity: {instance.config.capacity}  |  Tools: {len(instance.tools)} types\n")

    print("--- Tool usage ---")
    print_tool_usage(state, instance)

    print("\n--- Daily breakdown ---")
    print_daily_breakdown(state, instance)
