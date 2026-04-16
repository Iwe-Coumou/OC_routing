import logging
from collections import defaultdict
from instance import Instance

log = logging.getLogger(__name__)


def validate_schedule(scheduled: list, instance: Instance) -> bool:
    valid = True

    scheduled_ids = [e['request'].id for e in scheduled]
    expected_ids  = [r.id for r in instance.requests]
    if sorted(scheduled_ids) != sorted(expected_ids):
        print(f"FAIL: scheduled requests {sorted(scheduled_ids)} != expected {sorted(expected_ids)}")
        valid = False

    for entry in scheduled:
        r = entry['request']
        d = entry['delivery_day']
        p = entry['pickup_day']

        if not (r.earliest <= d <= r.latest):
            print(f"FAIL: req={r.id} delivery_day={d} outside window [{r.earliest}, {r.latest}]")
            valid = False

        if p != d + r.duration:
            print(f"FAIL: req={r.id} pickup_day={p} != delivery_day={d} + duration={r.duration}")
            valid = False

        if p > instance.config.days:
            print(f"FAIL: req={r.id} pickup_day={p} exceeds horizon {instance.config.days}")
            valid = False

    tool_by_type = {t.id: t for t in instance.tools}
    loans = defaultdict(lambda: [0] * (instance.config.days + 2))
    pickups = defaultdict(lambda: [0] * (instance.config.days + 2))
    for entry in scheduled:
        r = entry['request']
        loans[r.machine_type][entry['delivery_day']] += r.num_machines
        loans[r.machine_type][entry['pickup_day']]   -= r.num_machines
        pickups[r.machine_type][entry['pickup_day']] += r.num_machines

    for machine_type, diff in loans.items():
        current = 0
        limit = tool_by_type[machine_type].num_available
        for day, delta in enumerate(diff):
            current += delta
            # Check within-day peak: after deliveries, before pickups.
            # current is the end-of-day net; adding pickups[day] back gives
            # the before-pickup count, matching Validate._calculateSolution.
            peak = current + pickups[machine_type][day]
            if peak > limit:
                print(f"FAIL: type={machine_type} day={day} peak use={peak} exceeds available={limit}")
                valid = False

    if valid:
        log.debug("OK: schedule is valid")
    return valid
