from instance import Instance
from .state import build_state
from .repair import next_unscheduled, fallback, try_backwards_extend, try_forward_chain


def build_schedule(instance: Instance) -> dict:
    state = build_state(instance)

    while True:
        changed = try_backwards_extend(state, instance)
        if changed:
            continue

        request = next_unscheduled(state)
        if request is None:
            break

        chained = try_forward_chain(state, instance, request)
        if not chained:
            fallback(state, instance, request)

    return state
