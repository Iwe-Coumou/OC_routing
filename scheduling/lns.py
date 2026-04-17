import random
from tqdm import tqdm
from .state import uncommit_request, snapshot, restore
from .cost import compute_cost_estimate
from .greedy_edd import place_unscheduled


def destroy_random(state, fraction=0.2):
    """Destroy a random subset for exploration."""
    sampled = random.sample(
        state['scheduled'],
        max(1, int(len(state['scheduled']) * fraction))
    )
    return [e['request'] for e in sampled]


def destroy_peak_day(state, instance):
    """Destroy all requests on loan during the peak tool-cost day."""
    peak_tool, peak_day = None, None
    peak_count = 0
    for tool in instance.tools:
        for d in range(1, instance.config.days + 1):
            count = sum(
                1 for e in state['scheduled']
                if e['request'].machine_type == tool.id
                and e['delivery_day'] <= d < e['pickup_day']
            )
            if count > peak_count:
                peak_count = count
                peak_tool = tool.id
                peak_day = d

    return [
        e['request'] for e in state['scheduled']
        if e['request'].machine_type == peak_tool
        and e['delivery_day'] <= peak_day < e['pickup_day']
    ]


def destroy_most_overlapping(state, k=None):
    """Destroy the k requests contributing most to peak tool loans."""
    scores = {}
    for e in state['scheduled']:
        r = e['request']
        scores[r.id] = sum(
            1 for other in state['scheduled']
            if other['request'].id != r.id
            and other['request'].machine_type == r.machine_type
            and other['delivery_day'] < e['pickup_day']
            and e['delivery_day'] < other['pickup_day']
        )

    k = k or max(1, len(state['scheduled']) // 5)
    targets = sorted(state['scheduled'], key=lambda e: scores[e['request'].id], reverse=True)[:k]
    return [e['request'] for e in targets]


def optimize_initial(state, instance, iterations=500, patience=500):
    best_cost = compute_cost_estimate(state, instance)
    best_snap = snapshot(state)
    no_improve = 0
    total_improvements = 0

    operator_weights = {'random': 1.0, 'peak_day': 1.0, 'most_overlapping': 1.0}

    pbar = tqdm(range(iterations), desc="LNS", unit="iter")
    for _ in pbar:

        op_name = random.choices(
            list(operator_weights.keys()),
            weights=list(operator_weights.values())
        )[0]

        if op_name == 'random':
            seed_reqs = destroy_random(state)
        elif op_name == 'peak_day':
            seed_reqs = destroy_peak_day(state, instance)
        else:
            seed_reqs = destroy_most_overlapping(state)

        for req in seed_reqs:
            uncommit_request(state, req)

        place_unscheduled(state, instance)
        cost = compute_cost_estimate(state, instance)

        if cost < best_cost:
            best_cost = cost
            best_snap = snapshot(state)
            no_improve = 0
            total_improvements += 1
            operator_weights[op_name] = min(operator_weights[op_name] * 1.2, 5.0)
        else:
            restore(state, instance, best_snap)
            no_improve += 1
            operator_weights[op_name] = max(operator_weights[op_name] * 0.95, 0.1)

        pbar.set_postfix(
            best=f"{best_cost:>20,.1f}",
            impr=f"{total_improvements:>4}",
            stale=f"{no_improve:>4}",
            op=op_name[:3].upper(),
        )
        if no_improve >= patience:
            break

    return best_cost
