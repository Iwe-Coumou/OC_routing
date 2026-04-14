import random
from tqdm import tqdm
from .state import commit_request, uncommit_request, snapshot, restore
from .cost import compute_cost
from .greedy_minload import repair

OP_SHORT = {
    'random':           'RND',
    'peak_day':         'PKD',
    'most_overlapping': 'OVL',
    'chain':            'CHN',
}


def destroy_random(state, destroy_fraction):
    """Destroy a random subset of scheduled requests."""
    sampled = random.sample(
        state['scheduled'],
        max(1, int(len(state['scheduled']) * destroy_fraction))
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

    targets = [
        e['request'] for e in state['scheduled']
        if e['request'].machine_type == peak_tool
        and e['delivery_day'] <= peak_day < e['pickup_day']
    ]
    return targets


def destroy_most_overlapping(state, k=None):
    """Destroy requests contributing most to peak loans."""
    scores = {}
    for e in state['scheduled']:
        r = e['request']
        overlap = sum(
            1 for other in state['scheduled']
            if other['request'].id != r.id
            and other['request'].machine_type == r.machine_type
            and other['delivery_day'] < e['pickup_day']
            and e['delivery_day'] < other['pickup_day']
        )
        scores[r.id] = overlap

    k = k or max(1, len(state['scheduled']) // 5)
    targets = sorted(
        state['scheduled'],
        key=lambda e: scores[e['request'].id],
        reverse=True
    )[:k]
    return [e['request'] for e in targets]


def destroy_chain(state):
    """Pick a random chain and destroy all requests in it."""
    roots = [e for e in state['scheduled'] if e['chained_from'] is None]
    if not roots:
        return [random.choice(state['scheduled'])['request']]

    root = random.choice(roots)
    chain = [root]
    cur = root
    while True:
        next_e = next(
            (e for e in state['scheduled'] if e.get('chained_from') and
             e['chained_from']['request'].id == cur['request'].id),
            None
        )
        if not next_e:
            break
        chain.append(next_e)
        cur = next_e

    return [e['request'] for e in chain]


def repair_by_cost(state, instance, requests):
    for req in sorted(requests, key=lambda r: r.latest):
        best_day, best_cost = None, float('inf')

        for d in range(req.earliest, req.latest + 1):
            if req.pickup_day(d) > instance.config.days:
                continue
            try:
                commit_request(state, instance, req, d, chained_from=None)
            except ValueError:
                continue
            cost = compute_cost(state, instance)
            uncommit_request(state, req)
            if cost < best_cost:
                best_cost = cost
                best_day = d

        if best_day is not None:
            commit_request(state, instance, req, best_day, chained_from=None)
        # else: leave in unscheduled for greedy repair

    repair(state, instance)


def optimize_initial(state, instance, iterations=500,
                     destroy_fraction=0.2, patience=500):
    best_cost = compute_cost(state, instance)
    best_snap = snapshot(state)
    no_improve = 0
    total_improvements = 0

    operator_weights = {
        'random':           1.0,
        'peak_day':         1.0,
        'most_overlapping': 1.0,
        'chain':            1.0,
    }

    pbar = tqdm(range(iterations), desc="LNS", unit="iter")
    for _ in pbar:

        if no_improve > patience // 2:
            op_name = random.choices(
                list(operator_weights.keys()),
                weights=[0.1, 0.4, 0.4, 0.1]
            )[0]
        else:
            op_name = random.choices(
                list(operator_weights.keys()),
                weights=list(operator_weights.values())
            )[0]

        if op_name == 'random':
            seed_reqs = destroy_random(state, destroy_fraction)
        elif op_name == 'peak_day':
            seed_reqs = destroy_peak_day(state, instance)
        elif op_name == 'most_overlapping':
            seed_reqs = destroy_most_overlapping(state)
        else:
            seed_reqs = destroy_chain(state)

        seed_ids = {r.id for r in seed_reqs}

        for entry in state['scheduled']:
            if (entry['chained_from'] is not None and
                    entry['chained_from']['request'].id in seed_ids):
                state['pool'][entry['request'].machine_type][entry['delivery_day']] += entry['pool_consumed']
                entry['chained_from'] = None
                entry['pool_consumed'] = 0

        for req in seed_reqs:
            uncommit_request(state, req)

        if random.random() < 0.3 or no_improve > patience // 2:
            repair_by_cost(state, instance, seed_reqs)
        else:
            repair(state, instance)

        cost = compute_cost(state, instance)

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
            op=OP_SHORT[op_name],
        )
        if no_improve >= patience:
            break

    return best_cost
