import math
import random
from collections import defaultdict
from tqdm import tqdm
from .state import uncommit_request, snapshot, restore, place_unscheduled
from .cost import compute_cost_estimate


def destroy_random(state, fraction=0.2):
    sampled = random.sample(
        state['scheduled'],
        max(1, int(len(state['scheduled']) * fraction))
    )
    return [e['request'] for e in sampled]


def destroy_peak_day(state, instance):
    peak_weighted, peak_tool, peak_day = 0, None, None
    for tool in instance.tools:
        diff = state['loans'].get(tool.id, [0] * (instance.config.days + 2))
        pickups = state['pickups_per_day'].get(tool.id, [0] * (instance.config.days + 2))
        running = 0
        for d in range(1, instance.config.days + 1):
            running += diff[d]
            weighted = (running + pickups[d]) * tool.cost
            if weighted > peak_weighted:
                peak_weighted = weighted
                peak_tool = tool.id
                peak_day = d

    if peak_tool is None:
        return []
    return [
        e['request'] for e in state['scheduled']
        if e['request'].machine_type == peak_tool
        and e['delivery_day'] <= peak_day < e['pickup_day']
    ]


def destroy_most_overlapping(state, instance, k=None):
    n = instance.config.days + 2
    concurrent = {}
    for tool in instance.tools:
        diff = state['loans'].get(tool.id, [0] * n)
        pickups = state['pickups_per_day'].get(tool.id, [0] * n)
        running, counts = 0, []
        for d in range(n):
            running += diff[d]
            counts.append(running + pickups[d])
        concurrent[tool.id] = counts

    scores = {}
    for e in state['scheduled']:
        r = e['request']
        counts = concurrent.get(r.machine_type, [])
        scores[r.id] = sum(
            counts[d] for d in range(e['delivery_day'], e['pickup_day'] + 1)
            if d < len(counts)
        )

    k = k or max(1, len(state['scheduled']) // 5)
    targets = sorted(state['scheduled'], key=lambda e: scores[e['request'].id], reverse=True)[:k]
    return [e['request'] for e in targets]


def repair_edd(state, instance):
    place_unscheduled(state, instance)


def repair_latest(state, instance):
    place_unscheduled(state, instance, key=lambda r: (-r.latest, r.num_machines * r.duration))


def repair_heavy(state, instance):
    place_unscheduled(state, instance, key=lambda r: (-r.num_machines * r.duration, r.latest))


def repair_random(state, instance):
    place_unscheduled(state, instance, randomize=True)


def repair_geographic(state, instance):
    locs = [e['request'].location_id for e in state['scheduled']]

    def geo_key(r):
        if not locs:
            return 0
        return min(instance.get_distance(r.location_id, l) for l in locs)

    place_unscheduled(state, instance, key=geo_key)


def destroy_heavy_day(state, instance, k=None):
    tool_by_type = {t.id: t for t in instance.tools}
    load_per_day = defaultdict(int)
    for e in state['scheduled']:
        r = e['request']
        load = r.num_machines * tool_by_type[r.machine_type].size
        load_per_day[e['delivery_day']] += load
        load_per_day[e['pickup_day']] += load
    if not load_per_day:
        return []
    peak_day = max(load_per_day, key=load_per_day.get)
    candidates = [
        e for e in state['scheduled']
        if e['delivery_day'] == peak_day or e['pickup_day'] == peak_day
    ]
    candidates.sort(
        key=lambda e: e['request'].num_machines * tool_by_type[e['request'].machine_type].size,
        reverse=True,
    )
    k = k or max(1, len(candidates) // 3)
    return [e['request'] for e in candidates[:k]]


def schedule_lns(state, instance, iterations=500, patience=150):
    best_cost = compute_cost_estimate(state, instance)
    current_cost = best_cost
    best_snap = snapshot(state)
    no_improve = 0
    total_improvements = 0
    total_sa_accepts = 0

    destroy_weights = {'random': 1.0, 'peak_day': 1.0, 'most_overlapping': 1.0, 'heavy_day': 1.0}
    repair_weights = {'edd': 1.0, 'latest': 1.0, 'heavy': 1.0, 'random': 1.0, 'geographic': 1.0}
    repair_fns = {
        'edd': repair_edd,
        'latest': repair_latest,
        'heavy': repair_heavy,
        'random': repair_random,
        'geographic': repair_geographic,
    }

    T = max(best_cost * 0.05, 1.0)
    sa_alpha = 0.995

    pbar = tqdm(range(iterations), desc="LNS", unit="iter")
    for _ in pbar:
        current_snap = snapshot(state)

        op_name = random.choices(
            list(destroy_weights.keys()),
            weights=list(destroy_weights.values())
        )[0]
        repair_name = random.choices(
            list(repair_weights.keys()),
            weights=list(repair_weights.values())
        )[0]

        if op_name == 'random':
            seed_reqs = destroy_random(state)
        elif op_name == 'peak_day':
            seed_reqs = destroy_peak_day(state, instance)
        elif op_name == 'most_overlapping':
            seed_reqs = destroy_most_overlapping(state, instance)
        else:
            seed_reqs = destroy_heavy_day(state, instance)

        for req in seed_reqs:
            uncommit_request(state, req)

        repair_fns[repair_name](state, instance)
        unscheduled = sum(len(v) for v in state['unscheduled'].values())
        if unscheduled > 0:
            restore(state, instance, current_snap)
            no_improve += 1
            destroy_weights[op_name] = max(destroy_weights[op_name] * 0.95, 0.1)
            repair_weights[repair_name] = max(repair_weights[repair_name] * 0.95, 0.1)
            pbar.set_postfix(
                best=f"{best_cost:.3e}",
                impr=total_improvements,
                sa=total_sa_accepts,
                stale=no_improve,
                op=f"{op_name[:3].upper()}+{repair_name[:3].upper()}",
            )
            continue

        cost = compute_cost_estimate(state, instance)

        if cost < best_cost:
            accepted = True
            sa_accept = False
        else:
            delta = cost - current_cost
            accepted = random.random() < math.exp(-delta / T)
            sa_accept = accepted

        T = max(T * sa_alpha, 1e-6)

        if accepted:
            current_cost = cost
            if cost < best_cost:
                best_cost = cost
                best_snap = snapshot(state)
                no_improve = 0
                total_improvements += 1
                destroy_weights[op_name] = min(destroy_weights[op_name] * 1.2, 5.0)
                repair_weights[repair_name] = min(repair_weights[repair_name] * 1.2, 5.0)
            else:
                no_improve += 1
                total_sa_accepts += 1
                destroy_weights[op_name] = min(destroy_weights[op_name] * 1.1, 5.0)
                repair_weights[repair_name] = min(repair_weights[repair_name] * 1.1, 5.0)
        else:
            restore(state, instance, current_snap)
            no_improve += 1
            destroy_weights[op_name] = max(destroy_weights[op_name] * 0.95, 0.1)
            repair_weights[repair_name] = max(repair_weights[repair_name] * 0.95, 0.1)

        pbar.set_postfix(
            best=f"{best_cost:.3e}",
            impr=total_improvements,
            sa=total_sa_accepts,
            stale=no_improve,
            op=f"{op_name[:3].upper()}+{repair_name[:3].upper()}",
        )

        if no_improve >= patience:
            break

    restore(state, instance, best_snap)
    return best_cost
