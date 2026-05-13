"""Continuous latent CP-DE search for VeRoLog schedules.

Implements CVAE-Opt (Hottung et al., ICLR 2021) adapted for VeRoLog 2017:
  - differential evolution explores a continuous latent vector
  - a CP-SAT decoder maps each vector to a feasible delivery-day assignment
  - the existing routing/cost code evaluates the decoded solution exactly

Key improvements over the original version:
  - Archive-seeded initialization: best archived solutions are converted directly
    to DE seed vectors via state_to_search_vector, giving DE an informed start in
    the neighbourhood of known-good solutions (the CVAE-Opt §4.1 inference idea)
  - Encoded posteriors: when a checkpoint is loaded, each archived solution is
    also encoded through the CVAE encoder; DE starts from N(mu, sigma) samples
    centred on the known-good region of latent space
  - The --encode-archives flag activates both mechanisms automatically
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

from cvae.archive import archive_routes, route_set_is_complete
from instance import Instance
from routing.export import cost_from_routes, write_solution
from routing.solver import solve_routing
from scheduling.state import build_state, commit_request, validate_schedule


@dataclass
class Candidate:
    vector: np.ndarray
    cost: float
    routes: dict | None
    breakdown: dict | None


def latent_dimension(instance: Instance) -> int:
    """One day-bias coordinate plus one target-day coordinate per request."""
    return instance.config.days + len(instance.requests)


def decode_latent_schedule(
    instance: Instance,
    vector: np.ndarray,
    *,
    cp_time_seconds: float = 5.0,
) -> dict | None:
    """Decode a continuous vector into a feasible schedule state.

    The hard constraints enforce one delivery day per request and per-tool stock
    availability.  The objective follows the real cost hierarchy where possible:
    lower bound on max vehicles, lower bound on vehicle-days, peak tool cost,
    then latent tie-break terms that select among equivalent schedules.
    """
    days = instance.config.days
    n_req = len(instance.requests)
    if len(vector) != latent_dimension(instance):
        raise ValueError(f"latent vector has length {len(vector)}, expected {latent_dimension(instance)}")

    day_bias = vector[:days]
    request_latent = vector[days:]
    tool_by_id = {t.id: t for t in instance.tools}

    model = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for idx, req in enumerate(instance.requests):
        choices = []
        for day in range(req.earliest, req.latest + 1):
            var = model.NewBoolVar(f"x_{req.id}_{day}")
            x[(req.id, day)] = var
            choices.append(var)
        model.AddExactlyOne(choices)

    usage_by_tool_day: dict[tuple[int, int], cp_model.IntVar] = {}
    for tool in instance.tools:
        for day in range(1, days + 1):
            terms = []
            for req in instance.requests:
                if req.machine_type != tool.id:
                    continue
                for delivery_day in range(req.earliest, req.latest + 1):
                    if delivery_day <= day <= delivery_day + req.duration:
                        terms.append(req.num_machines * x[(req.id, delivery_day)])
            usage = model.NewIntVar(0, tool.num_available, f"use_{tool.id}_{day}")
            model.Add(usage == sum(terms))
            usage_by_tool_day[(tool.id, day)] = usage

    vehicle_days = []
    max_vehicles = model.NewIntVar(0, len(instance.requests), "max_vehicles")
    for day in range(1, days + 1):
        delivery_terms = []
        pickup_terms = []
        for req in instance.requests:
            load = req.num_machines * tool_by_id[req.machine_type].size
            for delivery_day in range(req.earliest, req.latest + 1):
                if delivery_day == day:
                    delivery_terms.append(load * x[(req.id, delivery_day)])
                if delivery_day + req.duration == day:
                    pickup_terms.append(load * x[(req.id, delivery_day)])

        delivery_load = model.NewIntVar(0, 100000, f"delivery_load_{day}")
        pickup_load = model.NewIntVar(0, 100000, f"pickup_load_{day}")
        vehicles = model.NewIntVar(0, len(instance.requests), f"vehicles_{day}")
        model.Add(delivery_load == sum(delivery_terms))
        model.Add(pickup_load == sum(pickup_terms))
        model.Add(vehicles * instance.config.capacity >= delivery_load)
        model.Add(vehicles * instance.config.capacity >= pickup_load)
        model.Add(max_vehicles >= vehicles)
        vehicle_days.append(vehicles)

    tool_peak_terms = []
    for tool in instance.tools:
        peak = model.NewIntVar(0, tool.num_available, f"tool_peak_{tool.id}")
        for day in range(1, days + 1):
            model.Add(peak >= usage_by_tool_day[(tool.id, day)])
        tool_peak_terms.append(peak * tool.cost)

    latent_terms = []
    clipped_day_bias = np.clip(day_bias, -3.0, 3.0)
    for idx, req in enumerate(instance.requests):
        span = max(1, req.latest - req.earliest)
        target = req.earliest + _sigmoid(float(request_latent[idx])) * span
        for day in range(req.earliest, req.latest + 1):
            day_penalty = abs(day - target) * 50.0
            bias = clipped_day_bias[day - 1] * 25.0
            coeff = int(round(day_penalty + bias))
            if coeff:
                latent_terms.append(coeff * x[(req.id, day)])

    model.Minimize(
        max_vehicles * max(1, instance.config.vehicle_cost)
        + sum(vehicle_days) * max(1, instance.config.vehicle_day_cost)
        + sum(tool_peak_terms)
        + sum(latent_terms)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cp_time_seconds
    solver.parameters.num_search_workers = max(1, min(8, os.cpu_count() or 1))
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return None

    state = build_state(instance)
    for req in instance.requests:
        chosen = None
        for day in range(req.earliest, req.latest + 1):
            if solver.BooleanValue(x[(req.id, day)]):
                chosen = day
                break
        if chosen is None:
            return None
        commit_request(state, instance, req, chosen)

    if not validate_schedule(state["scheduled"], instance):
        return None
    return state


def evaluate_vector(
    instance: Instance,
    vector: np.ndarray,
    *,
    cp_time_seconds: float,
    route_time_seconds: int,
    quality: bool,
    routing_fixed_cost: int | None = None,
) -> Candidate:
    state = decode_latent_schedule(instance, vector, cp_time_seconds=cp_time_seconds)
    if state is None:
        return Candidate(vector=vector.copy(), cost=float("inf"), routes=None, breakdown=None)

    original_vehicle_day_cost = instance.config.vehicle_day_cost
    if routing_fixed_cost is not None:
        instance.config.vehicle_day_cost = routing_fixed_cost
    try:
        fast_routes = solve_routing(state, instance, fast=True)
        routes = solve_routing(
            state,
            instance,
            fast=False,
            time_limit_seconds=route_time_seconds,
            initial_routes=fast_routes,
        )
        if not quality:
            # Useful for very quick experiments, but not for benchmark output: the
            # repository's fast routing model is a speed approximation and can admit
            # routes that the official validator rejects on mixed pickup/delivery days.
            routes = fast_routes
    finally:
        instance.config.vehicle_day_cost = original_vehicle_day_cost

    breakdown = cost_from_routes(routes, instance)
    if not route_set_is_complete(routes, instance):
        return Candidate(vector=vector.copy(), cost=float("inf"), routes=None, breakdown=None)
    return Candidate(
        vector=vector.copy(),
        cost=breakdown["total"],
        routes=routes,
        breakdown=breakdown,
    )


def differential_evolution(
    instance: Instance,
    *,
    population_size: int = 24,
    iterations: int = 20,
    budget_seconds: float = 300.0,
    f_weight: float = 0.3,
    crossover_rate: float = 0.95,
    cp_time_seconds: float = 3.0,
    route_time_seconds: int = 3,
    seed: int = 0,
    quality: bool = True,
    seed_vectors: list[np.ndarray] | None = None,
    routing_fixed_cost: int | None = None,
    instance_path: str | Path | None = None,
    archive_dir: str | Path | None = None,
    archive_top_k: int = 50,
    archive_all_feasible: bool = False,
    archive_cost_limit: float | None = None,
) -> Candidate:
    """Run DE over the latent CP decoder."""
    if population_size < 4:
        raise ValueError("differential evolution requires population_size >= 4")
    rng = np.random.default_rng(seed)
    random.seed(seed)
    dim = latent_dimension(instance)
    start = time.monotonic()

    vectors = rng.normal(0.0, 1.0, size=(population_size, dim))
    base_vectors = []
    if seed_vectors:
        for vector in seed_vectors:
            if len(vector) != dim:
                raise ValueError(f"seed vector has length {len(vector)}, expected {dim}")
            base_vectors.append(np.asarray(vector, dtype=np.float64))
    base_vectors.append(np.zeros(dim, dtype=np.float64))
    early = np.zeros(dim, dtype=np.float64)
    early[: instance.config.days] = np.linspace(-1.0, 1.0, instance.config.days)
    base_vectors.append(early)
    late = np.zeros(dim, dtype=np.float64)
    late[: instance.config.days] = np.linspace(1.0, -1.0, instance.config.days)
    base_vectors.append(late)
    for idx, vector in enumerate(base_vectors[:population_size]):
        vectors[idx, :] = vector
    if seed_vectors:
        start_idx = min(len(base_vectors), population_size)
        for idx in range(start_idx, population_size):
            vectors[idx, :] = np.clip(seed_vectors[0] + rng.normal(0.0, 0.35, size=dim), -5.0, 5.0)

    population = [
        evaluate_vector(
            instance,
            vector,
            cp_time_seconds=cp_time_seconds,
            route_time_seconds=route_time_seconds,
            quality=quality,
            routing_fixed_cost=routing_fixed_cost,
        )
        for vector in vectors
    ]
    if archive_all_feasible:
        for idx, candidate in enumerate(population):
            _archive_candidate(
                candidate,
                instance=instance,
                instance_path=instance_path,
                archive_dir=archive_dir,
                source=f"initial{idx}_seed{seed}",
                keep_top=archive_top_k,
                cost_limit=archive_cost_limit,
            )
    best = min(population, key=lambda c: c.cost)
    print(f"initial best: {best.cost:,.0f}", flush=True)
    _archive_candidate(
        best,
        instance=instance,
        instance_path=instance_path,
        archive_dir=archive_dir,
        source=f"initial_seed{seed}",
        keep_top=archive_top_k,
        cost_limit=archive_cost_limit,
    )

    for iteration in range(iterations):
        if time.monotonic() - start >= budget_seconds:
            break
        for i in range(population_size):
            if time.monotonic() - start >= budget_seconds:
                break
            choices = [j for j in range(population_size) if j != i]
            a_idx, b_idx, c_idx = rng.choice(choices, size=3, replace=False)
            mutant = (
                population[a_idx].vector
                + f_weight * (population[b_idx].vector - population[c_idx].vector)
            )
            mask = rng.random(dim) < crossover_rate
            mask[rng.integers(0, dim)] = True
            trial = np.where(mask, mutant, population[i].vector)
            trial = np.clip(trial, -5.0, 5.0)
            candidate = evaluate_vector(
                instance,
                trial,
                cp_time_seconds=cp_time_seconds,
                route_time_seconds=route_time_seconds,
                quality=quality,
                routing_fixed_cost=routing_fixed_cost,
            )
            if archive_all_feasible:
                _archive_candidate(
                    candidate,
                    instance=instance,
                    instance_path=instance_path,
                    archive_dir=archive_dir,
                    source=f"trial_i{i}_iter{iteration + 1}_seed{seed}",
                    keep_top=archive_top_k,
                    cost_limit=archive_cost_limit,
                )
            if candidate.cost < population[i].cost:
                population[i] = candidate
                if candidate.cost < best.cost:
                    best = candidate
                    print(
                        f"iter {iteration + 1}: best={best.cost:,.0f} "
                        f"{_format_breakdown(best.breakdown)}",
                        flush=True,
                    )
                    _archive_candidate(
                        best,
                        instance=instance,
                        instance_path=instance_path,
                        archive_dir=archive_dir,
                        source=f"iter{iteration + 1}_seed{seed}",
                        keep_top=archive_top_k,
                        cost_limit=archive_cost_limit,
                    )

    return best


def build_archive_seed_vectors(
    instance: Instance,
    instance_path: str | Path,
    archive_dir: str | Path,
    *,
    max_seeds: int = 16,
    model=None,
    device=None,
    n_encoded_samples: int = 4,
    seed: int = 0,
) -> list[np.ndarray]:
    """Build DE seed vectors from the best archived solutions.

    Two seed types are generated:
    1. *Direct seeds*: convert each archived solution state to a search vector via
       ``state_to_search_vector``.  This places DE directly in the neighbourhood
       of every known-good solution without needing a model at all.
    2. *Encoded seeds* (when a CVAE model is provided): encode each archived
       solution through the CVAE posterior, then sample N(mu, sigma) to get
       diverse candidates near the known-good latent region — the CVAE-Opt §4.1
       inference procedure.
    """
    from cvae.archive import best_archived
    from cvae.model import state_to_search_vector
    from routing.export import read_solution

    records = best_archived(instance_path=instance_path, archive_dir=archive_dir)
    seeds: list[np.ndarray] = []
    max_days = instance.config.days

    for _cost, path in records[:max_seeds]:
        try:
            state, _ = read_solution(str(path), instance)
        except Exception:
            continue
        # Direct seed: the solution converted to a DE search vector
        try:
            vec = state_to_search_vector(instance, state, max_days)
            seeds.append(vec)
        except Exception:
            pass
        # Encoded seeds via CVAE posterior (CVAE-Opt §4.1)
        if model is not None and device is not None:
            try:
                from cvae.model import predict_latent_vectors_from_encoded
                encoded = predict_latent_vectors_from_encoded(
                    model, instance, state,
                    n_samples=n_encoded_samples,
                    device=device,
                    seed=seed + len(seeds),
                )
                seeds.extend(encoded)
            except Exception:
                pass
        if len(seeds) >= max_seeds * (1 + n_encoded_samples):
            break

    print(f"archive seeds: {len(seeds)} vectors from {len(records)} archived solutions", flush=True)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent CP-DE search for VeRoLog instances")
    parser.add_argument("--instance", required=True, help="Path to a VeRoLog .txt instance")
    parser.add_argument("--budget", type=float, default=300.0, help="Wall-clock budget in seconds")
    parser.add_argument("--population", type=int, default=24, help="DE population size")
    parser.add_argument("--iterations", type=int, default=20, help="DE iteration cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cp-time", type=float, default=3.0, help="CP-SAT decode time per candidate")
    parser.add_argument("--route-time", type=int, default=3, help="Quality routing time per day")
    parser.add_argument(
        "--routing-fixed-cost",
        type=int,
        help=(
            "Temporary fixed cost used inside OR-Tools route construction. "
            "The final solution is still scored with the instance's official cost."
        ),
    )
    parser.add_argument("--checkpoint", help="Schedule-prior checkpoint from cvae.train")
    parser.add_argument("--checkpoint-samples", type=int, default=4, help="Number of seed vectors sampled from checkpoint")
    parser.add_argument(
        "--encode-archives",
        action="store_true",
        help=(
            "Seed DE from best archived solutions: convert each to a direct search vector "
            "(no model needed) and, when a checkpoint is provided, also encode through the "
            "CVAE posterior to sample N(mu, sigma) (CVAE-Opt §4.1 inference). "
            "Recommended: always use this flag when re-running on B3."
        ),
    )
    parser.add_argument(
        "--encode-archives-max",
        type=int,
        default=16,
        help="Maximum number of archived solutions to use for seeding (default 16).",
    )
    parser.add_argument(
        "--fast-routing",
        action="store_true",
        help="Use fast approximate routing during DE. Faster, but output may not validate.",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", help="Output solution path")
    parser.add_argument(
        "--archive-dir",
        help="Persist every incumbent improvement under this archive directory.",
    )
    parser.add_argument("--archive-top-k", type=int, default=50)
    parser.add_argument(
        "--archive-all-feasible",
        action="store_true",
        help="Archive every feasible evaluated candidate, not only incumbent improvements.",
    )
    parser.add_argument(
        "--archive-cost-limit",
        type=float,
        help="When archiving all feasible candidates, only persist candidates at or below this cost.",
    )
    args = parser.parse_args()

    instance = Instance(args.instance)
    seed_vectors: list[np.ndarray] = []

    model = None
    device = None
    if args.checkpoint:
        import torch
        from cvae.model import load_checkpoint, predict_latent_vectors

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_checkpoint(args.checkpoint, device=device)
        prior_seeds = predict_latent_vectors(
            model,
            instance,
            n_samples=args.checkpoint_samples,
            device=device,
            seed=args.seed,
        )
        seed_vectors.extend(prior_seeds)
        print(f"loaded checkpoint {args.checkpoint}: {len(prior_seeds)} prior seed(s)", flush=True)

    if args.encode_archives and args.archive_dir:
        archive_seeds = build_archive_seed_vectors(
            instance,
            args.instance,
            args.archive_dir,
            max_seeds=args.encode_archives_max,
            model=model,
            device=device,
            n_encoded_samples=2,
            seed=args.seed,
        )
        seed_vectors = archive_seeds + seed_vectors  # archive seeds take priority

    best = differential_evolution(
        instance,
        population_size=args.population,
        iterations=args.iterations,
        budget_seconds=args.budget,
        cp_time_seconds=args.cp_time,
        route_time_seconds=args.route_time,
        seed=args.seed,
        quality=args.quality or not args.fast_routing,
        seed_vectors=seed_vectors if seed_vectors else None,
        routing_fixed_cost=args.routing_fixed_cost,
        instance_path=args.instance,
        archive_dir=args.archive_dir,
        archive_top_k=args.archive_top_k,
        archive_all_feasible=args.archive_all_feasible,
        archive_cost_limit=args.archive_cost_limit,
    )
    if best.routes is None or best.breakdown is None or not math.isfinite(best.cost):
        raise SystemExit("no feasible decoded solution found")

    output = args.output
    if output is None:
        name = os.path.splitext(os.path.basename(args.instance))[0]
        output = os.path.join("solutions", "cvae_de", f"{name}_solution.txt")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    write_solution(best.routes, instance, output)
    print(f"wrote {output}")
    print(f"best: {best.cost:,.0f} {_format_breakdown(best.breakdown)}")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _format_breakdown(breakdown: dict | None) -> str:
    if not breakdown:
        return ""
    return (
        f"vehicles={breakdown['max_vehicles']} "
        f"vehicle_days={breakdown['vehicle_days_count']} "
        f"tool={breakdown['tool']:,.0f} "
        f"distance={breakdown['distance']:,.0f}"
    )


def _archive_candidate(
    candidate: Candidate,
    *,
    instance: Instance,
    instance_path: str | Path | None,
    archive_dir: str | Path | None,
    source: str,
    keep_top: int,
    cost_limit: float | None,
) -> None:
    if archive_dir is None or instance_path is None or candidate.routes is None:
        return
    if cost_limit is not None and candidate.cost > cost_limit:
        return
    try:
        output = archive_routes(
            instance_path=instance_path,
            instance=instance,
            route_set=candidate.routes,
            archive_dir=archive_dir,
            source=source,
            keep_top=keep_top,
        )
        print(f"archived incumbent {output}", flush=True)
    except Exception as exc:
        print(f"archive failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
