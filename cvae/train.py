"""Train the schedule prior used by latent CP-DE search."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from instance import Instance
from routing.export import cost_from_routes, read_solution
from scheduling.feasibility import repair_feasibility
from scheduling.state import build_schedule, validate_schedule

from cvae.archive import best_archived, route_set_is_complete
from cvae.model import (
    FEATURE_DIM,
    AttentiveScheduleCVAE,
    InstanceTensor,
    ScheduleCVAE,
    SchedulePriorNet,
    collate_instance_tensors,
    make_instance_tensor,
    target_from_state,
)
from cvae.search import decode_latent_schedule, latent_dimension


def instance_paths(root: str | Path) -> list[Path]:
    """Return instance paths from a directory or a single file."""
    root = Path(root)
    if root.is_file():
        return [root]
    paths = []
    for path in sorted(root.glob("*.txt")):
        if path.name.endswith("_solution.txt"):
            continue
        paths.append(path)
    return paths


def jitter_instance_tensor(
    instance: Instance,
    tensor: InstanceTensor,
    *,
    jitter_prob: float = 0.4,
    rng: np.random.Generator,
    max_days: int,
) -> InstanceTensor:
    """Create a symmetry-augmented copy by randomly shifting request delivery days.

    This is the VeRoLog adaptation of the CVAE-Opt symmetry-breaking augmentation
    (Hottung et al. ICLR 2021, §3.3): for requests with slack in their time window,
    shift the delivery day by ±1 with probability *jitter_prob*.  The jittered
    assignment may not satisfy tool-availability constraints, but training on diverse
    near-feasible targets teaches the encoder that nearby day assignments map to
    nearby latent vectors — exactly the smoothness property DE needs.
    """
    if tensor.target_z is None:
        return tensor
    target_z_np = tensor.target_z.numpy().copy()
    for idx, req in enumerate(instance.requests):
        span = req.latest - req.earliest
        if span < 1:
            continue
        if rng.random() >= jitter_prob:
            continue
        # Reverse the logit to recover the current fractional day
        logit_val = float(target_z_np[idx])
        frac = 1.0 / (1.0 + math.exp(-logit_val))
        current_day = round(req.earliest + frac * span)
        current_day = max(req.earliest, min(req.latest, current_day))
        # Random ±1 shift
        delta = int(rng.choice([-1, 1]))
        new_day = max(req.earliest, min(req.latest, current_day + delta))
        new_frac = float(np.clip((new_day - req.earliest) / span, 1e-3, 1.0 - 1e-3))
        target_z_np[idx] = float(np.clip(math.log(new_frac / (1.0 - new_frac)), -5.0, 5.0))
    return InstanceTensor(
        features=tensor.features,
        request_mask=tensor.request_mask,
        target_z=torch.from_numpy(target_z_np),
        target_day_bias=tensor.target_day_bias,
    )


def available_states(
    instance: Instance,
    instance_path: Path,
    cp_time_seconds: float,
    samples_per_instance: int,
    seed: int,
    generated_samples_per_instance: int | None = None,
) -> list[tuple[dict, str]]:
    """Return multiple supervised target states for one instance."""
    name = instance_path.stem
    solution_candidates = _solution_candidate_paths(name, instance_path)
    solution_states: list[tuple[dict, str, tuple]] = []
    generated_states: list[tuple[dict, str, tuple]] = []
    seen = set()
    for solution in solution_candidates:
        try:
            state, route_set = read_solution(str(solution), instance)
        except Exception:
            continue
        if route_set_is_complete(route_set, instance) and validate_schedule(state["scheduled"], instance):
            signature = _state_signature(state)
            if signature not in seen:
                seen.add(signature)
                rank = (0, _route_cost(route_set, instance), *schedule_score(state, instance))
                solution_states.append((state, str(solution), rank))

    generated_samples = samples_per_instance if generated_samples_per_instance is None else generated_samples_per_instance
    vectors = _seed_vectors(instance, generated_samples, seed)
    for vector in vectors:
        state = decode_latent_schedule(instance, vector, cp_time_seconds=cp_time_seconds)
        if state is None:
            continue
        signature = _state_signature(state)
        if signature not in seen:
            seen.add(signature)
            rank = (1, float("inf"), *schedule_score(state, instance))
            generated_states.append((state, "cp_decoder", rank))

    states = sorted(solution_states, key=lambda item: item[2])
    remaining = max(0, samples_per_instance - len(states))
    states.extend(sorted(generated_states, key=lambda item: item[2])[:remaining])

    if not states:
        state = build_schedule(instance)
        if sum(len(v) for v in state["unscheduled"].values()) > 0:
            repair_feasibility(state, instance)
        if not validate_schedule(state["scheduled"], instance):
            raise ValueError(f"could not build a valid target for {instance_path}")
        states.append((state, "build_schedule", (2, float("inf"), *schedule_score(state, instance))))

    return [(state, source) for state, source, _ in states[:samples_per_instance]]


def _solution_candidate_paths(name: str, instance_path: Path) -> list[Path]:
    """Return ordinary solution files plus persistent archive files."""
    seen = set()
    paths: list[Path] = []
    for solution in sorted(Path("solutions").glob(f"**/{name}*_solution.txt")):
        key = str(solution.resolve())
        if key not in seen:
            seen.add(key)
            paths.append(solution)

    for _, solution in best_archived(instance_path=instance_path, archive_dir="solutions/cvae_archive"):
        key = str(solution.resolve())
        if key not in seen:
            seen.add(key)
            paths.append(solution)
    return paths


def schedule_score(state: dict, instance: Instance) -> tuple[int, int, int]:
    tool_by_id = {t.id: t for t in instance.tools}
    delivery_load = {day: 0 for day in range(1, instance.config.days + 1)}
    pickup_load = {day: 0 for day in range(1, instance.config.days + 1)}
    for entry in state["scheduled"]:
        req = entry["request"]
        load = req.num_machines * tool_by_id[req.machine_type].size
        delivery_load[entry["delivery_day"]] += load
        pickup_load[entry["pickup_day"]] += load
    vehicles = [
        max(
            int(np.ceil(delivery_load[day] / instance.config.capacity)),
            int(np.ceil(pickup_load[day] / instance.config.capacity)),
        )
        for day in range(1, instance.config.days + 1)
    ]
    max_vehicles = max(vehicles, default=0)
    vehicle_days = sum(v for v in vehicles if v > 0)

    tool_cost = 0
    for tool in instance.tools:
        diff = state["loans"][tool.id]
        pickups = state["pickups_per_day"][tool.id]
        running = 0
        peak = 0
        for day in range(1, instance.config.days + 1):
            running += diff[day]
            peak = max(peak, running + pickups[day])
        tool_cost += peak * tool.cost
    return max_vehicles, vehicle_days, tool_cost


def _route_cost(route_set: dict, instance: Instance) -> float:
    try:
        return float(cost_from_routes(route_set, instance)["total"])
    except Exception:
        return float("inf")


def build_dataset(
    paths: list[Path],
    cp_time_seconds: float,
    max_days: int,
    samples_per_instance: int,
    seed: int,
    generated_samples_per_instance: int | None,
    jitter_prob: float = 0.0,
    jitter_copies: int = 0,
) -> tuple[list, list[dict]]:
    examples = []
    manifest = []
    jitter_rng = np.random.default_rng(seed + 77777) if jitter_prob > 0 and jitter_copies > 0 else None
    for path in paths:
        instance = Instance(str(path))
        states = available_states(
            instance,
            path,
            cp_time_seconds,
            samples_per_instance,
            seed,
            generated_samples_per_instance,
        )
        for idx, (state, source) in enumerate(states):
            base_tensor = make_instance_tensor(instance, state=state, max_days=max_days)
            examples.append(base_tensor)
            score = schedule_score(state, instance)
            manifest.append(
                {
                    "instance": str(path),
                    "source": source,
                    "sample": idx,
                    "requests": len(instance.requests),
                    "days": instance.config.days,
                    "score": list(score),
                }
            )
            print(
                f"target {path.name}#{idx}: source={source} requests={len(instance.requests)} "
                f"score=maxveh {score[0]} vehdays {score[1]} tool {score[2]}",
                flush=True,
            )
            # Symmetry-breaking augmentation: jittered copies of each training example
            if jitter_rng is not None:
                for _ in range(jitter_copies):
                    jittered = jitter_instance_tensor(
                        instance,
                        base_tensor,
                        jitter_prob=jitter_prob,
                        rng=jitter_rng,
                        max_days=max_days,
                    )
                    examples.append(jittered)
                    manifest.append(
                        {**manifest[-1], "source": source + "_jitter", "sample": idx}
                    )
    return examples, manifest


def train_model(
    examples: list,
    *,
    max_days: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    output: Path,
    manifest: list[dict],
    model_type: str,
    beta: float,
    resume: Path | None = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"training device: {device}", flush=True)
    model, checkpoint_meta = _build_model(
        model_type=model_type,
        max_days=max_days,
        device=device,
        resume=resume,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    z_loss_fn = nn.SmoothL1Loss(reduction="none")
    day_loss_fn = nn.MSELoss()

    order = list(range(len(examples)))
    for epoch in range(1, epochs + 1):
        random.shuffle(order)
        total_loss = 0.0
        total_z = 0.0
        total_day = 0.0
        batches = 0
        model.train()
        for start in range(0, len(order), batch_size):
            items = [examples[i] for i in order[start : start + batch_size]]
            batch = collate_instance_tensors(items, max_days=max_days)
            features = batch.features.to(device)
            mask = batch.request_mask.to(device)
            target_z = batch.target_z.to(device)
            target_day = batch.target_day_bias.to(device)

            if isinstance(model, (ScheduleCVAE, AttentiveScheduleCVAE)):
                pred_z, pred_day, mu, logvar = model(features, mask, target_z)
                kl_loss = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            else:
                pred_z, pred_day = model(features, mask)
                kl_loss = torch.zeros((), device=device)
            z_loss = z_loss_fn(pred_z, target_z)
            z_loss = (z_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            day_loss = day_loss_fn(pred_day, target_day)
            loss = z_loss + 0.1 * day_loss + beta * kl_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite training loss: loss={loss.item()} "
                    f"z={z_loss.item()} day={day_loss.item()} kl={kl_loss.item()}"
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.detach().cpu())
            total_z += float(z_loss.detach().cpu())
            total_day += float(day_loss.detach().cpu())
            batches += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:03d}: loss={total_loss / batches:.5f} "
                f"z={total_z / batches:.5f} day={total_day / batches:.5f}",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, AttentiveScheduleCVAE):
        mtype = "attentive_cvae"
    elif isinstance(model, ScheduleCVAE):
        mtype = "schedule_cvae"
    else:
        mtype = "schedule_prior"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": mtype,
            "feature_dim": FEATURE_DIM,
            "hidden_dim": checkpoint_meta["hidden_dim"],
            "latent_dim": getattr(model, "latent_dim", 0),
            "num_heads": getattr(model, "num_heads", 4),
            "num_layers": getattr(model, "num_layers", 3),
            "max_days": max_days,
            "resume": str(resume) if resume else None,
            "manifest": manifest,
        },
        output,
    )
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote checkpoint {output}", flush=True)
    print(f"wrote manifest {manifest_path}", flush=True)


def _build_model(
    *,
    model_type: str,
    max_days: int,
    device: torch.device,
    resume: Path | None,
) -> tuple[nn.Module, dict]:
    hidden_dim = 256 if model_type == "attcvae" else 128
    latent_dim = 64 if model_type == "attcvae" else 32
    num_heads = 4
    num_layers = 3

    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=False)
        checkpoint_type = payload.get("model_type")
        # Allow resuming attentive_cvae checkpoint with --model attcvae
        valid = {
            "cvae": "schedule_cvae",
            "attcvae": "attentive_cvae",
            "prior": "schedule_prior",
        }
        expected_type = valid.get(model_type)
        if checkpoint_type != expected_type:
            raise ValueError(
                f"resume checkpoint is {checkpoint_type!r}, but --model {model_type!r} expects {expected_type!r}"
            )
        checkpoint_days = int(payload["max_days"])
        if checkpoint_days != max_days:
            raise ValueError(
                f"resume checkpoint max_days={checkpoint_days}, current training max_days={max_days}"
            )
        hidden_dim = int(payload.get("hidden_dim", hidden_dim))
        latent_dim = int(payload.get("latent_dim", latent_dim))
        num_heads = int(payload.get("num_heads", num_heads))
        num_layers = int(payload.get("num_layers", num_layers))

    if model_type == "attcvae":
        model = AttentiveScheduleCVAE(
            feature_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            max_days=max_days,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    elif model_type == "cvae":
        model = ScheduleCVAE(
            feature_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            max_days=max_days,
            latent_dim=latent_dim,
        )
    else:
        model = SchedulePriorNet(
            feature_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            max_days=max_days,
        )

    if resume is not None:
        model.load_state_dict(payload["state_dict"])
        print(f"resumed checkpoint {resume}", flush=True)

    return model.to(device), {"hidden_dim": hidden_dim, "latent_dim": latent_dim}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the schedule prior for latent CP-DE")
    parser.add_argument("--instances", default="instances")
    parser.add_argument("--output", default="cvae/checkpoints/schedule_prior.pt")
    parser.add_argument("--resume", help="Existing checkpoint to continue training from")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cp-time", type=float, default=2.0)
    parser.add_argument("--samples-per-instance", type=int, default=8)
    parser.add_argument(
        "--generated-samples-per-instance",
        type=int,
        help="Number of CP-decoder targets to generate per instance before filling from archives/solutions.",
    )
    parser.add_argument("--model", choices=["cvae", "prior", "attcvae"], default="attcvae",
                        help="Model type: attcvae = transformer CVAE-Opt (recommended), cvae = original MLP CVAE, prior = prior net")
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--jitter-prob", type=float, default=0.4,
                        help="Per-request day-jitter probability for symmetry-breaking augmentation (0=disabled)")
    parser.add_argument("--jitter-copies", type=int, default=3,
                        help="Number of jittered copies per training example (symmetry-breaking augmentation)")
    args = parser.parse_args()

    paths = instance_paths(args.instances)
    if not paths:
        raise SystemExit(f"no instance files found under {args.instances}")
    max_days = max(Instance(str(path)).config.days for path in paths)
    print(f"training on {len(paths)} instances, max_days={max_days}", flush=True)
    examples, manifest = build_dataset(
        paths,
        args.cp_time,
        max_days,
        args.samples_per_instance,
        args.seed,
        args.generated_samples_per_instance,
        jitter_prob=args.jitter_prob,
        jitter_copies=args.jitter_copies,
    )
    print(f"dataset size: {len(examples)} examples (incl. augmentation)", flush=True)
    train_model(
        examples,
        max_days=max_days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output=Path(args.output),
        manifest=manifest,
        model_type=args.model,
        beta=args.beta,
        resume=Path(args.resume) if args.resume else None,
    )


def _seed_vectors(instance: Instance, samples: int, seed: int) -> list[np.ndarray]:
    dim = latent_dimension(instance)
    vectors = []
    zero = np.zeros(dim, dtype=np.float64)
    vectors.append(zero)
    early = np.zeros(dim, dtype=np.float64)
    early[: instance.config.days] = np.linspace(-1.0, 1.0, instance.config.days)
    vectors.append(early)
    late = np.zeros(dim, dtype=np.float64)
    late[: instance.config.days] = np.linspace(1.0, -1.0, instance.config.days)
    vectors.append(late)
    rng = np.random.default_rng(seed + 1234 + len(instance.requests))
    while len(vectors) < samples:
        vectors.append(rng.normal(0.0, 1.0, size=dim))
    return vectors


def _state_signature(state: dict) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((e["request"].id, e["delivery_day"]) for e in state["scheduled"]))


if __name__ == "__main__":
    main()
