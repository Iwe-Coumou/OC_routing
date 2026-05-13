"""Data helpers for latent-search experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from instance import Instance
from routing.export import cost_from_routes, read_solution
from scheduling.state import validate_schedule

from cvae.representation import Solution, Token


@dataclass(frozen=True)
class SolutionRecord:
    instance_path: Path
    solution_path: Path
    cost: float
    tokens: list[Token]


def discover_solution_files(root: str | Path = "solutions") -> list[Path]:
    """Return all text solution files under ``root``."""
    return sorted(Path(root).glob("**/*_solution.txt"))


def infer_instance_path(solution_path: str | Path, instances_dir: str | Path = "instances") -> Path:
    """Infer an instance path from the conventional ``*_solution.txt`` name."""
    name = Path(solution_path).name
    if not name.endswith("_solution.txt"):
        raise ValueError(f"not a solution file name: {solution_path}")
    instance_name = name[: -len("_solution.txt")] + ".txt"
    return Path(instances_dir) / instance_name


def load_solution_record(
    solution_path: str | Path,
    instance_path: str | Path | None = None,
) -> SolutionRecord:
    """Load and validate one existing solution file."""
    solution_path = Path(solution_path)
    instance_path = Path(instance_path) if instance_path else infer_instance_path(solution_path)
    instance = Instance(str(instance_path))
    state, routes = read_solution(str(solution_path), instance)
    if not validate_schedule(state["scheduled"], instance):
        raise ValueError(f"solution is not schedule-feasible: {solution_path}")
    cost = cost_from_routes(routes, instance)["total"]
    return SolutionRecord(
        instance_path=instance_path,
        solution_path=solution_path,
        cost=cost,
        tokens=Solution.to_sequence(state, routes),
    )


def load_valid_solution_records(
    solutions_dir: str | Path = "solutions",
    instances_dir: str | Path = "instances",
) -> list[SolutionRecord]:
    """Load every conventionally named valid solution under ``solutions_dir``."""
    records = []
    for path in discover_solution_files(solutions_dir):
        instance_path = infer_instance_path(path, instances_dir)
        if not instance_path.exists():
            continue
        try:
            records.append(load_solution_record(path, instance_path))
        except Exception:
            continue
    return records

