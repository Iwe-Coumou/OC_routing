"""Persistent elite-solution archive for CVAE experiments."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from instance import Instance
from routing.export import cost_from_routes, read_solution, write_solution
from scheduling.state import validate_schedule


def archive_routes(
    *,
    instance_path: str | Path,
    instance: Instance,
    route_set: dict,
    archive_dir: str | Path = "solutions/cvae_archive",
    source: str = "search",
    keep_top: int | None = 50,
) -> Path:
    """Write a routed solution to the persistent archive."""
    if not route_set_is_complete(route_set, instance):
        raise ValueError("route set is incomplete and will not be archived")
    breakdown = cost_from_routes(route_set, instance)
    out_dir = Path(archive_dir) / Path(instance_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    filename = f"{int(breakdown['total']):013d}_{stamp}_{_safe_name(source)}.txt"
    output = out_dir / filename
    write_solution(route_set, instance, str(output))
    _append_manifest(out_dir, output, source, breakdown)
    if keep_top is not None and keep_top > 0:
        prune_archive(instance_path=instance_path, archive_dir=archive_dir, keep_top=keep_top)
    return output


def archive_solution_file(
    *,
    instance_path: str | Path,
    solution_path: str | Path,
    archive_dir: str | Path = "solutions/cvae_archive",
    source: str = "external",
    keep_top: int | None = 50,
) -> Path:
    """Validate and copy an existing solution file into the archive."""
    instance = Instance(str(instance_path))
    state, route_set = read_solution(str(solution_path), instance)
    if not validate_schedule(state["scheduled"], instance):
        raise ValueError(f"{solution_path} is not a complete valid schedule")
    breakdown = cost_from_routes(route_set, instance)
    out_dir = Path(archive_dir) / Path(instance_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    filename = f"{int(breakdown['total']):013d}_{stamp}_{_safe_name(source)}.txt"
    output = out_dir / filename
    shutil.copy2(solution_path, output)
    _append_manifest(out_dir, output, source, breakdown)
    if keep_top is not None and keep_top > 0:
        prune_archive(instance_path=instance_path, archive_dir=archive_dir, keep_top=keep_top)
    return output


def best_archived(
    *,
    instance_path: str | Path,
    archive_dir: str | Path = "solutions/cvae_archive",
) -> list[tuple[float, Path]]:
    """Return archived solutions sorted by exact current cost."""
    instance = Instance(str(instance_path))
    out_dir = Path(archive_dir) / Path(instance_path).stem
    records = []
    for path in out_dir.glob("*.txt"):
        try:
            state, route_set = read_solution(str(path), instance)
            if not route_set_is_complete(route_set, instance):
                continue
            if not validate_schedule(state["scheduled"], instance):
                continue
            records.append((float(cost_from_routes(route_set, instance)["total"]), path))
        except Exception:
            continue
    return sorted(records, key=lambda item: item[0])


def route_set_is_complete(route_set: dict, instance: Instance) -> bool:
    """Return True iff every request has exactly one correct delivery and pickup."""
    req_by_id = {req.id: req for req in instance.requests}
    deliveries: dict[int, list[int]] = {req.id: [] for req in instance.requests}
    pickups: dict[int, list[int]] = {req.id: [] for req in instance.requests}

    for day, routes in route_set.items():
        for route in routes:
            for stop in route.stops:
                if stop.request_id not in req_by_id:
                    return False
                if stop.action == "delivery":
                    deliveries[stop.request_id].append(day)
                elif stop.action == "pickup":
                    pickups[stop.request_id].append(day)
                else:
                    return False

    for req in instance.requests:
        if len(deliveries[req.id]) != 1 or len(pickups[req.id]) != 1:
            return False
        delivery_day = deliveries[req.id][0]
        pickup_day = pickups[req.id][0]
        if not req.earliest <= delivery_day <= req.latest:
            return False
        if pickup_day != delivery_day + req.duration:
            return False
    return True


def prune_archive(
    *,
    instance_path: str | Path,
    archive_dir: str | Path = "solutions/cvae_archive",
    keep_top: int = 50,
) -> None:
    """Keep only the current top-K valid archived solutions for one instance."""
    records = best_archived(instance_path=instance_path, archive_dir=archive_dir)
    keep = {path for _, path in records[:keep_top]}
    for _, path in records[keep_top:]:
        if path not in keep:
            path.unlink(missing_ok=True)


def _append_manifest(out_dir: Path, output: Path, source: str, breakdown: dict) -> None:
    record = {
        "file": output.name,
        "source": source,
        "cost": breakdown["total"],
        "max_vehicles": breakdown["max_vehicles"],
        "vehicle_days": breakdown["vehicle_days_count"],
        "tool": breakdown["tool"],
        "distance": breakdown["distance"],
        "timestamp": time.time(),
    }
    with (out_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:80] or "solution"


def _stamp() -> str:
    now = time.time()
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{time.time_ns() % 1_000_000_000:09d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive a validated VeRoLog solution")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--archive-dir", default="solutions/cvae_archive")
    parser.add_argument("--source", default="external")
    parser.add_argument("--keep-top", type=int, default=50)
    args = parser.parse_args()

    output = archive_solution_file(
        instance_path=args.instance,
        solution_path=args.solution,
        archive_dir=args.archive_dir,
        source=args.source,
        keep_top=args.keep_top,
    )
    print(output)


if __name__ == "__main__":
    main()
