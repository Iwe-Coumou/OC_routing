"""Run multiple independent CVAE searches and persist elite incumbents."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cvae.archive import best_archived


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel multi-seed CVAE search launcher")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--checkpoint-samples", type=int, default=16)
    parser.add_argument("--cp-time", type=float, default=1.0)
    parser.add_argument("--route-time", type=int, default=1)
    parser.add_argument("--routing-fixed-cost", type=int, default=100000)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--archive-dir", default="solutions/cvae_archive")
    parser.add_argument("--archive-top-k", type=int, default=50)
    parser.add_argument("--archive-all-feasible", action="store_true")
    parser.add_argument("--archive-cost-limit", type=float)
    parser.add_argument(
        "--encode-archives",
        action="store_true",
        help="Seed each worker from archived solutions (direct + CVAE-encoded). Recommended.",
    )
    parser.add_argument("--encode-archives-max", type=int, default=16)
    parser.add_argument("--output-dir", default="solutions/cvae_parallel")
    parser.add_argument("--log-dir", default="cvae/logs")
    args = parser.parse_args()

    instance_stem = Path(args.instance).stem
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[int, subprocess.Popen, object, object]] = []
    for idx in range(args.workers):
        seed = args.seed_start + idx
        output = output_dir / f"{instance_stem}_seed{seed}_solution.txt"
        stdout = (log_dir / f"{instance_stem}_seed{seed}_parallel.log").open("w", encoding="utf-8")
        stderr = (log_dir / f"{instance_stem}_seed{seed}_parallel.err").open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "cvae.search",
            "--instance",
            args.instance,
            "--checkpoint",
            args.checkpoint,
            "--checkpoint-samples",
            str(args.checkpoint_samples),
            "--budget",
            str(args.budget),
            "--population",
            str(args.population),
            "--iterations",
            str(args.iterations),
            "--cp-time",
            str(args.cp_time),
            "--route-time",
            str(args.route_time),
            "--routing-fixed-cost",
            str(args.routing_fixed_cost),
            "--seed",
            str(seed),
            "--output",
            str(output),
            "--archive-dir",
            args.archive_dir,
            "--archive-top-k",
            str(args.archive_top_k),
        ]
        if args.archive_all_feasible:
            cmd.append("--archive-all-feasible")
        if args.archive_cost_limit is not None:
            cmd.extend(["--archive-cost-limit", str(args.archive_cost_limit)])
        if args.encode_archives:
            cmd.append("--encode-archives")
            cmd.extend(["--encode-archives-max", str(args.encode_archives_max)])
        print(f"starting seed {seed}: {output}", flush=True)
        procs.append((seed, subprocess.Popen(cmd, stdout=stdout, stderr=stderr), stdout, stderr))

    failed = []
    for seed, proc, stdout, stderr in procs:
        rc = proc.wait()
        stdout.close()
        stderr.close()
        if rc != 0:
            failed.append((seed, rc))
        print(f"seed {seed} finished rc={rc}", flush=True)

    records = best_archived(instance_path=args.instance, archive_dir=args.archive_dir)
    if records:
        best_cost, best_path = records[0]
        print(f"best archived: {best_cost:,.0f} {best_path}", flush=True)
    if failed:
        raise SystemExit(f"failed searches: {failed}")


if __name__ == "__main__":
    main()
