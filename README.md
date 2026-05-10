# OC Routing — VeRoLog 2017 CVRP Solver

Solver for the VeRoLog 2017 challenge: a capacitated vehicle routing problem where tools must be rented, delivered to customers, and picked up after a fixed loan duration. The objective is to minimise total cost across tool rental, vehicle usage, vehicle-days, and distance.

## How it works

1. **Greedy schedule** — requests are placed using earliest-deadline-first, producing a capacity-feasible initial schedule.

2. **ALNS optimisation** (`route_lns`) — an adaptive large neighbourhood search with simulated annealing acceptance. Each iteration destroys a subset of requests, repairs the schedule with a cost-targeted heuristic, re-routes only the affected days using OR-Tools, and accepts or rejects based on true routing cost. Operator weights adapt based on success history.

The routing solver uses OR-Tools: fast solves (SAVINGS heuristic, parallel threads) during the ALNS loop; a final GLS solve for the best found solution.

## Usage

```bash
# single run
python main.py instances/B2.txt

# repeated warm-start runs until no improvement
python run.py instances/B2.txt --patience 3

# run all instances (optionally in parallel)
python bench.py
python bench.py --workers 2
```

The solution is written to `instances/B2_solution.txt`. If a solution file already exists, the solver warm-starts from it and only overwrites if it finds a lower cost.

## Structure

```
instance.py      — loads and wraps the teacher-provided instance format
main.py          — entry point (CLI, greedy schedule, optimisation, file output)
run.py           — reruns main.py in a loop with warm-starting
bench.py         — runs all instances and prints a cost summary table
visualisation.py — generates a GIF animation of the daily vehicle routes

scheduling/
  state.py       — schedule state, commit/uncommit, greedy builder, validator
  cost.py        — cost estimation (tool, vehicle, distance)

routing/
  solver.py      — OR-Tools CVRP solver, Stop/VehicleRoute dataclasses
  export.py      — writes/reads VeRoLog solution files, exact cost from routes

optimiser/
  lns.py         — ALNS + SA optimisation loop
  break_fns.py   — destroy operators (tool cost, vehicle cost, distance, geographic)
  repair_fns.py  — repair operators (MRV insertion, cheapest insertion)
```

## Dependencies

```
ortools
tqdm
pandas
numpy
matplotlib
```
