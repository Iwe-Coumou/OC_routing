# OC Routing — VeRoLog 2017 CVRP Solver

Solver for the VeRoLog 2017 challenge: a capacitated vehicle routing problem where tools must be rented, delivered to customers, and picked up after a fixed loan duration. The objective is to minimise total cost across tool rental, vehicle usage, vehicle-days, and distance.

## How it works

The solver runs in two phases:

1. **Scheduling** — decides which day each request is delivered and picked up. Starts with a greedy earliest-deadline-first schedule, then improves it with an LNS loop using adaptive destroy/repair operators.

2. **Routing** — given the schedule, solves a daily CVRP for each day using OR-Tools. Fast solves (SAVINGS heuristic, parallel threads) are used during optimisation; a final GLS solve polishes the best found solution.

The main optimisation loop (`route_lns`) is an ALNS with simulated annealing acceptance. It destroys a subset of requests, repairs the schedule with a cost-targeted heuristic, re-routes only the affected days, and accepts or rejects based on true routing cost.

## Usage

```bash
# single run
python main.py instances/B2.txt

# repeated warm-start runs until no improvement
python run.py instances/B2.txt --patience 3
```

The solution is written to `instances/B2_solution.txt`. If a solution file already exists, the solver warm-starts from it and only overwrites if it finds a lower cost.

## Structure

```
instance.py          — loads and wraps the teacher-provided instance format
main.py              — entry point (CLI, scheduling, optimisation, file output)
run.py               — reruns main.py in a loop with warm-starting
visualisation.py     — generates a GIF animation of the daily vehicle routes

scheduling/
  state.py           — schedule state, commit/uncommit, greedy builder, validator
  cost.py            — cost estimation functions (tool, vehicle, distance)
  lns.py             — scheduling-phase LNS with ALNS weights
  analysis.py        — per-day and per-tool usage breakdown (printing)

routing/
  model.py           — OR-Tools CVRP solver, Stop/VehicleRoute dataclasses
  export.py          — writes/reads VeRoLog solution files, exact cost from routes

optimiser/
  lns.py             — main ALNS + SA optimisation loop (route_lns)
  break_fns.py       — destroy operators (tool cost, vehicle cost, distance, geographic)
  repair_fns.py      — repair operators (MRV insertion, cheapest insertion)
```

## Dependencies

```
ortools
tqdm
pandas
numpy
matplotlib
```
