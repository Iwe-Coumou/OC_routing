# OC Routing — VeRoLog 2017 CVRP Solver

Solver for the VeRoLog 2017 challenge: a capacitated vehicle routing problem where tools must be rented, delivered to customers, and picked up after a fixed loan duration. The objective is to minimise total cost across tool rental, vehicle usage, vehicle-days, and distance.

## Methods

Two solver methods are available, selected with `--method`:

**`alns`** (default) — Greedy construction followed by ALNS + simulated annealing. Supports warm-starting from a previous solution.

**`greedy_gls`** (benchmark) — Greedy construction followed directly by GLS routing, no schedule optimisation. Used as a baseline to quantify ALNS improvement.

## Usage

```bash
# single run (ALNS)
python main.py instances/B2.txt

# single run with benchmark method
python main.py instances/B2.txt --method greedy_gls

# repeated warm-start runs until no improvement
python run.py instances/B2.txt --patience 3

# run all instances with a given method
python bench.py
python bench.py --method greedy_gls

# compare costs across methods
python compare.py
python compare.py --detail
```

Solutions are written to `solutions/<method>/<instance>_solution.txt`. If a solution file already exists, the solver warm-starts from it (ALNS only) and only overwrites if it finds a lower cost.

## Structure

```
instance.py      — loads and wraps the teacher-provided instance format
main.py          — entry point (CLI, method dispatch, file output)
run.py           — reruns main.py in a loop with warm-starting
bench.py         — runs all instances and prints a cost summary table
compare.py       — compares solution costs across methods
visualisation.py — generates a GIF animation of the daily vehicle routes

solutions/
  alns/          — solution files produced by the ALNS method
  greedy_gls/    — solution files produced by the greedy+GLS benchmark

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

---

## Algorithm details

### Greedy construction

Requests are sorted by latest delivery deadline (earliest-deadline-first). Each request is inserted on the first feasible day in its time window `[earliest, latest]`. Feasibility is checked in O(days) using a difference array over tool loans: for each day from delivery through to pickup, the cumulative loan count plus same-day pickups must not exceed the available tool count for that type. The result is a complete, capacity-feasible schedule.

### ALNS with simulated annealing

The outer loop runs for a fixed number of iterations. Each iteration:

1. **Destroy** — a destroy operator removes a subset of requests from the schedule.
2. **Repair** — a repair operator reinserts them, targeting a specific cost driver.
3. **Route** — the affected days are re-solved with OR-Tools to get an exact routing cost.
4. **Accept/reject** — the candidate is accepted if it improves cost, or with probability exp(−Δ/T) under simulated annealing (SA).

The SA temperature T starts at 2% of the initial cost and decays geometrically (α = 0.998), reaching ~37% of T₀ at iteration 500. This allows early diversification and later intensification.

Operator weights adapt via ALNS: operators that produce accepted improvements receive a reward multiplier (×1.5); those producing SA-accepted moves get a smaller reward (×1.05); rejected operators are penalised (×0.90). Weights are bounded to [0.1, 10.0]. In each iteration, 80% of iterations use a cost-targeted destroy operator (selected proportionally to both weight and current cost breakdown), and 20% use random or geographic destroy with a randomly weighted repair.

### Destroy operators

| Operator | Cost driver | Mechanism |
|---|---|---|
| `break_tool_cost` | Tool rental | Finds the tool type and day with the highest weighted peak (concurrent loans × unit cost); removes all requests of that type active on that day |
| `break_vehicle_cost` | Fleet size | Finds the day using the most vehicles; removes requests on that day sorted by load (largest first) |
| `break_vehicle_day_cost` | Vehicle-days | Scores each request by average route distance on its delivery/pickup days; removes top-k |
| `break_distance_cost` | Routing distance | Computes detour cost per stop (marginal distance added to its route); removes the k requests with the highest total detour |
| `break_worst_day` | Routing distance | Same detour scoring but restricted to stops on the single worst-distance day |
| `break_geographic` | (diversification) | Picks a random seed location; removes the k geographically nearest requests |

### Repair operators

All repair operators use **MRV ordering** (Minimum Remaining Values): requests with fewer feasible days are inserted first to avoid dead-ends. Each targets a different cost component when choosing which feasible day to assign:

| Operator | Placement criterion |
|---|---|
| `repair_tool_cost` | Minimises the new peak tool count across the loan window |
| `repair_vehicle_cost` | Minimises the projected peak vehicle count on delivery and pickup days |
| `repair_vehicle_day_cost` | Places on the day pair with the highest existing load (consolidates stops) |
| `repair_distance_cost` | Minimises cheapest-insertion cost into existing routes on delivery and pickup days |

A randomisation parameter ε gives each operator a chance to pick uniformly from all feasible days instead of the greedy-best, allowing the search to escape local optima.

### Feasibility and value tracking

The schedule state maintains two difference arrays per tool type: `loans[t][day]` records the net change in tools on loan (delivery: +machines, pickup: −machines), and `pickups_per_day[t][day]` tracks same-day returns. The current loan count on any day is the prefix sum of `loans`. The within-day peak — the number of tools simultaneously in use before morning pickups are counted — is `prefix_sum + pickups[day]`, which must not exceed the available count.

This representation makes feasibility checks O(days) and commit/uncommit O(1) (just increment/decrement two array entries).

The true routing cost is computed by OR-Tools after each repair step; no estimate is used for the accept/reject decision. The cost breakdown (tool rental, fleet size, vehicle-days, distance) is maintained from the last accepted best solution and used to weight operator selection.

### Performance

Several decisions keep the ALNS loop fast enough for 500 iterations on large instances:

- **Incremental routing** — after each destroy/repair, only the days whose stop sets changed are re-solved; all other days carry over their existing routes unchanged.
- **Fast vs. quality solves** — during the ALNS loop, routing uses OR-Tools' SAVINGS heuristic with no time limit (< 1 ms per day). The final solve uses GLS with a 30-second time limit per day to squeeze out extra routing quality.
- **Parallel day solves** — during fast routing, all days are submitted concurrently to a `ThreadPoolExecutor`, exploiting OR-Tools' thread-safety and the independence of each day's routing subproblem.
- **Difference arrays** — O(1) commit/uncommit and O(days) feasibility checks avoid re-scanning the full schedule on every ALNS move.
- **MRV ordering in repair** — inserting the most constrained requests first avoids expensive backtracking when the schedule is nearly full.

---

## Dependencies

```
ortools
tqdm
pandas
numpy
matplotlib
```
