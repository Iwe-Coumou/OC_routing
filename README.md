# OC Routing — VeRoLog 2017 CVRP Solver

Solver for the VeRoLog 2017 challenge: a capacitated vehicle routing problem where tools must be rented, delivered to customers, and picked up after a fixed loan duration. The objective is to minimise total cost across tool rental, vehicle usage, vehicle-days, and distance.

## Methods

Several solver methods are available, selected with `--method`:

**`alns`** (default) — Greedy construction followed by ALNS + simulated annealing. Supports warm-starting from a previous solution.

**`greedy_gls`** (benchmark) — Greedy construction (best of multiple orderings) followed directly by GLS routing, no schedule optimisation.

**`greedy_edd_gls`, `greedy_tight_gls`, `greedy_heavy_gls`, `greedy_late_gls`** (construction benchmarks) — Each uses a single fixed construction heuristic followed by GLS routing, allowing direct comparison of construction strategies.

## Usage

```bash
# single run (ALNS)
python main.py instances/B2.txt

# single run with a specific method
python main.py instances/B2.txt --method greedy_gls
python main.py instances/B2.txt --method greedy_tight_gls

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
  greedy_*/      — solution files for each construction benchmark

scheduling/
  state.py       — schedule state, commit/uncommit, greedy builder, validator
  cost.py        — cost estimation (tool, vehicle, distance)

routing/
  solver.py      — OR-Tools CVRP solver, Stop/VehicleRoute dataclasses
  export.py      — writes/reads VeRoLog solution files, exact cost from routes

optimiser/
  lns.py         — ALNS + SA optimisation loop
  break_fns.py   — destroy operators
  repair_fns.py  — repair operators
```

---

## Algorithm details

### Greedy construction

Multiple construction heuristics are tried and the one producing the lowest estimated cost is kept. The orderings are: earliest-deadline-first (EDD), tightest time window first, heaviest tool demand first, and latest-start first. Ten randomised orderings are also tried. Each ordering places requests one by one on the first feasible day; feasibility is checked in O(days) using a difference array over tool loans.

If a single fixed ordering is required (benchmark methods), `build_schedule_single` is used instead.

### ALNS with simulated annealing

The outer loop runs for a fixed number of iterations. Each iteration:

1. **Destroy** — a break operator removes a subset of requests from the schedule.
2. **Repair** — a repair operator reinserts them.
3. **Route** — the affected days are re-solved with OR-Tools to get an exact routing cost.
4. **Accept/reject** — the candidate is accepted if it improves cost, or with probability exp(−Δ/T) under simulated annealing (SA).

The SA temperature T starts at 2% of the initial cost and decays geometrically (α = 0.998), reaching ~37% of T₀ at iteration 500.

### Operator selection — unified adaptive pool

Break and repair operators are selected independently from two separate adaptive pools. Each pool uses the ALNS weight update rule: accepted improvements receive a reward multiplier (×1.5), SA-accepted moves a smaller reward (×1.05), and rejected moves a penalty (×0.90). Weights are bounded to [0.1, 10.0].

**Break pool** (`tool`, `vehicle`, `vehicle_days`, `distance`, `worst_day`, `random`, `geo`): cost-targeted operators are weighted by their adaptive weight multiplied by their normalised share of the current cost breakdown, so the search naturally focuses on whichever cost component currently dominates. `random` and `geo` use raw adaptive weights and start at 0.625 (giving ~20% combined probability at the start).

**Repair pool** (`tool`, `vehicle`, `vehicle_days`, `distance`): selected purely by adaptive weight, independent of the break operator chosen.

This decoupled design gives the search `n_break × n_repair` effective operator combinations, all explored and weighted by the data rather than fixed pairing assumptions.

### Adaptive destroy size

The destroy size k is scaled by `k_scale`, which starts at 1.0 and shrinks by 10% on every repair failure (floor 0.1), recovering by 2% on every success. On instances where tool availability is fully saturated and the schedule has little flexibility, `k_scale` quickly converges to 0.1, automatically reducing destroy sizes to avoid infeasible repairs.

### Break operators

| Operator | Structural property targeted | Mechanism |
|---|---|---|
| `break_tool_cost` | Tool rental peak | Finds the tool type and day with the highest weighted peak (concurrent loans × unit cost); removes all requests of that type active on that day |
| `break_vehicle_cost` | Busiest routing day | Finds the day using the most vehicles; removes requests on that day sorted by load |
| `break_vehicle_day_cost` | Route density | Scores requests by average route distance on their delivery/pickup days; removes top-k |
| `break_distance_cost` | Stop detour cost | Computes marginal detour per stop; removes the k requests with the highest total detour |
| `break_worst_day` | Worst routing day | Same detour scoring restricted to the single worst-distance day |
| `break_geographic` | Spatial clustering | Picks a random seed location; removes the k geographically nearest requests |
| `random` | (diversification) | Uniformly random sample of k requests |

### Repair operators

All repair operators use **MRV ordering** (Minimum Remaining Values): requests with fewer feasible days are inserted first to avoid dead-ends. Each operator chooses among feasible days using a different placement criterion:

| Operator | Placement criterion |
|---|---|
| `repair_tool_cost` | Minimises the new peak tool count across the loan window |
| `repair_vehicle_cost` | Minimises the projected peak vehicle count on delivery and pickup days |
| `repair_vehicle_day_cost` | Places on the day pair with the highest existing load (consolidates stops) |
| `repair_distance_cost` | Minimises cheapest-insertion cost into existing routes on delivery and pickup days |

A randomisation parameter ε gives each operator a chance to pick uniformly from all feasible days instead of the greedy-best, allowing the search to escape local optima.

### Feasibility and value tracking

The schedule state maintains two difference arrays per tool type: `loans[t][day]` records the net change in tools on loan (delivery: +machines, pickup: −machines), and `pickups_per_day[t][day]` tracks same-day returns. The current loan count on any day is the prefix sum of `loans`. The within-day peak — tools in use before morning pickups — is `prefix_sum + pickups[day]`, which must not exceed the available count.

This representation makes feasibility checks O(days) and commit/uncommit O(1).

The true routing cost is computed by OR-Tools after each repair step; no estimate is used for the accept/reject decision. The cost breakdown is maintained from the last accepted best solution and used to weight break operator selection.

### Performance

- **Incremental routing** — after each destroy/repair, only the days whose stop sets changed are re-solved; all other days carry over their existing routes unchanged.
- **Fast vs. quality solves** — the ALNS loop uses OR-Tools' SAVINGS heuristic (< 1 ms per day). The final solve uses GLS with a 30-second time limit per day.
- **Parallel day solves** — during fast routing, all days are solved concurrently via a `ThreadPoolExecutor`.
- **Difference arrays** — O(1) commit/uncommit and O(days) feasibility checks avoid re-scanning the full schedule on every ALNS move.
- **MRV ordering in repair** — inserting the most constrained requests first avoids dead-ends when the schedule is nearly full.

---

## Dependencies

```
ortools
tqdm
pandas
numpy
matplotlib
```
