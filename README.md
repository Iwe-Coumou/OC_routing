# OC Routing — VeRoLog 2017 CVRP Solver

Solver for the VeRoLog 2017 Solver Challenge: a multi-day capacitated vehicle routing problem where specialised tools (machines) must be rented, delivered to customers within time windows, and collected after a fixed loan duration. The objective is to minimise total cost across four components: tool rental, vehicle fleet size, vehicle-days used, and distance travelled.

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Input and Output Format](#3-input-and-output-format)
4. [Solution Approach Overview](#4-solution-approach-overview)
5. [Algorithm Details](#5-algorithm-details)
6. [Implementation Details](#6-implementation-details)
7. [Code Structure](#7-code-structure)
8. [Usage](#8-usage)
9. [Results](#9-results)
10. [Dependencies](#10-dependencies)

---

## 1. Problem Description

The problem is a variant of the **Capacitated Vehicle Routing Problem with Time Windows and Unavoidable Returns (CVRPTWUI)**. A depot owns a fleet of vehicles and a pool of specialised tools (machines). Customers submit requests: each request specifies a delivery time window, a tool type, a quantity of tools required, and a loan duration (the number of days the tools must remain at the customer's site before they can be collected).

The planning horizon spans multiple days (10–25 in the benchmark set). Each day, vehicles depart from the depot, serve a set of deliveries and pickups, and return to the depot. A vehicle's total route distance for a single day cannot exceed `MAX_TRIP_DISTANCE`; if a vehicle needs to travel further, it must return to the depot to reload and make a second trip (a **multi-trip** vehicle).

The key coupling between days is the tool inventory: tools delivered on day `d` must be picked up on day `d + duration`. This means delivery and pickup decisions on different days are not independent — they share the limited tool pool.

### What makes this problem hard

- Tool availability couples deliveries across all days: scheduling a delivery commits tools until pickup, constraining what can be scheduled later.
- The four cost components pull in different directions. Reducing tool rental peaks pushes requests to be scheduled close together (short windows), which can increase vehicle-days. Consolidating routes reduces distance but may require more vehicles on peak days.
- The two-phase structure (schedule assignment + routing) means the scheduling layer can only estimate routing cost; the true cost is only known after solving the routing sub-problem.

---

## 2. Mathematical Formulation

### Notation

| Symbol | Meaning |
|---|---|
| $D$ | Number of planning days |
| $R$ | Set of requests |
| $K$ | Set of tool types |
| $n_k$ | Number of tools of type $k$ available at the depot |
| $c_k$ | Daily rental cost per tool of type $k$ |
| $[e_r, l_r]$ | Delivery time window for request $r$ |
| $\delta_r$ | Loan duration for request $r$ (days) |
| $q_r$ | Number of tools required by request $r$ |
| $W$ | Vehicle weight capacity |
| $s_k$ | Weight (size) of one tool of type $k$ |
| $c_V$ | Fixed vehicle cost (paid once per vehicle ever used) |
| $c_{VD}$ | Vehicle-day cost (paid per vehicle per day it operates) |
| $c_D$ | Distance cost (per unit of distance) |

### Decision variables

- $d_r \in [e_r, l_r]$ — delivery day for request $r$
- $p_r = d_r + \delta_r$ — pickup day (derived, must satisfy $p_r \leq D$)
- Route assignment: which vehicle visits which customer on each day

### Objective function

$$\min \quad \underbrace{c_V \cdot \max_d |V_d|}_{\text{vehicle cost}} + \underbrace{c_{VD} \cdot \sum_d |V_d|}_{\text{vehicle-day cost}} + \underbrace{c_D \cdot \text{TotalDistance}}_{\text{distance cost}} + \underbrace{\sum_{k \in K} c_k \cdot \hat{n}_k}_{\text{tool rental cost}}$$

where $|V_d|$ is the number of vehicles used on day $d$ and $\hat{n}_k$ is the peak number of tools of type $k$ that are simultaneously on loan across all days.

The **tool rental cost** depends on the peak simultaneous loan count:

$$\hat{n}_k = \max_{d=1}^{D} \left( \text{tools of type } k \text{ at customers' sites on day } d \text{ before pickups} \right)$$

### Constraints

1. **Time window**: $e_r \leq d_r \leq l_r$ for all $r \in R$
2. **Horizon**: $d_r + \delta_r \leq D$ for all $r \in R$
3. **Tool availability** (within-day peak): for each tool type $k$ and each day $d$,
   $$\sum_{r : d_r \leq d < p_r} q_r \;\leq\; n_k$$
   This uses the morning peak convention: pickups collected on day $d$ reduce inventory only after the morning delivery peak has passed.
4. **Vehicle capacity**: the total weight loaded on any single vehicle trip does not exceed $W$
5. **Trip distance**: each vehicle trip (depot–customers–depot leg) does not exceed `MAX_TRIP_DISTANCE`
6. **All requests served**: every request must be assigned a delivery day and appear in exactly one delivery route and one pickup route

---

## 3. Input and Output Format

### Instance file

The instance files use the VeRoLog 2017 text format. The key sections are:

```
DAYS = 15                      # planning horizon
CAPACITY = 45                  # vehicle weight capacity
MAX_TRIP_DISTANCE = 16000      # maximum single-trip distance
DEPOT_COORDINATE = 0           # which coordinate index is the depot

VEHICLE_COST = 1               # multiplied by max vehicles ever used
VEHICLE_DAY_COST = 3000        # multiplied by total vehicle-days
DISTANCE_COST = 100000         # multiplied by total distance

TOOLS = 2                      # number of tool types
# id   weight   available   daily_cost
1      3        84          20000000
2      15       48          20000000

COORDINATES = 200              # number of locations (including depot)
# id   x        y
0      5445     2890           # depot
1      8341     6147
...

REQUESTS = 200
# id   location   earliest   latest   duration   tool_type   quantity
1      45         1          5        3          1           2
...
```

The distance matrix is computed as Euclidean distance between coordinates (via `calculateDistances()` in the teacher-provided parser).

### Solution file

Solutions follow the VeRoLog output format:

```
DATASET = VeRoLog solver challenge 2017
NAME = <instance name>

MAX_NUMBER_OF_VEHICLES = 19       # peak vehicles used on any single day
NUMBER_OF_VEHICLE_DAYS = 124      # sum of vehicles used across all days
TOOL_USE = 100 101 87             # peak simultaneous loan count per tool type
DISTANCE = 2327725                # total distance across all routes
COST = 1047575725                 # total objective value

DAY = 1
NUMBER_OF_VEHICLES = 19
START_DEPOT = 50 47 48            # tools loaded at start of day (per type)
FINISH_DEPOT = 50 47 48           # tools returned at end of day (per type)
1   R   0   492   0               # vehicle 1 route: depot→req492(delivery)→depot
1   V   1   0   -3   0            # vehicle 1, depot visit 1: loads 3 type-2 tools
1   V   2   0   0   -2            # depot visit 2 (for multi-trip): loads 2 type-3 tools
1   V   3   0   0   0             # final depot visit: returns 0 tools
1   D   19550                     # vehicle 1 total distance for the day
...
```

Routes use positive request IDs for deliveries and negative IDs for pickups. A `0` token in a route line marks the depot (start, end, or between trips for multi-trip vehicles).

---

## 4. Solution Approach Overview

The solver uses a two-phase decomposition:

1. **Scheduling phase** — assign a delivery day $d_r$ to each request, satisfying time-window and tool-availability constraints. This phase works purely on the schedule abstraction; routing is not solved.
2. **Routing phase** — given a fixed schedule, solve a daily CVRP for each day using OR-Tools. Each day is solved independently.

These two phases are coupled in the ALNS optimisation loop: the scheduling layer proposes changes (destroy + repair), and then only the affected days are re-routed. The true objective value is the routed cost; the scheduling layer uses cheapest-insertion and load-based estimates to guide search without routing every candidate.

### Solving pipeline for `--method alns` (default)

```
1. Greedy construction  →  schedule all requests, pick best of 5 orderings
2. CP-SAT repair        →  fix any unscheduled requests (rarely needed)
3. Fast routing         →  OR-Tools SAVINGS, aggregate capacity model
4. ALNS + SA loop       →  500 iterations × up to 3 restarts
     ├─ destroy: remove k requests from schedule
     ├─ repair:  reinsert using a cost-targeted heuristic
     └─ evaluate: re-route changed days, accept/reject via SA
5. Final quality routing →  GLS, full per-type capacity model, 15 s/day
6. Write solution        →  overwrite only if cost improved
```

---

## 5. Algorithm Details

### 5.1 Greedy construction

Five construction heuristics are tried; the one producing the lowest estimated cost (without routing) is kept.

Each heuristic sorts requests by a priority key and places them one by one on the earliest feasible day:

| Key name | Sort order | Rationale |
|---|---|---|
| `edd` (Earliest Deadline First) | `(latest, num_machines × duration)` | Serve tight-deadline requests first to avoid infeasibility |
| `tight` | `(latest − earliest, latest)` | Smallest time window first |
| `heavy` | `(−num_machines, latest)` | Largest load first — hardest to fit into capacity |
| `late` | `(−earliest, latest)` | Latest-starting requests first |
| `mrv` | Dynamic (see below) | Minimum Remaining Values — constraint satisfaction ordering |

**MRV ordering** (`place_unscheduled_mrv`): at each step, rather than using a fixed sort key, the algorithm counts how many feasible days remain for each unscheduled request and selects the one with the fewest options (most constrained). In case of ties it prefers narrower time windows, then earlier deadlines. This prevents blocking future placements by greedily consuming scarce days.

**Ejection chain repair** (`repair_by_ejection_chain`): if a heuristic leaves requests unscheduled, an ejection chain pass attempts to make room. For each unscheduled request (MRV-ordered), it tries direct placement first; if that fails, it finds a scheduled request of the same tool type, temporarily ejects it, inserts the blocked request, and tries to reinsert the ejected one. Each request may be ejected at most once per call, guaranteeing termination. This typically resolves the few infeasibilities that the greedy pass cannot handle.

**CP-SAT repair** (`repair_feasibility`): if the ejection chain cannot schedule all requests (which can happen on instances where tool availability is very tight), the CP-SAT solver rescheduled all requests of the blocked tool type as an independent interval-scheduling problem. The cumulative constraint enforces the tool availability limit. Current assignments are passed as hints to warm-start the solver. This is a last resort and rarely triggers in practice.

### 5.2 Cost estimation

Before routing is solved, the scheduling layer estimates cost using:

- **Tool cost**: computed exactly from the `loans` difference arrays — O(days) prefix sum to find the peak concurrent loan count per tool type.
- **Vehicle estimate**: total load per day divided by capacity gives a lower bound on vehicles. Nearest-neighbour tour distance gives an estimate of routing cost.

This estimate is used only during construction to pick the best ordering; the true cost is always the routed cost from OR-Tools.

### 5.3 OR-Tools routing

Each day's routing problem is a CVRP solved independently by OR-Tools' constraint programming solver.

**Model**: `RoutingIndexManager` and `RoutingModel` with nodes for the depot (node 0) and each delivery/pickup stop. The arc cost is `distance × DISTANCE_COST`; a fixed cost per vehicle equals `VEHICLE_DAY_COST`.

**Two solve modes** are used at different stages:

| Mode | Capacity model | First solution | Local search | Time limit | Use |
|---|---|---|---|---|---|
| **Fast** | Single aggregate dimension | SAVINGS | None | None | ALNS inner loop |
| **Quality** | Per-type dimensions + cross-constraints | Parallel cheapest insertion | GLS (λ=0.1) | 15 s/day | Final solve |

The **aggregate capacity dimension** in fast mode treats all tools as a single commodity with a `±load` callback. Since all loads are non-negative, `∑loads ≤ capacity` implies each individual tool type is also within capacity. This eliminates `n_types × n_nodes` CP variables and reduces SAVINGS solve time by roughly 10–20× on dense days.

The **quality mode** uses a separate dimension per tool type plus explicit cross-constraints (`∑ cumulative variables ≤ capacity` for every node and vehicle endpoint). This gives the local search more structure to work with but is slower.

**Vehicle count**: the solver caps the number of vehicles at `min_cap + 5` (fast) or `max(min_cap + 3, min_cap × 2)` (quality), where `min_cap = ⌈max(delivery_load, pickup_load) / capacity⌉`. If OR-Tools fails to find a solution within that cap, it retries with as many vehicles as there are stops.

**Multi-trip vehicles**: after OR-Tools returns single-trip routes, `merge_routes()` uses first-fit decreasing bin packing: routes are sorted by distance descending and packed into bins such that total distance per bin does not exceed `MAX_TRIP_DISTANCE`. A vehicle assigned multiple trips returns to the depot between trips.

**Warm-starting**: when the quality solver is called after the ALNS loop, the best routes found during the loop are passed as an initial assignment. OR-Tools uses `ReadAssignmentFromRoutes` to seed GLS from this solution rather than starting from scratch.

### 5.4 ALNS + simulated annealing

The main optimisation loop is Adaptive Large Neighbourhood Search (ALNS) combined with simulated annealing (SA) for acceptance.

**Loop structure**:
```
for iteration in range(iterations):
    snap = snapshot(state)
    select break operator (weighted random)
    select repair operator (weighted random)
    remove k requests from schedule
    reinsert requests using repair operator
    if any requests unscheduled: restore snap, penalise operators
    else:
        re-route changed days (fast mode, incremental)
        delta = candidate_cost - current_cost
        if delta < 0 or random() < exp(-delta / T):
            accept; update weights
        else:
            restore snap; penalise operators
        update best-feasible tracker
    update T
    if no_improve >= patience: restart or stop
```

**Simulated annealing**:
- Initial temperature $T_0 = 0.02 \times \text{initial routed cost}$
- Geometric cooling: $T \leftarrow T \times 0.998$ each iteration
- Acceptance probability for non-improving moves: $\exp(-\Delta / T)$

**Restarts**: after 150 consecutive iterations without improvement (patience), instead of stopping, the search resets all operator weights to their initial values and reheats $T$ to $0.5 \times T_0$. Up to 3 restarts are performed before stopping. This escapes local optima without discarding the best solution found.

**Incremental routing**: after each destroy/repair, only the days whose stop sets changed are re-routed. Unchanged days carry their existing routes. This reduces the routing overhead from O(days) to O(changed days) per iteration, giving roughly a 10–20× speedup on large instances.

### 5.5 Adaptive operator selection

Break and repair operators are selected independently from two separate adaptive pools. Each pool uses the ALNS weight update rule:

| Event | Weight multiplier |
|---|---|
| Move improved the best known cost | × 1.5 (reward) |
| Move accepted via SA (non-improving) | × 1.05 (small reward) |
| Move rejected | × 0.90 (penalty) |

Weights are bounded to [0.1, 10.0].

**Break pool**: cost-targeted operators (`tool`, `vehicle`, `vehicle_days`, `distance`, `worst_day`) are weighted by `adaptive_weight × (component_cost / total_cost)`. This means operators targeting the currently dominant cost component are automatically selected more often. The `random` and `geo` operators use raw adaptive weights and start at 0.625, giving them a combined ~20% initial probability.

**Repair pool**: the four repair operators are weighted purely by adaptive weight, selected independently of which break operator was chosen. This decoupled design allows all 7 × 4 = 28 operator combinations to be explored and weighted.

### 5.6 Destroy operators

| Operator | What it removes | Structural property targeted |
|---|---|---|
| `break_tool_cost` | All requests of the tool type with the highest weighted peak (concurrent loans × cost/unit), active on the peak day | Tool rental cost |
| `break_vehicle_cost` | Requests on the day with the most vehicles, sorted by load descending | Vehicle fleet size |
| `break_vehicle_day_cost` | Top-k requests by average route distance on their delivery/pickup days | Vehicle-day cost / route density |
| `break_distance_cost` | Top-k requests by total marginal detour across delivery + pickup stops | Distance cost |
| `break_worst_day` | Requests active on the day with highest total route distance, sorted by detour | Worst-day distance |
| `break_geographic` | k requests geographically nearest to a random seed location | Spatial clustering / diversification |
| `random` | Uniform random sample of k requests | Pure diversification |

**Adaptive destroy size** (`k_scale`): the number of requests removed scales with `k_scale`, which starts at 1.0 and shrinks by 10% whenever repair fails to schedule all requests (floor 0.1), recovering by 2% on each successful repair. On tight instances where the tool pool is nearly saturated, `k_scale` quickly converges to a small value, automatically reducing destroy size to avoid infeasible repairs.

### 5.7 Repair operators

All repair operators use **MRV ordering**: unscheduled requests are sorted by number of remaining feasible days (ascending) before insertion, so the most constrained requests are placed first. Ties are broken by window width then by deadline.

| Operator | Day selection criterion |
|---|---|
| `repair_tool_cost` | Minimises the new peak tool count across the loan window `[d, p]` |
| `repair_vehicle_cost` | Minimises the projected peak vehicle count on delivery and pickup days; tie-broken by maximising existing load (consolidation) |
| `repair_vehicle_day_cost` | Maximises `load[d] + load[p]` — places request on the day pair with highest existing load to consolidate trips |
| `repair_distance_cost` | Minimises cheapest-insertion cost into existing routes on delivery and pickup days; falls back to nearest-neighbour distance estimate if routes are not available |

Each operator also has a **randomisation parameter ε = 0.25**: with 25% probability, the operator selects uniformly from all feasible days instead of the greedy-best, providing diversification and helping escape local optima.

### 5.8 Feasibility tracking

The schedule state maintains two arrays per tool type:

- `loans[k][d]` — difference array: +`q_r` on `delivery_day`, −`q_r` on `pickup_day`. The prefix sum gives the number of tools of type `k` on loan at the start of day `d`.
- `pickups_per_day[k][d]` — total machines of type `k` scheduled for pickup on day `d`.

The **within-day peak** on day `d` is `prefix_sum(d) + pickups_per_day[d]`, which gives the number of tools at customers' sites before the morning pickup (the convention used by the official validator).

**Commit/uncommit**: updating the schedule is O(1) — only two entries in the difference array change. Feasibility checking is O(days) — one prefix-sum scan from 0 to `pickup_day`. This makes it cheap to evaluate thousands of candidate day assignments per ALNS iteration.

**Snapshot/restore**: the full schedule can be saved as a list of (request, delivery_day) pairs and restored in O(n) — used for SA rejection and best-feasible tracking.

---

## 6. Implementation Details

### Key data structures

```
state = {
    'loans':           defaultdict(list),   # difference arrays, one per tool type
    'pickups_per_day': defaultdict(list),   # same-day return counts per tool type
    'scheduled':       list[dict],          # {request, delivery_day, pickup_day}
    'unscheduled':     defaultdict(list),   # requests not yet placed, per tool type
}

route_set = {
    day: list[VehicleRoute]                 # one VehicleRoute per vehicle per day
}

VehicleRoute = {
    vehicle_id: int,
    stops: list[Stop],                      # ordered list of delivery/pickup stops
    distance: int,
    trips: list[list[Stop]],               # non-empty only for multi-trip vehicles
}

Stop = {
    request_id: int,
    action: 'delivery' | 'pickup',
    location_id: int,
    load: int,                             # weight of this stop (num_machines × size)
    machine_type: int,
}
```

### Performance design choices

- **Incremental routing**: the ALNS loop re-routes only changed days — if 3 days change out of 20, only those 3 are solved. This is the dominant speedup: routing is much more expensive than scheduling.
- **Fast vs. quality routing**: the ALNS loop uses the aggregate capacity model (fast), reducing the CP variable count by a factor equal to the number of tool types. The final solve uses the full per-type model with GLS.
- **MRV in repair**: placing the most-constrained requests first avoids dead-ends. Without MRV ordering, repairs on tight instances frequently fail to schedule all requests.
- **Difference arrays**: O(1) commit/uncommit avoids scanning the full schedule on every ALNS move.
- **Warm-start routing**: the quality final solve is seeded from the best routes found during the ALNS loop, reducing the time GLS needs to reach a good solution.

---

## 7. Code Structure

```
OC_routing/
│
├── instance.py               # Loads instance file, wraps teacher parser into clean dataclasses
├── main.py                   # CLI entry point: argument parsing, method dispatch, file I/O
├── run.py                    # Warm-start loop: repeatedly calls main.py, stops on no improvement
├── bench.py                  # Batch runner: solves all instances in instances/ directory
├── compare.py                # Comparison table: prints costs side-by-side across methods
├── visualisation.py          # GIF animation of daily vehicle routes
├── test_feasibility.py       # Standalone test for the CP-SAT feasibility repair module
│
├── scheduling/
│   ├── state.py              # Core schedule state: build_state, commit/uncommit,
│   │                         #   is_feasible, snapshot/restore, construction heuristics,
│   │                         #   ejection chain, MRV placement, validator
│   ├── cost.py               # Cost estimation: tool peak, vehicle estimate, distance estimate,
│   │                         #   print_cost, cost_breakdown
│   └── feasibility.py        # CP-SAT repair for over-subscribed tool types
│
├── routing/
│   ├── solver.py             # OR-Tools CVRP: Stop/VehicleRoute dataclasses,
│   │                         #   build_daily_stops, solve_day, solve_all_days,
│   │                         #   solve_routing, solve_routing_incremental, merge_routes
│   └── export.py             # Solution I/O: write_solution (VeRoLog format), read_solution,
│                             #   cost_from_routes (exact), depot inventory tracking
│
├── optimiser/
│   ├── lns.py                # ALNS + SA main loop: operator selection, weight updates,
│   │                         #   SA acceptance, restart logic, incremental routing calls
│   ├── break_fns.py          # 7 destroy operators
│   └── repair_fns.py         # 4 repair operators
│
├── InstanceCVRPTWUI.py       # Teacher-provided instance parser (do not modify)
├── baseCVRPTWUI.py           # Teacher-provided base parser class (do not modify)
├── Validate.py               # Teacher-provided solution validator (do not modify)
│
├── instances/                # Problem instance files (.txt)
└── solutions/
    ├── alns/                 # Solutions from the ALNS method
    ├── greedy_gls/           # Solutions from the greedy+GLS benchmark
    └── greedy_{edd,tight,heavy,late}_gls/   # Single-ordering benchmark solutions
```

### Module responsibilities

**`instance.py`**: wraps the teacher's `InstanceCVRPTWUI` parser into three clean dataclasses (`Config`, `Request`, `Tool`) and a distance matrix. The `Instance` object is the single source of truth passed to all other modules.

**`scheduling/state.py`**: the most central module. The `state` dict is the mutable schedule; all other scheduling functions operate on it. `build_schedule` tries all five construction orderings and returns the best. `validate_schedule` runs an independent feasibility check (used for debugging).

**`scheduling/cost.py`**: provides `cost_breakdown`, a fast O(n·days) estimate of all four cost components from the schedule alone (without routing). Used during construction to compare orderings and during the ALNS loop to weight the break operators.

**`scheduling/feasibility.py`**: last-resort repair using OR-Tools CP-SAT. Treats each tool type's scheduling problem as an independent interval scheduling problem with a cumulative constraint. Called at most once per run, before the ALNS loop starts.

**`routing/solver.py`**: wraps OR-Tools. `solve_routing_incremental` is the performance-critical function called inside the ALNS loop — it skips unchanged days. `solve_routing` is used for the initial fast solve and the final quality solve.

**`routing/export.py`**: handles reading and writing solution files in VeRoLog format. `cost_from_routes` computes the exact objective value (matching the official validator) by tracking depot tool inventories across all days. `read_solution` parses a solution file back into a `state` dict + `route_set`, enabling warm-starting.

**`optimiser/lns.py`**: the ALNS main loop. Manages weights, temperature, restarts, incremental routing, and the best-feasible tracker. Calls into `break_fns`, `repair_fns`, `scheduling.state`, and `routing.solver`.

---

## 8. Usage

### Prerequisites

```bash
pip install ortools tqdm pandas numpy matplotlib
```

### Single instance

```bash
# ALNS (default, best quality)
python main.py instances/B2.txt

# Greedy construction + GLS routing (no ALNS, faster)
python main.py instances/B2.txt --method greedy_gls

# Single fixed construction heuristic (for benchmarking individual orderings)
python main.py instances/B2.txt --method greedy_edd_gls
python main.py instances/B2.txt --method greedy_tight_gls
python main.py instances/B2.txt --method greedy_heavy_gls
python main.py instances/B2.txt --method greedy_late_gls
```

The solver automatically warm-starts from an existing solution file (ALNS only) and only overwrites it if the new cost is lower.

### Repeated warm-start loop

```bash
# Run until 3 consecutive runs with no improvement
python run.py instances/B2.txt --patience 3

# Run exactly 10 times
python run.py instances/B2.txt --runs 10

# Run indefinitely (Ctrl+C to stop)
python run.py instances/B2.txt
```

Each call to `run.py` launches `main.py` as a subprocess, which reads the existing solution file and warm-starts from it. This allows the solver to iteratively improve solutions over multiple runs.

### Batch solve

```bash
# Solve all r300 and r500 instances
python bench.py

# Solve with a different method
python bench.py --method greedy_gls

# Skip instances that already have a solution
python bench.py --skip-solved
```

### Compare methods

```bash
# Print cost table for all methods that have solution files
python compare.py

# Show per-component breakdown
python compare.py --detail

# Compare two specific methods
python compare.py --methods alns greedy_gls
```

### Visualisation

```bash
# Edit the paths at the bottom of visualisation.py then run:
python visualisation.py
```

Produces a GIF animating vehicle routes for each day in sequence.

### Feasibility repair test

```bash
python test_feasibility.py                   # tests on instances/B3.txt
python test_feasibility.py instances/B2.txt  # test on a specific instance
```

### Solution files

Solutions are written to `solutions/<method>/<instance>_solution.txt`. The file includes all four cost components and the full route plan in VeRoLog format. If a solution file already exists, the solver reads the cost from it and only overwrites if the new cost is strictly lower.

---

## 9. Results

The table below compares ALNS against greedy+GLS across the instances solved. Instances marked with `—` were not solved with that method.

### Instance set overview

| Set | Requests | Days | Capacity |
|---|---|---|---|
| B1 | 200 | 15 | 45 |
| B2 | 500 | 15 | 50 |
| B3 | 500 | 10 | 40 |
| r100d10 | 100 | 10 | 45–50 |
| r200d15 | 200 | 15 | 45–50 |
| r300d20 | 300 | 20 | 35–50 |
| r500d25 | 500 | 25 | 40–45 |

### Cost comparison (ALNS vs greedy+GLS)

| Instance | ALNS cost | greedy+GLS cost | Improvement |
|---|---|---|---|
| challenge_r300d20_1 | 3,924,190,461 | 4,475,199,432 | −12.3% |
| challenge_r300d20_2 | 3,029,432,058 | 3,211,839,358 | −5.7% |
| challenge_r300d20_3 | 208,718,468 | 240,754,181 | −13.3% |
| challenge_r300d20_4 | 2,976,341,170 | 3,274,511,654 | −9.1% |
| challenge_r300d20_5 | 234,635,680,005 | 281,870,152,007 | −16.7% |
| challenge_r500d25_1 | 1,863,837,078 | 1,963,726,558 | −5.1% |
| challenge_r500d25_2 | 4,089,184,675 | 4,500,414,235 | −9.1% |
| challenge_r500d25_3 | 323,349,507 | 413,149,489 | −21.7% |
| **Average** | | | **−11.6%** |

### ALNS solution breakdown (B3)

| Component | Value |
|---|---|
| Total cost | 1,047,575,725 |
| Max vehicles used (any day) | 19 |
| Total vehicle-days | 124 |
| Total distance | 2,327,725 |
| Tool use (peak per type) | 100, 101, 87 |

### ALNS solution breakdown (r500d25 instances)

| Instance | Cost | Max vehicles | Vehicle-days | Distance |
|---|---|---|---|---|
| r500d25_1 | 1,863,837,078 | 8 | 123 | 1,745,078 |
| r500d25_2 | 4,089,184,675 | 5 | 93 | 1,274,675 |
| r500d25_3 | 323,349,507 | 9 | 176 | 2,245,507 |

---

## 10. Dependencies

| Package | Purpose |
|---|---|
| `ortools` | CVRP routing solver (CP-SAT + routing library) |
| `tqdm` | Progress bars in the ALNS loop and routing |
| `pandas` | Data handling (used internally by teacher parser) |
| `numpy` | Coordinate and distance computations |
| `matplotlib` | GIF animation in `visualisation.py` |

Install with:

```bash
pip install ortools tqdm pandas numpy matplotlib
```
