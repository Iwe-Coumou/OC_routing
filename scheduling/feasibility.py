import logging
from ortools.sat.python import cp_model
from instance import Instance
from scheduling.state import commit_request, uncommit_request

log = logging.getLogger(__name__)


def repair_feasibility(state: dict, instance: Instance) -> bool:
    """Reschedule over-subscribed tool types using OR-Tools CP-SAT.

    For each tool type that has unscheduled requests, all requests of that
    type are uncommitted and re-solved as an independent interval-scheduling
    problem.  Tool types are independent (separate capacity pools), so
    solving them one at a time is safe.

    Returns True if all requests are now scheduled.
    """
    unscheduled_types = sorted({
        req.machine_type
        for reqs in state['unscheduled'].values()
        for req in reqs
    })

    for machine_type in unscheduled_types:
        all_requests = [r for r in instance.requests if r.machine_type == machine_type]

        # Preserve current delivery-day assignments as hints for the solver.
        hints = {
            e['request'].id: e['delivery_day']
            for e in state['scheduled']
            if e['request'].machine_type == machine_type
        }

        # Uncommit every scheduled request of this type before re-solving.
        to_uncommit = [
            e['request'] for e in state['scheduled']
            if e['request'].machine_type == machine_type
        ]
        for req in to_uncommit:
            uncommit_request(state, req)

        log.info(f"feasibility CP-SAT: type={machine_type}  n_requests={len(all_requests)}")
        solution = _cp_sat_schedule(all_requests, instance, hints)

        if solution is None:
            log.warning(f"feasibility CP-SAT: type={machine_type} — no feasible schedule exists")
            return False

        for req, day in solution:
            commit_request(state, instance, req, day)

    return sum(len(v) for v in state['unscheduled'].values()) == 0


def _cp_sat_schedule(requests: list, instance: Instance, hints: dict) -> list | None:
    """Solve the interval-scheduling subproblem for one tool type.

    Each request occupies [delivery_day, pickup_day] inclusive (span =
    duration + 1 days).  The cumulative constraint enforces that the total
    machines on loan never exceeds the available count.

    Returns [(request, delivery_day), ...] or None if infeasible.
    """
    if not requests:
        return []

    machine_type = requests[0].machine_type
    tool    = next(t for t in instance.tools if t.id == machine_type)
    horizon = instance.config.days

    model = cp_model.CpModel()
    start_vars: list = []
    intervals:  list = []
    demands:    list = []

    for req in requests:
        # Loan is active on days [delivery_day, pickup_day] inclusive.
        # CP-SAT interval [start, start+span) covers exactly those days
        # when span = req.duration + 1.
        span  = req.duration + 1
        start = model.NewIntVar(req.earliest, req.latest,         f's{req.id}')
        end   = model.NewIntVar(req.earliest + span, req.latest + span, f'e{req.id}')
        iv    = model.NewIntervalVar(start, span, end,            f'i{req.id}')

        model.Add(start + req.duration <= horizon)  # pickup must not exceed horizon

        start_vars.append((req, start))
        intervals.append(iv)
        demands.append(req.num_machines)

        if req.id in hints:
            model.AddHint(start, hints[req.id])

    model.AddCumulative(intervals, demands, tool.num_available)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return [(req, solver.Value(sv)) for req, sv in start_vars]
    return None
