"""Canonical solution representation and safe symmetry probes.

The canonical sequence is route-level, not schedule-only: each token describes
one delivery or pickup stop at a concrete day, vehicle index, and route
position.  This gives the CVAE/search layer a stable serialization while the
existing ``scheduling`` and ``routing`` modules remain the source of truth for
feasibility and cost.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from instance import Instance
from routing.export import cost_from_routes
from routing.solver import Stop, VehicleRoute
from scheduling.state import build_state, commit_request, validate_schedule


Action = Literal["DELIVER", "PICKUP"]


@dataclass(frozen=True)
class Token:
    action: Action
    request_id: int
    day: int
    vehicle_idx: int
    position_in_route: int


class RepresentationError(ValueError):
    """Raised when a token sequence cannot describe a VeRoLog solution."""


class SymmetryNotApplicable(ValueError):
    """Raised when a proposed symmetry is not valid for this solution."""


class Solution:
    """Namespace for canonical sequence conversion helpers."""

    @staticmethod
    def to_sequence(state: dict, route_set: dict) -> list[Token]:
        """Serialize a synchronized ``(state, route_set)`` into tokens.

        ``state`` is accepted for API symmetry and future consistency checks;
        ordering is determined only from ``route_set``:
        ``(day, canonical vehicle index, position)``.
        """
        _ = state
        tokens: list[Token] = []
        for day in sorted(route_set):
            routes = sorted(
                enumerate(route_set[day]),
                key=lambda item: (item[1].vehicle_id, item[0]),
            )
            for vehicle_idx, (_, route) in enumerate(routes):
                for position, stop in enumerate(route.stops):
                    action: Action = "DELIVER" if stop.action == "delivery" else "PICKUP"
                    tokens.append(Token(action, stop.request_id, day, vehicle_idx, position))
        return tokens

    @staticmethod
    def from_sequence(
        tokens: Iterable[Token],
        instance: Instance,
        *,
        strict: bool = True,
    ) -> tuple[dict, dict]:
        """Reconstruct ``(state, route_set)`` from route tokens.

        The decoder enforces the same schedule feasibility used by the solver:
        a delivery token commits the request with ``commit_request``; a pickup
        token is accepted only if it appears on ``delivery_day + duration``.
        Invalid next tokens are rejected by the mask; in ``strict`` mode this is
        reported as ``RepresentationError``.
        """
        req_by_id = {r.id: r for r in instance.requests}
        tool_by_type = {t.id: t for t in instance.tools}
        ordered = sorted(tokens, key=lambda t: (t.day, t.vehicle_idx, t.position_in_route))

        state = build_state(instance)
        deliveries: dict[int, int] = {}
        pickups: set[int] = set()
        route_tokens: dict[tuple[int, int], list[Token]] = defaultdict(list)

        def reject(message: str) -> None:
            if strict:
                raise RepresentationError(message)

        for token in ordered:
            req = req_by_id.get(token.request_id)
            if req is None:
                reject(f"unknown request id {token.request_id}")
                continue

            if token.action == "DELIVER":
                if token.request_id in deliveries:
                    reject(f"duplicate delivery for request {token.request_id}")
                    continue
                try:
                    commit_request(state, instance, req, token.day)
                except ValueError as exc:
                    reject(str(exc))
                    continue
                deliveries[token.request_id] = token.day
                route_tokens[(token.day, token.vehicle_idx)].append(token)
            elif token.action == "PICKUP":
                delivery_day = deliveries.get(token.request_id)
                if delivery_day is None:
                    reject(f"pickup before delivery for request {token.request_id}")
                    continue
                expected = delivery_day + req.duration
                if token.day != expected:
                    reject(
                        f"pickup for request {token.request_id} on day {token.day}, "
                        f"expected {expected}"
                    )
                    continue
                if token.request_id in pickups:
                    reject(f"duplicate pickup for request {token.request_id}")
                    continue
                pickups.add(token.request_id)
                route_tokens[(token.day, token.vehicle_idx)].append(token)
            else:
                reject(f"unknown token action {token.action!r}")

        if strict:
            expected_ids = {r.id for r in instance.requests}
            if set(deliveries) != expected_ids:
                missing = sorted(expected_ids - set(deliveries))
                raise RepresentationError(f"missing deliveries: {missing}")
            if pickups != expected_ids:
                missing = sorted(expected_ids - pickups)
                raise RepresentationError(f"missing pickups: {missing}")
            if not validate_schedule(state["scheduled"], instance):
                raise RepresentationError("decoded schedule failed validation")

        route_set: dict[int, list[VehicleRoute]] = defaultdict(list)
        for (day, vehicle_idx), group in sorted(route_tokens.items()):
            group.sort(key=lambda t: t.position_in_route)
            stops = [_stop_from_token(t, req_by_id, tool_by_type) for t in group]
            if stops:
                route_set[day].append(
                    VehicleRoute(
                        vehicle_id=vehicle_idx,
                        stops=stops,
                        distance=_route_distance(stops, instance),
                    )
                )

        return state, dict(route_set)


def feasible_delivery_days(state: dict, instance: Instance, request_id: int) -> list[int]:
    """Return the delivery-day mask for one currently unscheduled request."""
    from scheduling.state import is_feasible

    req = next(r for r in instance.requests if r.id == request_id)
    return [
        day
        for day in range(req.earliest, req.latest + 1)
        if is_feasible(state, instance, req, day, day + req.duration)
    ]


def permute_vehicles_within_day(
    state: dict,
    route_set: dict,
    instance: Instance,
    day: int,
) -> tuple[dict, dict]:
    """Reverse the order of vehicles on one day.

    Vehicle labels are not part of the objective, so this should be a genuine
    cost-preserving symmetry whenever the day uses at least two vehicles.
    """
    routes = route_set.get(day, [])
    if len(routes) < 2:
        raise SymmetryNotApplicable(f"day {day} has fewer than two vehicles")
    new_routes = _copy_route_set(route_set)
    new_routes[day] = list(reversed(new_routes[day]))
    return _verified_variant(state, route_set, new_routes, instance)


def permute_within_vehicle(
    state: dict,
    route_set: dict,
    instance: Instance,
    day: int,
    vehicle_idx: int,
) -> tuple[dict, dict]:
    """Reverse a full vehicle route if doing so is feasible and cost-invariant."""
    if not _distance_symmetric(instance):
        raise SymmetryNotApplicable("distance matrix is not symmetric")
    new_routes = _copy_route_set(route_set)
    day_routes = new_routes.get(day, [])
    if vehicle_idx >= len(day_routes):
        raise SymmetryNotApplicable(f"vehicle {vehicle_idx} not present on day {day}")
    route = day_routes[vehicle_idx]
    if len(route.stops) < 2:
        raise SymmetryNotApplicable("route has fewer than two stops")
    route.stops = list(reversed(route.stops))
    route.distance = _route_distance(route.stops, instance)
    if not _route_capacity_feasible(route, instance):
        raise SymmetryNotApplicable("reversed route violates vehicle capacity")
    return _verified_variant(state, route_set, new_routes, instance)


def shift_request_day(
    state: dict,
    route_set: dict,
    instance: Instance,
    request_id: int,
    new_day: int,
) -> tuple[dict, dict]:
    """Shift one request only if exact routed cost is unchanged.

    In VeRoLog this is usually not a symmetry because changing the delivery day
    changes pickup day, daily vehicle loads, tool inventories, and routes.  The
    function still exists as a safe probe: it applies the move and rejects it if
    exact cost changes.
    """
    from routing.solver import solve_routing
    from scheduling.state import build_state

    req_by_id = {r.id: r for r in instance.requests}
    req = req_by_id[request_id]
    if not req.earliest <= new_day <= req.latest:
        raise SymmetryNotApplicable("new day outside delivery window")

    new_state = build_state(instance)
    for entry in state["scheduled"]:
        r = entry["request"]
        day = new_day if r.id == request_id else entry["delivery_day"]
        commit_request(new_state, instance, r, day)

    new_routes = solve_routing(new_state, instance, fast=True)
    return _verified_variant(state, route_set, new_routes, instance)


def swap_equivalent_tools(
    state: dict,
    route_set: dict,
    instance: Instance,
    tool_a: int,
    tool_b: int,
) -> tuple[dict, dict]:
    """Probe the tool-label symmetry.

    Tool labels are embedded in the fixed request data in this codebase, so a
    solution-only swap would make deliveries serve the wrong request tool type.
    It is therefore applicable only in the degenerate case where neither tool is
    used by any request.
    """
    tools = {t.id: t for t in instance.tools}
    a = tools[tool_a]
    b = tools[tool_b]
    if (a.size, a.cost) != (b.size, b.cost):
        raise SymmetryNotApplicable("tools are not equivalent")
    if any(r.machine_type in {tool_a, tool_b} for r in instance.requests):
        raise SymmetryNotApplicable(
            "tool labels are fixed by request.machine_type; no solution-only swap exists"
        )
    return _verified_variant(state, route_set, _copy_route_set(route_set), instance)


def _verified_variant(
    old_state: dict,
    old_routes: dict,
    new_routes: dict,
    instance: Instance,
) -> tuple[dict, dict]:
    before = cost_from_routes(old_routes, instance)["total"]
    after = cost_from_routes(new_routes, instance)["total"]
    if abs(before - after) > 1e-9:
        raise SymmetryNotApplicable(f"cost changed from {before} to {after}")
    new_state, canonical_routes = Solution.from_sequence(
        Solution.to_sequence(old_state, new_routes),
        instance,
        strict=True,
    )
    return new_state, canonical_routes


def _stop_from_token(token: Token, req_by_id: dict, tool_by_type: dict) -> Stop:
    req = req_by_id[token.request_id]
    load = req.num_machines * tool_by_type[req.machine_type].size
    return Stop(
        request_id=req.id,
        action="delivery" if token.action == "DELIVER" else "pickup",
        location_id=req.location_id,
        load=load,
        machine_type=req.machine_type,
    )


def _route_distance(stops: list[Stop], instance: Instance) -> int:
    if not stops:
        return 0
    locs = [instance.depot_id] + [s.location_id for s in stops] + [instance.depot_id]
    return sum(instance.get_distance(locs[i], locs[i + 1]) for i in range(len(locs) - 1))


def _copy_route_set(route_set: dict) -> dict:
    return {
        day: [
            VehicleRoute(route.vehicle_id, list(route.stops), route.distance)
            for route in routes
        ]
        for day, routes in route_set.items()
    }


def _distance_symmetric(instance: Instance) -> bool:
    n = len(instance.distance)
    for i in range(n):
        for j in range(i + 1, n):
            if instance.distance[i][j] != instance.distance[j][i]:
                return False
    return True


def _route_capacity_feasible(route: VehicleRoute, instance: Instance) -> bool:
    req_by_id = {r.id: r for r in instance.requests}
    tool_by_id = {t.id: t for t in instance.tools}
    current = [0] * len(instance.tools)
    node_visits = []
    for stop in route.stops:
        req = req_by_id[stop.request_id]
        idx = req.machine_type - 1
        if stop.action == "delivery":
            current[idx] -= req.num_machines
        else:
            current[idx] += req.num_machines
        node_visits.append(list(current))

    if not node_visits:
        return True
    bring = [0] * len(instance.tools)
    for visit in node_visits:
        bring = [min(a, b) for a, b in zip(bring, visit)]
    for visit in node_visits:
        load = 0
        for idx, value in enumerate(visit):
            tool = tool_by_id[idx + 1]
            load += (value - bring[idx]) * tool.size
        if load > instance.config.capacity:
            return False
    start_load = sum((-bring[idx]) * tool_by_id[idx + 1].size for idx in range(len(bring)))
    return start_load <= instance.config.capacity

