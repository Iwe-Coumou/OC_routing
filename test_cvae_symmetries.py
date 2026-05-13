import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from instance import Instance
from routing.export import cost_from_routes, read_solution
from scheduling.state import validate_schedule

from cvae.representation import (
    SymmetryNotApplicable,
    permute_vehicles_within_day,
    permute_within_vehicle,
)


def _b3_friend_solution():
    instance = Instance("instances/B3.txt")
    state, routes = read_solution("solutions/external/B3_friend_solution.txt", instance)
    assert validate_schedule(state["scheduled"], instance)
    return instance, state, routes


def test_vehicle_permutation_symmetry_preserves_b3_cost():
    instance, state, routes = _b3_friend_solution()
    day = next(day for day, day_routes in sorted(routes.items()) if len(day_routes) >= 2)

    new_state, new_routes = permute_vehicles_within_day(state, routes, instance, day)

    assert validate_schedule(new_state["scheduled"], instance)
    assert cost_from_routes(new_routes, instance)["total"] == pytest.approx(
        cost_from_routes(routes, instance)["total"],
        abs=1e-9,
    )


def test_route_reversal_probe_preserves_cost_when_applicable():
    instance, state, routes = _b3_friend_solution()

    for day, day_routes in sorted(routes.items()):
        for idx, route in enumerate(day_routes):
            if len(route.stops) < 2:
                continue
            try:
                new_state, new_routes = permute_within_vehicle(state, routes, instance, day, idx)
            except SymmetryNotApplicable:
                continue
            assert validate_schedule(new_state["scheduled"], instance)
            assert cost_from_routes(new_routes, instance)["total"] == pytest.approx(
                cost_from_routes(routes, instance)["total"],
                abs=1e-9,
            )
            return

    pytest.skip("no cost-preserving reversible B3 route found")
