import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from instance import Instance
from routing.export import cost_from_routes, read_solution
from scheduling.state import validate_schedule

from cvae.representation import Solution


def test_solution_sequence_roundtrip_preserves_cost():
    instance = Instance("testInstance.txt")
    state, routes = read_solution("testSolution.txt", instance)

    assert validate_schedule(state["scheduled"], instance)

    tokens = Solution.to_sequence(state, routes)
    decoded_state, decoded_routes = Solution.from_sequence(tokens, instance)

    assert validate_schedule(decoded_state["scheduled"], instance)
    assert cost_from_routes(decoded_routes, instance)["total"] == cost_from_routes(routes, instance)["total"]
