# Copyright 2021 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import tempfile
from typing import cast

import pytest
import unified_planning
from unified_planning.engines import PlanGenerationResultStatus

# from unified_planning.io import HPDLReader, HPDLWriter
from unified_planning.io.hpdl.hpdl_reader import HPDLReader

# from unified_planning.io.hpdl_writer import HPDLWriter
from unified_planning.model.problem_kind import full_numeric_kind
from unified_planning.model.types import _UserType
from unified_planning.shortcuts import *
from unified_planning.test import (
    TestCase,
    main,
    skipIfNoOneshotPlannerForProblemKind,
    skipIfNoOneshotPlannerSatisfiesOptimalityGuarantee,
)
from unified_planning.test.examples import get_example_problems

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PDDL_DOMAINS_PATH = os.path.join(FILE_PATH, "pddl")


class TestHpdlIO(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        # self.problems = get_example_problems()

    def test_hpdl_reader(self):
        reader = HPDLReader()

        domain_filename = os.path.join(PDDL_DOMAINS_PATH, "hpdl", "domain.hpdl")
        problem_filename = os.path.join(PDDL_DOMAINS_PATH, "hpdl", "problem.hpdl")

        problem = reader.parse_problem(domain_filename, problem_filename)

        assert isinstance(problem, up.model.htn.HierarchicalProblem)
        self.assertEqual(
            29, len(problem.fluents)
        )  # 16 functions + 13 predicates (9 derived)
        self.assertEqual(5, len(problem.actions))  # 14 actions + 12 inlines
        self.assertEqual(
            [
                "transport-person",
                "mover-avion",
                "hacer-escala",
                "embarcar-pasajeros",
                "desembarcar-pasajeros",
            ],
            [task.name for task in problem.tasks],
        )
        self.assertEqual(
            [
                "transport-person-Case1",
                "transport-person-Case2",
                "transport-person-Case3",
                "mover-avion-rapido-no-refuel",
                "mover-avion-rapido-refuel",
                "mover-avion-lento-no-refuel",
                "mover-avion-lento-refuel",
                "mover-avion-escala",
                "hacer-escala-rapido-no-refuel",
                "hacer-escala-rapido-refuel",
                "hacer-escala-lento-no-refuel",
                "hacer-escala-lento-refuel",
                "embarcar-pasajeros-Case1",
                "embarcar-pasajeros-Case2",
                "desembarcar-pasajeros-Case1",
                "desembarcar-pasajeros-Case2",
            ],
            [method.name for method in problem.methods],
        )

        for action in problem.actions:
            assert isinstance(action, up.model.action.DurativeAction)
        self.assertEqual(25, len(problem.task_network.subtasks))  # Goal

    def test_hpdl_reader_vgdl(self):
        reader = HPDLReader()

        domain_filename = os.path.join(PDDL_DOMAINS_PATH, "hpdl", "vgdl/domain.hpdl")
        problem_filename = os.path.join(PDDL_DOMAINS_PATH, "hpdl", "vgdl/problem.hpdl")
        problem = reader.parse_problem(domain_filename, problem_filename)

        assert isinstance(problem, up.model.htn.HierarchicalProblem)

        self.assertEqual(24, len(problem.fluents))  # 14 functions + 10 predicates
        self.assertEqual(26, len(problem.actions))  # 14 actions + 12 inlines
        self.assertEqual(
            [
                "Turn",
                "turn_avatar",
                "turn_objects",
                "create-interactions",
                "check-interactions",
            ],
            [task.name for task in problem.tasks],
        )
        self.assertEqual(
            [
                "Turn-finish_game",
                "Turn-turn",
                "turn_avatar-avatar_move_up",
                "turn_avatar-avatar_move_down",
                "turn_avatar-avatar_move_left",
                "turn_avatar-avatar_move_right",
                "turn_avatar-avatar_turn_up",
                "turn_avatar-avatar_turn_down",
                "turn_avatar-avatar_turn_left",
                "turn_avatar-avatar_turn_right",
                "turn_avatar-avatar_nil",
                "turn_objects-turn",
                "create-interactions-create",
                "create-interactions-base_case",
                "check-interactions-avatar_wall_stepback",
                "check-interactions-box_avatar_bounceforward",
                "check-interactions-box_wall_undoall",
                "check-interactions-box_box_undoall",
                "check-interactions-box_hole_killsprite",
                "check-interactions-base_case",
            ],
            [method.name for method in problem.methods],
        )
        self.assertEqual(1, len(problem.method("turn_avatar-avatar_move_up").subtasks))
        self.assertEqual(
            4, len(problem.method("check-interactions-avatar_wall_stepback").subtasks)
        )
        self.assertEqual(1, len(problem.task_network.subtasks))  # Goal
