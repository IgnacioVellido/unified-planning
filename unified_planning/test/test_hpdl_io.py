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

from unified_planning.io.hpdl import HPDLReader, HPDLWriter

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
HPDL_DOMAINS_PATH = os.path.join(FILE_PATH, "pddl/hpdl")


class TestHpdlIO(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        # self.problems = get_example_problems()

    def test_hpdl_reader(self):
        reader = HPDLReader()

        domain_filename = os.path.join(HPDL_DOMAINS_PATH, "zenotravel", "domain.hpdl")
        problem_filename = os.path.join(HPDL_DOMAINS_PATH, "zenotravel", "problem.hpdl")

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

        domain_filename = os.path.join(HPDL_DOMAINS_PATH, "vgdl", "domain.hpdl")
        problem_filename = os.path.join(HPDL_DOMAINS_PATH, "vgdl", "problem.hpdl")
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

    def test_hpdl_writer(self):
        reader = HPDLReader()

        domain_filename = os.path.join(HPDL_DOMAINS_PATH, "miconic", "domain.hpdl")
        problem_filename = os.path.join(HPDL_DOMAINS_PATH, "miconic", "problem.hpdl")
        problem = reader.parse_problem(domain_filename, problem_filename)

        w = HPDLWriter(problem)

        hpdl_domain = w.get_domain()
        hpdl_problem = w.get_problem()

        # print(hpdl_domain)
        # print(hpdl_problem)
        # w.write_domain(os.path.join(HPDL_DOMAINS_PATH, "", "test_domain.hpdl"))
        # w.write_problem(os.path.join(HPDL_DOMAINS_PATH, "", "test_problem.hpdl"))

        expected_domain = """(define (domain prob-domain)
 (:requirements
   :strips
   :typing
   :negative-preconditions
   :universal-preconditions
   :htn-expansion
 )
 (:types
    object__compiled - object
    person floor - object__compiled
 )
 (:predicates
  (type_member_floor ?var - object)
  (type_member_object__compiled ?var - object)
  (type_member_person ?var - object)
  (boarded ?var0 - person)
  (goal_ ?var0 - person)
  (lift_at ?var0 - floor)
  (origin ?var0 - person ?var1 - floor)
  (destination ?var0 - person ?var1 - floor)
 )
 (:task move
  :parameters (?f1 - object ?f2 - object )
  (:method move_method1
   :precondition (and
    (and (type_member_floor ?f1 - object) (type_member_floor ?f2 - object))
   )
   :tasks (
    (move_primitive ?f1 - object ?f2 - object )
   )
  )
 )
 (:task board
  :parameters (?p - object ?f - object )
  (:method board_method1
   :precondition (and
    (and (type_member_person ?p - object) (type_member_floor ?f - object))
   )
   :tasks (
    (board_primitive ?p - object ?f - object )
   )
  )
 )
 (:task debark
  :parameters (?p - object ?f - object )
  (:method debark_method1
   :precondition (and
    (and (type_member_person ?p - object) (type_member_floor ?f - object))
   )
   :tasks (
    (debark_primitive ?p - object ?f - object )
   )
  )
 )
 (:task solve_elevator
  :parameters ()
  (:method solve_elevator_m1_abort_ordering_0
   :precondition (and
    (forall (?p - person)
 (not (goal_ ?p)))
   )
   :tasks (
   )
  )
  (:method solve_elevator_m1_go_ordering_0
   :precondition (and
    (and (type_member_floor ?d - floor) (type_member_floor ?f - floor) (type_member_floor ?o - floor) (type_member_person ?p - person) (goal_ ?p - person) (lift_at ?f - floor) (origin ?p - person ?o - floor) (destination ?p - person ?d - floor))
   )
   :tasks (
    (deliver_person ?p - person ?o - floor ?d - floor )
    (solve_elevator )
   )
  )
 )
 (:task deliver_person
  :parameters (?p - person ?o - floor ?d - floor )
  (:method deliver_person_m2_ordering_0
   :precondition (and
    (and (type_member_floor ?d - floor) (type_member_floor ?f - floor) (type_member_floor ?o - floor) (type_member_person ?p - person) (lift_at ?f - floor))
   )
   :tasks (
    (move ?f - floor ?o - floor )
    (board ?p - person ?o - floor )
    (move ?o - floor ?d - floor )
    (debark ?p - person ?d - floor )
   )
  )
 )
 (:action move_primitive
  :parameters (?f1 - floor ?f2 - floor )
  :precondition (and
   (lift_at ?f1 - floor)
  )
  :effect (and
   (not (lift_at ?f1 - floor))(lift_at ?f2 - floor)
  )
 )
 (:action board_primitive
  :parameters (?p - person ?f - floor )
  :effect (and
   (boarded ?p - person)
  )
 )
 (:action debark_primitive
  :parameters (?p - person ?f - floor )
  :precondition (and
   (and (boarded ?p - person) (goal_ ?p - person))
  )
  :effect (and
   (not (boarded ?p - person))(not (goal_ ?p - person))
  )
 )
)
"""
        expected_problem = """(define (problem prob-problem)
 (:domain prob-domain)
 (:customization
  (= :time-format "%d/%m/%Y %H:%M:%S")
  (= :time-horizon-relative 2500)
  (= :time-start "05/06/2007 08:00:00")
  (= :time-unit :hours)
 )
 (:objects
   p0 - person
   f0 f1 - floor
 )
 (:init
  (lift_at f0)
  (goal_ p0)
  (origin p0 f1)
  (destination p0 f0)
  (type_member_floor f0)
  (type_member_floor f1)
  (type_member_object__compiled f0)
  (type_member_object__compiled f1)
  (type_member_object__compiled p0)
  (type_member_person p0)
 )
 (:tasks-goal
  :tasks (
    (solve_elevator )
  )
 )
)"""

        self.assertEqual(hpdl_domain, expected_domain)
        self.assertEqual(hpdl_problem, expected_problem)