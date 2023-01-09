# Copyright 2022 AIPlan4EU project
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
# limitations under the License.
#
"""
This module defines the Task class.
A Task has a name and a signature that defines the types of its parameters.
"""

import unified_planning as up
from unified_planning.environment import get_env, Environment
from typing import List, OrderedDict, Optional, Union
from unified_planning.model.fnode import FNode
from unified_planning.model.action import Action
from unified_planning.model.timing import Timepoint, TimepointKind
from unified_planning.model.types import Type
from unified_planning.model.expression import Expression
from unified_planning.model.parameter import Parameter

from unified_planning.model.timing import TimeInterval

class Task:
    """Represents an abstract task."""

    def __init__(
        self,
        name: str,
        _parameters: Optional[Union[OrderedDict[str, Type], List[Parameter]]] = None,
        _env: Environment = None,
        **kwargs: Type,
    ):
        self._env = get_env(_env)
        self._name = name
        self._parameters: List[Parameter] = []
        if _parameters is not None:
            assert len(kwargs) == 0
            if isinstance(_parameters, OrderedDict):
                for param_name, param_type in _parameters.items():
                    self._parameters.append(
                        up.model.parameter.Parameter(param_name, param_type, self._env)
                    )
            elif isinstance(_parameters, List):
                self._parameters = _parameters[:]
            else:
                raise NotImplementedError
        else:
            for param_name, param_type in kwargs.items():
                self._parameters.append(
                    up.model.parameter.Parameter(param_name, param_type, self._env)
                )

    def __repr__(self) -> str:
        sign = ""
        if len(self.parameters) > 0:
            sign_items = [f"{p.name}={str(p.type)}" for p in self.parameters]
            sign = f'[{", ".join(sign_items)}]'
        return f"{self.name}{sign}"

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, Task):
            return (
                self._name == oth._name
                and self._parameters == oth._parameters
                and self._env == oth._env
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash(self._name) + sum(map(hash, self._parameters))

    @property
    def name(self) -> str:
        """Returns the task's name."""
        return self._name

    @property
    def parameters(self) -> List[Parameter]:
        """Returns the task's parameters as a list."""
        return self._parameters

    def __call__(self, *args: Expression, ident: Optional[str] = None) -> "Subtask":
        """Returns a subtask with the given parameters."""
        return Subtask(self, *self._env.expression_manager.auto_promote(args))


# global counter to enable the creation of unique identifiers.
_task_id_counter = 0


class Subtask:
    def __init__(
        self,
        _task: Union[Action, Task],
        *args: Expression,
        ident: Optional[str] = None,
        _env: Environment = None,
    ):
        self._env = get_env(_env)
        self._task = _task
        self._ident: str
        if ident is not None:
            self._ident = ident
        else:
            # we have to create an unambiguous identifier as there might otherwise identical tasks
            global _task_id_counter
            _task_id_counter += 1
            self._ident = f"_t{_task_id_counter}"
        self._args = self._env.expression_manager.auto_promote(*args)

        self._duration_const: "up.model.timing.TimeInterval" = None
        self._start_const: "up.model.timing.TimeInterval" = None
        self._end_const: "up.model.timing.TimeInterval" = None
        assert len(self._args) == len(self._task.parameters)

    def __repr__(self):
        s = []
        params = ", ".join([str(a) for a in self._args])
        s.append(f"{self.identifier}: {self._task.name}({params})\n")
        s.append("        time_constraints = [\n")
        if self._start_const is not None:
            s.append(f"          start = {str(self._start_const)}\n")
        if self._end_const is not None:
            s.append(f"          end = {str(self._end_const)}\n")
        if self._duration_const is not None:
            s.append(f"          duration = {str(self._duration_const)}\n")
        s.append("        ]")
        return "".join(s)

    def __eq__(self, other):
        if not isinstance(other, Subtask):
            return False
        return (
            self._env == other._env
            and self._ident == other._ident
            and self._task == other._task
            and self._args == other._args
        )

    def __hash__(self):
        return hash(self._ident) + hash(self._task) + sum(map(hash, self._args))

    @property
    def task(self) -> Union[Task, Action]:
        return self._task

    @property
    def parameters(self) -> List["FNode"]:
        return self._args

    @property
    def identifier(self) -> str:
        """Unique identifier of the subtask in its task network."""
        return self._ident

    @property
    def start(self) -> Timepoint:
        """Timepoint representing the task's start time."""
        return Timepoint(TimepointKind.START, container=self.identifier)

    @property
    def end(self) -> Timepoint:
        """Timepoint representing the task's end time."""
        return Timepoint(TimepointKind.END, container=self.identifier)

    def set_start_constraint(
        self,
        start_const: "up.model.timing.TimeInterval",
        less_than: bool
    ):
        """
        Sets a `start time constraint` for this `action`.

        :param start_const: The new `start constraint` of this `action`.
        :param less_than: If new constraint is the result of `<` or `<=`.
        """
        # TODO: Make any needed verification
        if self._start_const is not None:
            self._start_const = self._join_time_constraints(
                self._start_const,
                start_const,
                less_than
            )
        else:
            self._start_const = start_const

    def set_end_constraint(
        self,
        end_const: "up.model.timing.TimeInterval",
        less_than: bool
    ):
        """
        Sets a `end time constraint` for this `action`.

        :param end_const: The new `end constraint` of this `action`.
        :param less_than: If new constraint is the result of `<` or `<=`.
        """
        # TODO: Make any needed verification
        if self._end_const is not None:
            self._end_const = self._join_time_constraints(
                self._end_const,
                end_const,
                less_than
            )
        else:
            self._end_const = end_const

    def set_duration_constraint(
        self,
        duration_const: "up.model.timing.TimeInterval", # TODO: Not sure about this type
        less_than: bool = False
    ):
        """
        Sets a `duration time constraint` for this `action`.

        :param duration_const: The new `duration constraint` of this `action`.
        :param less_than: If new constraint is the result of `<` or `<=`.
        """
        # TODO: Make any needed verification
        if self._duration_const is not None:
            self._duration_const = self._join_time_constraints(
                self._duration_const,
                duration_const,
                less_than
            )
        else:
            self._duration_const = duration_const


    def _join_time_constraints(self, t_old, t_new, less_than):
        """
        Assumes neither t_old or t_new is an assignment constraint
        (e.g. (= ?start 2))

        :param t_old: Constraint already stored.
        :param t_new: New time constraint.
        :param less_than: If new constraint is the result of `<` or `<=`.
        """
        if less_than: # We assume t_old is the opposite (< if >, > if <)
            lower = t_old.lower
            is_left_open = t_old._is_left_open

            upper = t_new.upper
            is_right_open = t_new._is_right_open
        else:
            lower = t_new.lower
            is_left_open = t_new._is_left_open

            upper = t_old.upper
            is_right_open = t_old._is_right_open

        # TODO: Shouldn't it be DurationInterval if changing duration?
        # It only "accepts" FNode though
        return TimeInterval(
            lower,
            upper,
            is_left_open,
            is_right_open
        )