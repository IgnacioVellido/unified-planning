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
# limitations under the License.
#
"""
This module defines the `Bind` class.
A basic `Bind` has a `parameter` and a `fluent`.
"""


import unified_planning as up
from typing import List, Callable, Dict
from unified_planning.model.operators import OperatorKind

class Bind:
    """
    This class represent the bind operator. It has a
    :class:`~unified_planning.model.Fluent` and the `parameter` where its
    value is assigned
    """

    def __init__(
        self,
        parameter: "up.model.Parameter",
        fluent: "up.model.fnode.FNode",
    ):
        self._parameter = parameter
        self._fluent = fluent
        assert (
            fluent.environment == parameter.environment
        ), "Bind fluent and parameter have different environment."
        # print(self)

    def __repr__(self) -> str:
        return f"bind ?{str(self._parameter)} <- {str(self._fluent)}"

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, Bind):
            return (
                self._parameter == oth._parameter
                and self._fluent == oth._fluent
            )
        else:
            return False

    def __hash__(self) -> int:
        return (
            hash(self._parameter)
            + hash(self._fluent)
        )

    @property
    def environment(self) -> "up.environment.Environment":
        """Returns this `Bind's Environment`."""
        return self._fluent.environment

    @property
    def fluent(self) -> "up.model.fnode.FNode":
        """Returns the `Fluent` that is assigned in this `Bind`."""
        return self._fluent

    @property
    def parameter(self) -> "up.model.parameter.Parameter":
        """Returns the `Parameter` that is created by this `Bind`."""
        return self._parameter

    # TODO: Needed?
    @property
    def args(self) -> List["FNode"]:
        """Returns the subexpressions of this expression."""
        return self._fluent.args

    # TODO: Needed?
    @property
    def node_type(self) -> OperatorKind:
        """Returns the `OperatorKind` that defines the semantic of this expression."""
        return OperatorKind.BIND



    # TODO: This shouldn't be here. Maybe Bind should be a subtype of FNode
    def is_bool_constant(self) -> bool:
        """Test whether the expression is a `boolean` constant."""
        return False

    def is_int_constant(self) -> bool:
        """Test whether the expression is an `integer` constant."""
        return False

    def is_real_constant(self) -> bool:
        """Test whether the expression is a `real` constant."""
        return False

    def is_true(self) -> bool:
        """Test whether the expression is the `True` Boolean constant."""
        return False

    def is_false(self) -> bool:
        """Test whether the expression is the `False` Boolean constant."""
        return False

    def is_and(self) -> bool:
        """Test whether the node is the `And` operator."""
        return False

    def is_or(self) -> bool:
        """Test whether the node is the `Or` operator."""
        return False

    def is_not(self) -> bool:
        """Test whether the node is the `Not` operator."""
        return False

    def is_implies(self) -> bool:
        """Test whether the node is the `Implies` operator."""
        return False

    def is_iff(self) -> bool:
        """Test whether the node is the `Iff` operator."""
        return False

    def is_exists(self) -> bool:
        """Test whether the node is the `Exists` operator."""
        return False

    def is_forall(self) -> bool:
        """Test whether the node is the `Forall` operator."""
        return False

    def is_fluent_exp(self) -> bool:
        """Test whether the node is a :class:`~unified_planning.model.Fluent` Expression."""
        return False

    def is_parameter_exp(self) -> bool:
        """Test whether the node is an :func:`action parameter <unified_planning.model.Action.parameters>`."""
        return False

    def is_variable_exp(self) -> bool:
        """Test whether the node is a :class:`~unified_planning.model.Variable` Expression."""
        return False

    def is_object_exp(self) -> bool:
        """Test whether the node is an :class:`~unified_planning.model.Object` Expression."""
        return False

    def is_timing_exp(self) -> bool:
        """Test whether the node is a :class:`~unified_planning.model.Timing` Expression."""
        return False

    def is_plus(self) -> bool:
        """Test whether the node is the `Plus` operator."""
        return False

    def is_minus(self) -> bool:
        """Test whether the node is the `Minus` operator."""
        return False

    def is_times(self) -> bool:
        """Test whether the node is the `Times` operator."""
        return False

    def is_div(self) -> bool:
        """Test whether the node is the `Div` operator."""
        return False

    def is_equals(self) -> bool:
        """Test whether the node is the `Equals` operator."""
        return False

    def is_le(self) -> bool:
        """Test whether the node is the `LE` operator."""
        return False

    def is_lt(self) -> bool:
        """Test whether the node is the `LT` operator."""
        return False

    def is_dot(self) -> bool:
        """Test whether the node is the `DOT` operator."""
        return False