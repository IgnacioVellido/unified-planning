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
        print(self)

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