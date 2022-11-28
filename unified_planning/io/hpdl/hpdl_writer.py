from fractions import Fraction
import sys
import re

from decimal import Decimal, localcontext
from warnings import warn

import unified_planning as up
import unified_planning.environment
import unified_planning.model.walkers as walkers
from unified_planning.model import (
    InstantaneousAction,
    DurativeAction,
    Fluent,
    Parameter,
    Problem,
    Object,
    FNode,
)
from unified_planning.model.htn import(
    Task,
    TaskNetwork,
    Method,
    HierarchicalProblem
)
from unified_planning.exceptions import (
    UPTypeError,
    UPProblemDefinitionError,
    UPException,
)
from unified_planning.model.types import _UserType
from typing import Callable, Dict, IO, List, Optional, Set, Union, cast
from io import StringIO
from functools import reduce

PDDL_KEYWORDS = {
    "define",
    "domain",
    "requirements",
    "types",
    "constants",
    "atomic",
    "predicates",
    "problem",
    "atomic",
    "constraints",
    "either",
    "number",
    "action",
    "parameters",
    "precondition",
    "effect",
    "and",
    "forall",
    "preference",
    "or",
    "not",
    "imply",
    "exists",
    "scale-up",
    "scale-down",
    "increase",
    "decrease",
    "durative-action",
    "duration",
    "condition",
    "at",
    "over",
    "start",
    "end",
    "all",
    "derived",
    "objects",
    "init",
    "goal",
    "when",
    "decrease",
    "always",
    "sometime",
    "within",
    "at-most-once",
    "sometime-after",
    "sometime-before",
    "always-within",
    "hold-during",
    "hold-after",
    "metric",
    "minimize",
    "maximize",
    "total-time",
    "is-violated",
    "strips",
    "negative-preconditions",
    "typing",
    "disjunctive-preconditions",
    "equality",
    "existential-preconditions",
    "universal-preconditions",
    "quantified-preconditions",
    "conditional-effects",
    "fluents",
    "adl",
    "durative-actions",
    "derived-predicates",
    "timed-initial-literals",
    "preferences",
    "htn-expansion",
    "metatags",
    # "derived-predicates",
}

# The following map is used to mangle the invalid names by their class.
INITIAL_LETTER: Dict[type, str] = {
    InstantaneousAction: "a",
    DurativeAction: "a",
    Fluent: "f",
    Parameter: "p",
    Problem: "p",
    Object: "o",
    Task: "t",
    TaskNetwork: "t",
    Method: "m",
}


class ObjectsExtractor(walkers.DagWalker):
    """Returns the object instances appearing in the expression."""

    def __init__(self):
        walkers.dag.DagWalker.__init__(self)

    def get(self, expression: "up.model.FNode") -> Dict[_UserType, Set[Object]]:
        """Returns all the free vars of the given expression."""
        return self.walk(expression)

    def walk_object_exp(
        self, expression: "up.model.FNode", args: List[Dict[_UserType, Set[Object]]]
    ) -> Dict[_UserType, Set[Object]]:
        res: Dict[_UserType, Set[Object]] = {}
        for a in args:
            _update_domain_objects(res, a)
        obj = expression.object()
        assert obj.type.is_user_type()
        res.setdefault(cast(_UserType, obj.type), set()).add(obj)
        return res

    @walkers.handles(
        set(up.model.OperatorKind).difference((up.model.OperatorKind.OBJECT_EXP,))
    )
    def walk_all_types(
        self, expression: "up.model.FNode", args: List[Dict[_UserType, Set[Object]]]
    ) -> Dict[_UserType, Set[Object]]:
        res: Dict[_UserType, Set[Object]] = {}
        for a in args:
            _update_domain_objects(res, a)
        return res


class ConverterToPDDLString(walkers.DagWalker):
    """Expression converter to a PDDL string."""

    DECIMAL_PRECISION = 10  # Number of decimal places to print real constants

    def __init__(
        self,
        env: "up.environment.Environment",
        get_mangled_name: Callable[
            [
                Union[
                    "up.model.Type",
                    "up.model.Action",
                    "up.model.Fluent",
                    "up.model.Object",
                ]
            ],
            str,
        ],
    ):
        walkers.DagWalker.__init__(self)
        self.get_mangled_name = get_mangled_name
        self.simplifier = env.simplifier

    def convert(self, expression):
        """Converts the given expression to a PDDL string."""
        return self.walk(self.simplifier.simplify(expression))

    def walk_exists(self, expression, args):
        assert len(args) == 1
        vars_string_list = [
            f"{self.get_mangled_name(v)} - {self.get_mangled_name(v.type)}"
            for v in expression.variables()
        ]
        return f'(exists ({" ".join(vars_string_list)})\n {args[0]})'

    def walk_forall(self, expression, args):
        assert len(args) == 1
        vars_string_list = [
            f"{self.get_mangled_name(v)} - {self.get_mangled_name(v.type)}"
            for v in expression.variables()
        ]
        return f'(forall ({" ".join(vars_string_list)})\n {args[0]})'

    def walk_variable_exp(self, expression, args):
        assert len(args) == 0
        return f"{self.get_mangled_name(expression.variable())}"

    def walk_and(self, expression, args):
        assert len(args) > 1
        return f'(and {" ".join(args)})'

    def walk_or(self, expression, args):
        assert len(args) > 1
        return f'(or {" ".join(args)})'

    def walk_not(self, expression, args):
        assert len(args) == 1
        return f"(not {args[0]})"

    def walk_implies(self, expression, args):
        assert len(args) == 2
        return f"(imply {args[0]} {args[1]})"

    def walk_iff(self, expression, args):
        assert len(args) == 2
        return f"(and (imply {args[0]} {args[1]}) (imply {args[1]} {args[0]}) )"

    def walk_fluent_exp(self, expression, args):
        fluent = expression.fluent()
        return f'({self.get_mangled_name(fluent)}{" " if len(args) > 0 else ""}{" ".join(args)})'

    def walk_param_exp(self, expression, args):
        assert len(args) == 0
        p = expression.parameter()
        return f"{self.get_mangled_name(p)}"

    def walk_object_exp(self, expression, args):
        assert len(args) == 0
        o = expression.object()
        return f"{self.get_mangled_name(o)}"

    def walk_bool_constant(self, expression, args):
        # raise up.exceptions.UPUnreachableCodeError
        if not expression.constant_value(): # = False
            return "(< 1 0)"
        else:
            return "(> 1 0)"
        # TODO: ASK WHAT TO DO WITH THIS
        # This fails in VGDL, when we force backtracking, because it simplifies
        # to false and here raises and exception
        # e.g. (not (= (coordinate_y ?x) (coordinate_y ?y))) and (= (coordinate_y ?x) (coordinate_y ?y))
        # 
        # At the moment: Because we can't print True/False, printing predicate
        # that evaluates to that

    def walk_real_constant(self, expression, args):
        assert len(args) == 0
        frac = expression.constant_value()

        with localcontext() as ctx:
            ctx.prec = self.DECIMAL_PRECISION
            dec = frac.numerator / Decimal(frac.denominator, ctx)

            if Fraction(dec) != frac:
                warn(
                    "The PDDL printer cannot exactly represent the real constant '%s'"
                    % frac
                )
            return str(dec)

    def walk_int_constant(self, expression, args):
        assert len(args) == 0
        return str(expression.constant_value())

    def walk_plus(self, expression, args):
        assert len(args) > 1
        return reduce(lambda x, y: f"(+ {y} {x})", args)

    def walk_minus(self, expression, args):
        assert len(args) == 2
        return f"(- {args[0]} {args[1]})"

    def walk_times(self, expression, args):
        assert len(args) > 1
        return reduce(lambda x, y: f"(* {y} {x})", args)

    def walk_div(self, expression, args):
        assert len(args) == 2
        return f"(/ {args[0]} {args[1]})"

    def walk_le(self, expression, args):
        assert len(args) == 2
        return f"(<= {args[0]} {args[1]})"

    def walk_lt(self, expression, args):
        assert len(args) == 2
        return f"(< {args[0]} {args[1]})"

    def walk_equals(self, expression, args):
        assert len(args) == 2
        return f"(= {args[0]} {args[1]})"


class HPDLWriter:
    """This class can be used to write a :class:`~unified_planning.model.Problem` in `PDDL`."""

    def __init__(self, problem: "up.model.Problem", needs_requirements: bool = True):
        if not isinstance(problem, HierarchicalProblem):
            raise UPProblemDefinitionError(
                "The given problem is not a HierarchicalProblem, use PDDLWriter intead."
            )
        self.problem = problem
        self.problem_kind = self.problem.kind
        self.needs_requirements = needs_requirements
        # otn represents the old to new renamings
        self.otn_renamings: Dict[
            Union[
                "up.model.Type",
                "up.model.Action",
                "up.model.Fluent",
                "up.model.Object",
                "up.model.Parameter",
                "up.model.Variable",
            ],
            str,
        ] = {}
        # nto represents the new to old renamings
        self.nto_renamings: Dict[
            str,
            Union[
                "up.model.Type",
                "up.model.Action",
                "up.model.Fluent",
                "up.model.Object",
                "up.model.Parameter",
                "up.model.Variable",
            ],
        ] = {}
        # those 2 maps are "symmetrical", meaning that "(otn[k] == v) implies (nto[v] == k)"
        self.domain_objects: Optional[Dict[_UserType, Set[Object]]] = None

    def _write_domain(self, out: IO[str]):
        if self.problem_kind.has_intermediate_conditions_and_effects():
            raise UPProblemDefinitionError(
                "PDDL2.1 does not support ICE.\nICE are Intermediate Conditions and Effects therefore when an Effect (or Condition) are not at StartTIming(0) or EndTIming(0)."
            )
        if self.problem_kind.has_timed_goals():
            raise UPProblemDefinitionError(
                "PDDL2.1 does not support timed effects or timed goals."
            )
        obe = ObjectsExtractor()
        out.write("(define ")
        if self.problem.name is None:
            name = "hpdl"
        else:
            name = _get_pddl_name(self.problem)
        out.write(f"(domain {name}-domain)\n")

        # TODO: Write customization

        if self.needs_requirements:
            self._write_requirements(out)

        self._write_types(out)

        if self.domain_objects is None:
            # This method populates the self._domain_objects map
            self._populate_domain_objects(obe)
        assert self.domain_objects is not None

        if len(self.domain_objects) > 0:
            self._write_constants(out)

        self._write_fluents(out)
        self._write_tasks(out)
        self._write_actions(out)
        out.write(")\n")

    def _write_requirements(self, out: IO[str]):
        out.write(" (:requirements :strips")
        if self.problem_kind.has_flat_typing():
            out.write(" :typing")
        if self.problem_kind.has_negative_conditions():
            out.write(" :negative-preconditions")
        if self.problem_kind.has_disjunctive_conditions():
            out.write(" :disjunctive-preconditions")
        if self.problem_kind.has_equality():
            out.write(" :equality")
        if (
            self.problem_kind.has_continuous_numbers()
            or self.problem_kind.has_discrete_numbers()
        ):
            out.write(" :numeric-fluents")
        if self.problem_kind.has_conditional_effects():
            out.write(" :conditional-effects")
        if self.problem_kind.has_existential_conditions():
            out.write(" :existential-preconditions")
        if self.problem_kind.has_universal_conditions():
            out.write(" :universal-preconditions")
        if (
            self.problem_kind.has_continuous_time()
            or self.problem_kind.has_discrete_time()
        ):
            out.write(" :durative-actions")
        if self.problem_kind.has_duration_inequalities():
            out.write(" :duration-inequalities")
        if (
            self.problem_kind.has_actions_cost()
            or self.problem_kind.has_plan_length()
        ):
            out.write(" :action-costs")
        # TODO: Check if metatasks (modify problem_kind.py)
        # TODO: Check if python-fluents (wait for discussion)
        # if self.problem_kind.has_hierarchical() # This is checked on init
        out.write(" :htn-expansion")
        out.write(")\n")

    def _write_types(self, out: IO[str]):
        if self.problem_kind.has_hierarchical_typing():
            user_types_hierarchy = self.problem.user_types_hierarchy
            out.write(f" (:types\n")
            stack: List["unified_planning.model.Type"] = (
                user_types_hierarchy[None] if None in user_types_hierarchy else []
            )
            out.write(
                f'    {" ".join(self._get_mangled_name(t) for t in stack)} - object\n'
            )
            while stack:
                current_type = stack.pop()
                direct_sons: List["unified_planning.model.Type"] = user_types_hierarchy[
                    current_type
                ]
                if direct_sons:
                    stack.extend(direct_sons)
                    out.write(
                        f'    {" ".join([self._get_mangled_name(t) for t in direct_sons])} - {self._get_mangled_name(current_type)}\n'
                    )
            out.write(" )\n")
        else:
            out.write(
                f' (:types {" ".join([self._get_mangled_name(t) for t in self.problem.user_types])})\n'
                if len(self.problem.user_types) > 0
                else ""
            )

    def _write_constants(self, out: IO[str]):
        out.write(" (:constants")
        for ut, os in self.domain_objects.items():
            if len(os) > 0:
                out.write(
                    f'\n   {" ".join([self._get_mangled_name(o) for o in os])} - {self._get_mangled_name(ut)}'
                )
        out.write("\n )\n")

    def _write_fluents(self, out: IO[str]):
        predicates = []
        functions = []
        for f in self.problem.fluents:
            if f.type.is_bool_type():
                params = []
                i = 0
                for param in f.signature:
                    if param.type.is_user_type():
                        params.append(
                            f" {self._get_mangled_name(param)} - {self._get_mangled_name(param.type)}"
                        )
                        i += 1
                    else:
                        raise UPTypeError("PDDL supports only user type parameters")
                predicates.append(f'({self._get_mangled_name(f)}{"".join(params)})')
            elif f.type.is_int_type() or f.type.is_real_type():
                params = []
                i = 0
                for param in f.signature:
                    if param.type.is_user_type():
                        params.append(
                            f" {self._get_mangled_name(param)} - {self._get_mangled_name(param.type)}"
                        )
                        i += 1
                    else:
                        raise UPTypeError("PDDL supports only user type parameters")
                functions.append(f'({self._get_mangled_name(f)}{"".join(params)})')
            else:
                raise UPTypeError("PDDL supports only boolean and numerical fluents")
        if self.problem.kind.has_actions_cost() or self.problem.kind.has_plan_length():
            functions.append("(total-cost)")

        predicates_str = "\n  ".join(predicates)
        functions_str = "\n  ".join(functions)
        out.write(
            f' (:predicates {predicates_str}\n )\n' if len(predicates) > 0 else ""
        )
        out.write(
            f' (:functions {functions_str}\n )\n' if len(functions) > 0 else ""
        )

    def _write_parameters(self, out: IO[str], parameters):
        for ap in parameters:
            if ap.type.is_user_type():
                out.write(
                    f"{self._get_mangled_name(ap)} - {self._get_mangled_name(ap.type)} "
                )
            else:
                raise UPTypeError("PDDL supports only user type parameters")


    # TODO: Refactor
    def _get_subtasks_str(self, network: Union[TaskNetwork, Method], get_types: bool = False):
        """Write subtasks ordered in HPDL style ([ for parallelism)"""
        def get_time_constraint(s):
            def const_str(op, open, var, const):
                if isinstance(const, FNode):
                    converter = ConverterToPDDLString(self.problem.env, self._get_mangled_name)
                    const = converter.convert(const)

                return f'({op}{"" if open else "="} ?{var} {const})'


            # TODO: class Subtask properties for constraints?
            res = ""

            vars = ["start", "end", "dur"]
            constraints = [s._start_const, s._end_const, s._duration_const]
            for var, constraint in zip(vars, constraints):
                if constraint is not None:
                    lower_delay = constraint.lower.delay
                    upper_delay = constraint.upper.delay

                    if lower_delay == upper_delay:
                        op = '='
                        const = lower_delay
                        open = True
                        res += const_str(op, open, var, const) + " "
                    else:
                        if lower_delay != 0:
                            op = '>'
                            const = lower_delay
                            open = constraint.is_left_open()
                            res += const_str(op, open, var, const) + " "
                        if upper_delay != 0:
                            op = '<'
                            const = upper_delay
                            open = constraint.is_right_open()
                            res += const_str(op, open, var, const) + " "

            return f'\n     (and {res})\n' if res else ""

        # TODO: Give property to const or something because this is not nice
        def get_tags_const(const):
            """Get str names of both tags in ordering constraint"""
            return (const.arg(0).timing().timepoint.container,
                    const.arg(1).timing().timepoint.container)

        def subtask_to_str(s, get_types):
            if get_types: # For domain subtasks
                res = f'    {self._subtasks_to_str(s.task)}'
            else: # For problem task-goal
                res = f'    ({self._get_mangled_name(s.task)} {" ".join([str(p) for p in s.parameters])})'    
            
            # Write time-constraints
            time_const_str = get_time_constraint(s)
            if time_const_str:
                res = f'    ({time_const_str} {res}' + "\n    )"

            return res


        subtask_len = len(network.subtasks)
        if subtask_len == 0:
            return ""

        # Dict subtask_name -> order in task network
        subtask_map = {}
        for i, s in enumerate(network.subtasks):
            subtask_map[s.identifier] = i

        # Create matrix of restriction orders
        matrix = []
        for _ in range(subtask_len):
            matrix.append([False]*subtask_len) # Empty row, all False

        # Set True if constraint
        constraints = network.constraints
        for c in constraints:
            t1, t2 = get_tags_const(c)
            matrix[subtask_map[t1]][subtask_map[t2]] = True

        # Create groups of parallel tasks based on matrix
        subtasks = network.subtasks
        groups = [[subtasks[0]]]
        group_idx = 0
        for row in range(1,subtask_len):
            s = subtasks[row]

            if matrix[row] == matrix[row-1]: # Both parallel, add to existing group
                # Same rows means both have the same constraints, and are parallel
                groups[group_idx].append(s)
            else: # Both sequential, add to new group
                group_idx += 1
                groups.append([s])

        # Print groups
        subtasks_str = ""
        for g in groups:
            if len(g) == 0:
                pass
            elif len(g) > 1: # Parallels
                subtasks_str += "    [\n "
                subtasks_str += "\n ".join([subtask_to_str(s, get_types) for s in g])
                subtasks_str += "\n    ]\n"
            else: # Sequential
                s = g[0]
                subtasks_str += subtask_to_str(s, get_types) + "\n"

        return subtasks_str

    def _get_preconditions_effects_str(self, action):
        """For actions and inlines only"""
        converter = ConverterToPDDLString(self.problem.env, self._get_mangled_name)
            
        precondition_str = "\n  ".join([converter.convert(p) for p in action.preconditions])
        
        effect_str = ""
        for e in action.effects:
            if e.is_conditional():
                effect_str += f"(when {converter.convert(e.condition)}"
            if e.value.is_true():
                effect_str += (f"{converter.convert(e.fluent)}")
            elif e.value.is_false():
                effect_str += (f"(not {converter.convert(e.fluent)})")
            elif e.is_increase():
                effect_str += (
                    f"(increase {converter.convert(e.fluent)} {converter.convert(e.value)})"
                )
            elif e.is_decrease():
                effect_str += (
                    f"(decrease {converter.convert(e.fluent)} {converter.convert(e.value)})"
                )
            else:
                effect_str += (
                    f"(assign {converter.convert(e.fluent)} {converter.convert(e.value)})"
                )
            if e.is_conditional():
                effect_str += (f")")

        return precondition_str, effect_str

    def _subtasks_to_str(self, subtask):
        s = f"({self._get_mangled_name(subtask)} "
        if "inline" in subtask.name:
            # TODO: Clean. Refactor, similar code to _write_actions and _write_tasks
            pre_str, eff_str = self._get_preconditions_effects_str(subtask)
            return f"(:inline (and {pre_str}) (and {eff_str}))"
        else:
            for ap in subtask.parameters: # Type needed for variables not defined in :parameters
                s += f"{self._get_mangled_name(ap)} - {self._get_mangled_name(ap.type)} "
            return s+")"

    # TODO: Put proper indentation
    # TODO: Time constraints
    # TODO: What happens with variables not defined on :parameters and used in
    # multiple places??
    def _write_tasks(self, out:IO[str]):
        converter = ConverterToPDDLString(self.problem.env, self._get_mangled_name)

        # Get methods name of each task
        methods = dict((t.name, []) for t in self.problem.tasks)
        for m in self.problem.methods:
            methods[m.task.name].append(m.name)

        # Print tasks
        for t in self.problem.tasks:
            out.write(f" (:task {self._get_mangled_name(t)}")
            out.write(f"\n  :parameters (")
            self._write_parameters(out, t.parameters)
            out.write(")")

            # Print methods
            for m_name in methods[t.name]:
                m = self.problem.method(m_name)
                out.write(f"\n  (:method {self._get_mangled_name(m)}")
                # out.write(f"\n  :parameters (")
                
                # TODO: Check if SIADEX needs precondition tag even if empty
                # TODO: All methods contains only one and precondition, split
                # TODO: Will likely need to print variable type too
                if len(m.preconditions) > 0:
                    precondition_str = "\n  ".join([converter.convert(p) for p in m.preconditions])
                    out.write(
                        f'\n   :precondition (and\n    {precondition_str}\n    )'
                    )

                # Subtasks
                out.write(
                    f'\n   :tasks (\n{self._get_subtasks_str(m, get_types=True)}   )'
                )

                out.write("\n  )")

            out.write("\n )\n")


    def _write_actions(self, out: IO[str]):
        obe = ObjectsExtractor()
        converter = ConverterToPDDLString(self.problem.env, self._get_mangled_name)
        costs = {}
        metrics = self.problem.quality_metrics
        if len(metrics) == 1:
            metric = metrics[0]
            if isinstance(metric, up.model.metrics.MinimizeActionCosts):
                for a in self.problem.actions:
                    cost_exp = metric.get_action_cost(a)
                    costs[a] = cost_exp
                    if cost_exp is not None:
                        _update_domain_objects(self.domain_objects, obe.get(cost_exp))
            elif isinstance(metric, up.model.metrics.MinimizeSequentialPlanLength):
                for a in self.problem.actions:
                    costs[a] = self.problem.env.expression_manager.Int(1)
        elif len(metrics) > 1:
            raise up.exceptions.UPUnsupportedProblemTypeError(
                "Only one metric is supported!"
            )
            
        for a in self.problem.actions:
            # TODO: Improve how inline is checked
            # Ignore inline. Already written in _write_method
            if isinstance(a, up.model.InstantaneousAction):
                if not "inline" in self._get_mangled_name(a):
                    out.write(f" (:action {self._get_mangled_name(a)}")
                    out.write(f"\n  :parameters (")
                    self._write_parameters(out, a.parameters)
                    out.write(")")

                    pre_str, eff_str = self._get_preconditions_effects_str(a)
                    if len(a.preconditions) > 0:
                        out.write(
                            f'\n  :precondition (and\n   {pre_str}\n  )'
                        )
                    if len(a.effects) > 0:
                        out.write(
                            f"\n  :effect (and\n   {eff_str}\n  )"
                        )
                    out.write("\n )\n")
            elif isinstance(a, DurativeAction):
                out.write(f" (:durative-action {self._get_mangled_name(a)}")
                out.write(f"\n  :parameters (")
                self._write_parameters(out, a.parameters)
                out.write(")")
                l, r = a.duration.lower, a.duration.upper
                if l == r:
                    out.write(f"\n  :duration (= ?duration {converter.convert(l)})")
                else:
                    out.write(f"\n  :duration (and ")
                    if a.duration.is_left_open():
                        out.write(f"(> ?duration {converter.convert(l)})")
                    else:
                        out.write(f"(>= ?duration {converter.convert(l)})")
                    if a.duration.is_right_open():
                        out.write(f"(< ?duration {converter.convert(r)})")
                    else:
                        out.write(f"(<= ?duration {converter.convert(r)})")
                    out.write(")")
                if len(a.conditions) > 0:
                    out.write(f"\n  :condition (and")
                    for interval, cl in a.conditions.items():
                        for c in cl:
                            out.write("\n   ")
                            if interval.lower == interval.upper:
                                if interval.lower.is_from_start():
                                    out.write(f"(at start {converter.convert(c)})")
                                else:
                                    out.write(f"(at end {converter.convert(c)})")
                            else:
                                if not interval.is_left_open():
                                    out.write(f"(at start {converter.convert(c)})")
                                out.write(f"(over all {converter.convert(c)})")
                                if not interval.is_right_open():
                                    out.write(f"(at end {converter.convert(c)})")
                    out.write("\n  )")
                if len(a.effects) > 0:
                    out.write("\n  :effect (and")
                    for t, el in a.effects.items():
                        for e in el:
                            out.write("\n   ")
                            if t.is_from_start():
                                out.write(f"(at start ")
                            else:
                                out.write(f"(at end ")
                            if e.is_conditional():
                                out.write(f"(when {converter.convert(e.condition)}")
                            if e.value.is_true():
                                out.write(f"{converter.convert(e.fluent)}")
                            elif e.value.is_false():
                                out.write(f"(not {converter.convert(e.fluent)})")
                            elif e.is_increase():
                                out.write(
                                    f"(increase {converter.convert(e.fluent)} {converter.convert(e.value)})"
                                )
                            elif e.is_decrease():
                                out.write(
                                    f"(decrease {converter.convert(e.fluent)} {converter.convert(e.value)})"
                                )
                            else:
                                out.write(
                                    f"(assign {converter.convert(e.fluent)} {converter.convert(e.value)})"
                                )
                            if e.is_conditional():
                                out.write(f")")
                            out.write(")")
                    if a in costs:
                        out.write(
                            f" (at end (increase (total-cost) {converter.convert(costs[a])}))"
                        )
                    out.write("\n  )")
                
                out.write("\n )\n")
            else:
                raise NotImplementedError

    def _write_objects(self, out: IO[str]):
        out.write(" (:objects")
        for t in self.problem.user_types:
            constants_of_this_type = self.domain_objects.get(
                cast(_UserType, t), None
            )
            if constants_of_this_type is None:
                objects = [o for o in self.problem.all_objects if o.type == t]
            else:
                objects = [
                    o
                    for o in self.problem.all_objects
                    if o.type == t and o not in constants_of_this_type
                ]
            if len(objects) > 0:
                out.write(
                    f'\n   {" ".join([self._get_mangled_name(o) for o in objects])} - {self._get_mangled_name(t)}'
                )
        out.write("\n )\n")

    def _write_init(self, out: IO[str]):
        converter = ConverterToPDDLString(self.problem.env, self._get_mangled_name)

        out.write(" (:init")
        # FIXME: Why are we calling initial_values? It's extremely slow
        # for f, v in self.problem.initial_values.items():
        # Can't we use explicit_initial_values?
        for f, v in self.problem.explicit_initial_values.items():
            # TODO: (at start), (between)
            if v.is_true():
                out.write(f"\n  {converter.convert(f)}")
            elif v.is_false():
                pass
            else:
                out.write(f"\n  (= {converter.convert(f)} {converter.convert(v)})")
        if self.problem.kind.has_actions_cost():
            out.write(f"\n  (= (total-cost) 0)")
        out.write("\n )\n")

    # FIXME
    def _write_problem(self, out: IO[str]):
        if self.problem.name is None:
            name = "pddl"
        else:
            name = _get_pddl_name(self.problem)
        out.write(f"(define (problem {name}-problem)\n")
        out.write(f" (:domain {name}-domain)\n")

        # TODO: :customization

        if self.domain_objects is None:
            # This method populates the self._domain_objects map
            self._populate_domain_objects(ObjectsExtractor())
        assert self.domain_objects is not None
        if len(self.problem.user_types) > 0:
            self._write_objects(out)

        self._write_init(out) # FIXME
        # print(self.problem.initial_values.items())

        # Print task-goal
        out.write(
            f' (:tasks-goal\n  :tasks (\n{self._get_subtasks_str(self.problem.task_network)}  )\n )\n'
        )
    
        metrics = self.problem.quality_metrics
        if len(metrics) > 0:
            raise up.exceptions.UPUnsupportedProblemTypeError(
                "HPDL does not support metrics!"
            )
        out.write(")")

    def print_domain(self):
        """Prints to std output the `PDDL` domain."""
        self._write_domain(sys.stdout)

    def print_problem(self):
        """Prints to std output the `PDDL` problem."""
        self._write_problem(sys.stdout)

    def get_domain(self) -> str:
        """Returns the `PDDL` domain."""
        out = StringIO()
        self._write_domain(out)
        return out.getvalue()

    def get_problem(self) -> str:
        """Returns the `PDDL` problem."""
        out = StringIO()
        self._write_problem(out)
        return out.getvalue()

    def write_domain(self, filename: str):
        """Dumps to file the `PDDL` domain."""
        with open(filename, "w") as f:
            self._write_domain(f)

    def write_problem(self, filename: str):
        """Dumps to file the `PDDL` problem."""
        with open(filename, "w") as f:
            self._write_problem(f)

    def _get_mangled_name(
        self,
        item: Union[
            "up.model.Type",
            "up.model.Action",
            "up.model.Fluent",
            "up.model.Object",
            "up.model.Parameter",
            "up.model.Variable",
        ],
    ) -> str:
        """This function returns a valid and unique PDDL name."""

        # If we already encountered this item, return it
        if item in self.otn_renamings:
            return self.otn_renamings[item]

        if isinstance(item, up.model.Type):
            assert item.is_user_type()
            original_name = cast(_UserType, item).name
            tmp_name = _get_pddl_name(item)
            # If the problem is hierarchical and the name is object, we want to change it
            if self.problem_kind.has_hierarchical_typing() and tmp_name == "object":
                tmp_name = f"{tmp_name}_"
        else:
            original_name = item.name
            tmp_name = _get_pddl_name(item)
        # if the pddl valid name is the same of the original one and it does not create conflicts,
        # it can be returned
        if tmp_name == original_name and tmp_name not in self.nto_renamings:
            new_name = tmp_name
        else:
            count = 0
            new_name = tmp_name
            while self.problem.has_name(new_name) or new_name in self.nto_renamings:
                new_name = f"{tmp_name}_{count}"
                count += 1
        assert (
            new_name not in self.nto_renamings
            and new_name not in self.otn_renamings.values()
        )
        self.otn_renamings[item] = new_name
        self.nto_renamings[new_name] = item
        return new_name

    def get_item_named(
        self, name: str
    ) -> Union[
        "up.model.Type",
        "up.model.Action",
        "up.model.Fluent",
        "up.model.Object",
        "up.model.Parameter",
        "up.model.Variable",
    ]:
        """
        Since `PDDL` has a stricter set of possible naming compared to the `unified_planning`, when writing
        a :class:`~unified_planning.model.Problem` it is possible that some things must be renamed. This is why the `PDDLWriter`
        offers this method, that takes a `PDDL` name and returns the original `unified_planning` data structure that corresponds
        to the `PDDL` entity with the given name.

        This method takes a name used in the `PDDL` domain or `PDDL` problem generated by this `PDDLWriter` and returns the original
        item in the `unified_planning` `Problem`.

        :param name: The name used in the generated `PDDL`.
        :return: The `unified_planning` model entity corresponding to the given name.
        """
        try:
            return self.nto_renamings[name]
        except KeyError:
            raise UPException(f"The name {name} does not correspond to any item.")

    def get_pddl_name(
        self,
        item: Union[
            "up.model.Type",
            "up.model.Action",
            "up.model.Fluent",
            "up.model.Object",
            "up.model.Parameter",
            "up.model.Variable",
        ],
    ) -> str:
        """
        This method takes an item in the :class:`~unified_planning.model.Problem` and returns the chosen name for the same item in the `PDDL` problem
        or `PDDL` domain generated by this `PDDLWriter`.

        :param item: The `unified_planning` entity renamed by this `PDDLWriter`.
        :return: The `PDDL` name of the given item.
        """
        try:
            return self.otn_renamings[item]
        except KeyError:
            raise UPException(
                f"The item {item} does not correspond to any item renamed."
            )

    def _populate_domain_objects(self, obe: ObjectsExtractor):
        self.domain_objects = {}
        # Iterate the actions to retrieve domain objects
        for a in self.problem.actions:
            if isinstance(a, up.model.InstantaneousAction):
                for p in a.preconditions:
                    _update_domain_objects(self.domain_objects, obe.get(p))
                for e in a.effects:
                    if e.is_conditional():
                        _update_domain_objects(
                            self.domain_objects, obe.get(e.condition)
                        )
                    _update_domain_objects(self.domain_objects, obe.get(e.fluent))
                    _update_domain_objects(self.domain_objects, obe.get(e.value))
            elif isinstance(a, DurativeAction):
                _update_domain_objects(self.domain_objects, obe.get(a.duration.lower))
                _update_domain_objects(self.domain_objects, obe.get(a.duration.upper))
                for interval, cl in a.conditions.items():
                    for c in cl:
                        _update_domain_objects(self.domain_objects, obe.get(c))
                for t, el in a.effects.items():
                    for e in el:
                        if e.is_conditional():
                            _update_domain_objects(
                                self.domain_objects, obe.get(e.condition)
                            )
                        _update_domain_objects(self.domain_objects, obe.get(e.fluent))
                        _update_domain_objects(self.domain_objects, obe.get(e.value))


def _get_pddl_name(
    item: Union[
        "up.model.Type",
        "up.model.Action",
        "up.model.Fluent",
        "up.model.Object",
        "up.model.Parameter",
        "up.model.Variable",
        "up.model.Problem",
    ]
) -> str:
    """This function returns a pddl name for the chosen item"""
    name = item.name  # type: ignore
    assert name is not None
    name = name.lower()
    regex = re.compile(r"^[a-zA-Z]+.*")
    if (
        re.match(regex, name) is None
    ):  # If the name does not start with an alphabetic char, we make it start with one.
        name = f'{INITIAL_LETTER.get(type(item), "x")}_{name}'

    name = re.sub("[^0-9a-zA-Z_]", "_", name)  # Substitute non-valid elements with "_"
    while (
        name in PDDL_KEYWORDS
    ):  # If the name is in the keywords, apply an underscore at the end until it is not a keyword anymore.
        name = f"{name}_"
    if isinstance(item, up.model.Parameter) or isinstance(item, up.model.Variable):
        name = f"?{name}"
    return name


def _update_domain_objects(
    dict_to_update: Dict[_UserType, Set[Object]], values: Dict[_UserType, Set[Object]]
) -> None:
    """Small utility method that updated a UserType -> Set[Object] dict with another dict of the same type."""
    for ut, os in values.items():
        os_to_update = dict_to_update.setdefault(ut, set())
        os_to_update |= os
