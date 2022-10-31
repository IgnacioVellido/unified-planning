import typing
from collections import OrderedDict
from fractions import Fraction
from itertools import product
from typing import Callable, Dict, List, Tuple, Union, cast

import pyparsing
import unified_planning as up
import unified_planning.model.htn as htn
import unified_planning.model.walkers
from unified_planning import model
from unified_planning.environment import Environment, get_env
from unified_planning.exceptions import UPUsageError
from unified_planning.io.hpdl import HPDLGrammar
from unified_planning.model import FNode, expression, problem
from unified_planning.model.effect import SimulatedEffect
from unified_planning.model.expression import Expression

if pyparsing.__version__ < "3.0.0":
    from pyparsing import ParseResults
else:
    from pyparsing.results import ParseResults


class HPDLReader:
    """
    Parse a `HPDL` domain file and, optionally, a `HPDL` problem file and generate the equivalent :class:`~unified_planning.model.Problem`.
    """

    def __init__(self, env: typing.Optional[Environment] = None):
        self._env = get_env(env)
        self._em = self._env.expression_manager
        self._tm = self._env.type_manager
        self._operators: Dict[str, Callable] = {
            "and": self._em.And,
            "or": self._em.Or,
            "not": self._em.Not,
            "imply": self._em.Implies,
            ">=": self._em.GE,
            "<=": self._em.LE,
            ">": self._em.GT,
            "<": self._em.LT,
            "=": self._em.Equals,
            "+": self._em.Plus,
            "-": self._em.Minus,
            "/": self._em.Div,
            "*": self._em.Times,
        }
        grammar = HPDLGrammar()
        self._pp_domain = grammar.domain
        self._pp_problem = grammar.problem
        self._pp_parameters = grammar.parameters
        self._fve = self._env.free_vars_extractor
        self._totalcost: typing.Optional[model.FNode] = None

        self.problem: model.Problem = None
        self.types_map: Dict[str, "model.Type"] = {}
        self.object_type_needed: bool = False
        self.universal_assignments: Dict["model.Action", List[ParseResults]] = {}
        self.has_actions_cost: bool = False

        self.inline_version = 0  # inline id counter
        self.derived = {}  # Parsed derived Dict[str, List[FNode]]

    # Parses sub_expressions and calls add_effect
    def _add_effect(
        self,
        act: Union[model.InstantaneousAction, model.DurativeAction],
        types_map: Dict[str, model.Type],
        universal_assignments: typing.Optional[
            Dict["model.Action", List[ParseResults]]
        ],
        exp: Union[ParseResults, str],
        cond: Union[model.FNode, bool] = True,
        timing: typing.Optional[model.Timing] = None,
        assignments: Dict[str, "model.Object"] = {},
    ):
        to_add = [(exp, cond)]
        while to_add:
            exp, cond = to_add.pop(0)
            if len(exp) == 0:
                continue  # ignore the case where the effect list is empty, e.g., `:effect ()`
            op = exp[0]
            if op == "and":
                exp = exp[1:]
                for e in exp:
                    to_add.append((e, cond))
            elif op == "when":
                cond = self._parse_exp({}, exp[1], assignments, types_map)
                to_add.append((exp[2], cond))
            elif op == "not":
                exp = exp[1]
                eff = (
                    self._parse_exp({}, exp, assignments, types_map),
                    self._em.FALSE(),
                    cond,
                )
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "assign":
                eff = (
                    self._parse_exp({}, exp[1], assignments, types_map),
                    self._parse_exp({}, exp[2], assignments, types_map),
                    cond,
                )
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore
                # act.add_effect(timing, exp[0], exp[1], exp[2])
            elif op == "increase":
                eff = (
                    self._parse_exp({}, exp[1], assignments, types_map),
                    self._parse_exp({}, exp[2], assignments, types_map),
                    cond,
                )
                act.add_increase_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "decrease":
                eff = (
                    self._parse_exp({}, exp[1], assignments, types_map),
                    self._parse_exp({}, exp[2], assignments, types_map),
                    cond,
                )
                act.add_decrease_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "forall":
                assert isinstance(exp, ParseResults)
                # Get the list of universal_assignments linked to this action. If it does not exist, default it to the empty list
                assert universal_assignments is not None
                action_assignments = universal_assignments.setdefault(act, [])
                action_assignments.append(exp)
            else:
                eff = (
                    self._parse_exp({}, exp, assignments, types_map),
                    self._em.TRUE(),
                    cond,
                )
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore

    # Checks (at start/end/overall) and calls _add_effect
    def _parse_condition(
        self,
        exp: Union[ParseResults, str],
        types_map: Dict[str, model.Type],
        vars: typing.Optional[Dict[str, model.Variable]] = None,
    ):
        to_add = [(exp, vars)]
        res = []
        while to_add:
            exp, vars = to_add.pop(0)
            op = exp[0]
            if op == "and":
                for e in exp[1:]:
                    to_add.append((e, vars))
            elif op in self.derived:  # Check derived
                for d in self.derived[op]:
                    res.append((model.StartTiming(), d))
            elif op == "forall":
                vars_string = " ".join(exp[1])
                vars_res = self._pp_parameters.parseString(vars_string)
                if vars is None:
                    vars = {}

                for g in vars_res["params"]:
                    # TODO: Check this
                    # t = types_map[g[1] if len(g) > 1 else "object"]
                    t = self.types_map[g[1] if len(g) > 1 else "object"]
                    for o in g[0]:
                        vars[o] = up.model.Variable(o, t, self._env)
                to_add.append((exp[2], vars))
            elif len(exp) == 3 and op == "at" and exp[1] == "start":
                cond = self._parse_exp(
                    {} if vars is None else vars, exp[2], {}, types_map
                )
                if vars is not None:
                    cond = self._em.Forall(cond, *vars.values())
                res.append((model.StartTiming(), cond))
            elif len(exp) == 3 and op == "at" and exp[1] == "end":
                cond = self._parse_exp(
                    {} if vars is None else vars, exp[2], {}, types_map
                )
                if vars is not None:
                    cond = self._em.Forall(cond, *vars.values())
                res.append((model.EndTiming(), cond))
            elif len(exp) == 3 and op == "over" and exp[1] == "all":
                t_all = model.OpenTimeInterval(model.StartTiming(), model.EndTiming())
                cond = self._parse_exp(
                    {} if vars is None else vars, exp[2], {}, types_map
                )
                if vars is not None:
                    cond = self._em.Forall(cond, *vars.values())
                res.append((t_all, cond))
            else:  # HPDL accept any exp, and considers (at start ...)
                # vars = {} if vars is None else vars
                cond = self._parse_exp({} if vars is None else vars, exp, {}, types_map)
                # cond = self._parse_exp({} if vars is None else vars, exp, {}, types_map)
                if vars is not None:
                    cond = self._em.Forall(cond, *vars.values())
                res.append((model.StartTiming(), cond))

        return res

    # Checks (at start/end/overall) and calls _add_effect
    def _add_timed_effects(
        self,
        act: model.DurativeAction,
        types_map: Dict[str, model.Type],
        universal_assignments: typing.Optional[
            Dict["model.Action", List[ParseResults]]
        ],
        eff: ParseResults,
        assignments: Dict[str, "model.Object"] = {},
    ):
        to_add = [eff]
        while to_add:
            eff = to_add.pop(0)
            op = eff[0]
            if op == "and":
                for e in eff[1:]:
                    to_add.append(e)
            elif len(eff) == 3 and op == "at" and eff[1] == "start":
                self._add_effect(
                    act,
                    types_map,
                    universal_assignments,
                    eff[2],
                    timing=model.StartTiming(),
                    assignments=assignments,
                )
            elif len(eff) == 3 and op == "at" and eff[1] == "end":
                self._add_effect(
                    act,
                    types_map,
                    universal_assignments,
                    eff[2],
                    timing=model.EndTiming(),
                    assignments=assignments,
                )
            elif len(eff) == 3 and op == "forall":
                assert universal_assignments is not None
                action_assignments = universal_assignments.setdefault(act, [])
                action_assignments.append(eff)
            else:  # HPDL accept any exp, and considers (at end ...)
                self._add_effect(
                    act,
                    types_map,
                    universal_assignments,
                    eff,  # CHANGED
                    timing=model.EndTiming(),
                    assignments=assignments,
                )

    def _check_if_object_type_is_needed(self, domain_res) -> bool:
        for p in domain_res.get("predicates", []):
            for g in p[1]:
                if len(g) <= 1 or g[1] == "object":
                    return True
        for p in domain_res.get("functions", []):
            for g in p[1]:
                if len(g) <= 1 or g[1] == "object":
                    return True
        for g in domain_res.get("constants", []):
            if len(g) <= 1 or g[1] == "object":
                return True
        for a in domain_res.get("actions", []):
            for g in a.get("params", []):
                if len(g) <= 1 or g[1] == "object":
                    return True
        return False

    def _durative_action_has_cost(self, dur_act: model.DurativeAction):
        if self._totalcost in self._fve.get(
            dur_act.duration.lower
        ) or self._totalcost in self._fve.get(dur_act.duration.upper):
            return False
        for _, cl in dur_act.conditions.items():
            for c in cl:
                if self._totalcost in self._fve.get(c):
                    return False
        for _, el in dur_act.effects.items():
            for e in el:
                if (
                    self._totalcost in self._fve.get(e.fluent)
                    or self._totalcost in self._fve.get(e.value)
                    or self._totalcost in self._fve.get(e.condition)
                ):
                    return False
        return True

    def _instantaneous_action_has_cost(self, act: model.InstantaneousAction):
        for c in act.preconditions:
            if self._totalcost in self._fve.get(c):
                return False
        for e in act.effects:
            if self._totalcost in self._fve.get(
                e.value
            ) or self._totalcost in self._fve.get(e.condition):
                return False
            if e.fluent == self._totalcost:
                if (
                    not e.is_increase()
                    or not e.condition.is_true()
                    or not (e.value.is_int_constant() or e.value.is_real_constant())
                ):
                    return False
        return True

    def _problem_has_actions_cost(self, problem: model.Problem):
        if (
            self._totalcost is None
            or not problem.initial_value(self._totalcost).constant_value() == 0
        ):
            return False
        for _, el in problem.timed_effects.items():
            for e in el:
                if (
                    self._totalcost in self._fve.get(e.fluent)
                    or self._totalcost in self._fve.get(e.value)
                    or self._totalcost in self._fve.get(e.condition)
                ):
                    return False
        for c in problem.goals:
            if self._totalcost in self._fve.get(c):
                return False
        return True

    def _build_problem(self, name: str, features: List[str]) -> model.Problem:
        features = set(features)
        if ":hierarchy" in features or ":htn-expansion" in features:
            return htn.HierarchicalProblem(
                name,
                self._env,
                initial_defaults={self._tm.BoolType(): self._em.FALSE()},
            )

        raise Exception("HPDLReader: Wrong reader, use PDDL instead")
        # return model.Problem(
        #     name,
        #     self._env,
        #     initial_defaults={self._tm.BoolType(): self._em.FALSE()},
        # )

    def _parse_types(self, types_list: List[str]):
        """Parses a type from the domain"""
        # types_list is a List of 1 or 2 elements, where the first one
        # is a List of types, and the second one can be their father,
        # if they have one.
        father: typing.Optional["model.Type"] = None
        if len(types_list) == 2:  # the types have a father
            if types_list[1] != "object":  # the father is not object
                father = self.types_map[types_list[1]]
            elif self.object_type_needed:  # the father is object, and object is needed
                object_type = self.types_map.get("object", None)
                if object_type is None:  # the type object is not defined
                    father = self._env.type_manager.UserType("object", None)
                    self.types_map["object"] = father
                else:
                    father = object_type
        else:
            assert (
                len(types_list) == 1
            ), "Malformed list of types, I was expecting either 1 or 2 elements"  # sanity check
        for type_name in types_list[0]:
            self.types_map[type_name] = self._env.type_manager.UserType(
                type_name, father
            )

    def _parse_predicate(self, predicate: List[str]) -> model.Fluent:
        name = predicate[0]
        params = self._parse_params(predicate[1])
        return model.Fluent(name, self._tm.BoolType(), params, self._env)

    def _parse_params(self, params: List[str]) -> OrderedDict:
        res_params = OrderedDict()
        for g in params:
            param_type = self.types_map[g[1] if len(g) > 1 else "object"]
            for param_name in g[0]:
                res_params[param_name] = param_type
        return res_params

    def _parse_function(self, func: OrderedDict) -> model.Fluent:
        name = func[0][0]  # Modified due to added grammar group
        params = self._parse_params(func[0][1])
        f = model.Fluent(name, self._tm.RealType(), params, self._env)
        if name == "total-cost":
            self.has_actions_cost = True
            self._totalcost = cast(model.FNode, self._em.FluentExp(f))
        return f

    def _parse_constant(self, constant: List[str]) -> List[model.Object]:
        o_type = self.types_map[constant[1] if len(constant) > 1 else "object"]
        objects: List[model.Object] = []
        for name in constant[0]:
            objects.append(model.Object(name, o_type, self._env))
        return objects

    def _build_task(self, task: OrderedDict) -> htn.Task:
        task_name = task["name"]

        task_params = self._parse_params(task.get("params", []))
        return htn.Task(task_name, task_params)

    def _parse_exp_str(
        self,
        var: Dict[str, model.Variable],
        exp: str,
        assignments: Dict[str, "model.Object"] = {},
        available_params: typing.Optional[
            OrderedDict[str, "model.Type"]
        ] = None,  # If we are parsing a precondition or effect, we need to know the valid parameters
    ) -> model.FNode:
        if exp[0] == "?" and exp[1:] in var:  # variable in a quantifier expression
            return self._em.VariableExp(var[exp[1:]])
        elif exp[0] in self.derived:  # Check derived
            res = []
            for d in self.derived[exp[0]]:
                res.append(d)

            op: Callable = self._operators["and"]
            return op(*res)
        elif exp in assignments:  # quantified assignment variable
            return self._em.ObjectExp(assignments[exp])
        elif exp[0] == "?":  # action parameter
            assert available_params is not None, "valid_params cannot be None"
            assert exp[1:] in available_params, f"Invalid parameter {exp[1:]}"

            return self._em.ParameterExp(
                model.Parameter(exp[1:], available_params[exp[1:]], self._env)
            )
        elif self.problem.has_fluent(exp):  # fluent
            return self._em.FluentExp(self.problem.fluent(exp))
        elif self.problem.has_object(exp):  # object
            return self._em.ObjectExp(self.problem.object(exp))
        else:  # number
            n = Fraction(exp)
            if n.denominator == 1:
                return self._em.Int(n.numerator)
            else:
                return self._em.Real(n)

    def _parse_exp_parse_result(
        self,
        var: Dict[str, model.Variable],
        exp: ParseResults,
        assignments: Dict[str, "model.Object"],
    ) -> Tuple[
        Union[model.FNode, None], Union[List[Tuple[typing.Any, typing.Any, bool]], None]
    ]:
        if len(exp) == 0:  # empty precodition
            return self._em.TRUE(), None
        elif exp[0] == "-" and len(exp) == 2:  # unary minus
            return None, [(var, exp, True), (var, exp[1], False)]
        elif exp[0] in self._operators:  # n-ary operators
            res = [(var, exp, True)]
            for e in exp[1:]:
                res.append((var, e, False))
            return None, res
        elif exp[0] in self.derived:  # Check derived and substitute
            res = []
            for d in self.derived[exp[0]]:
                res.append(d)

            op: Callable = self._operators["and"]
            return op(*res), None  # Returns the FNodes
        elif exp[0] in ["exists", "forall"]:  # quantifier operators
            vars_string = " ".join(exp[1])
            vars_res = self._pp_parameters.parseString(vars_string)
            vars = {}
            for g in vars_res["params"]:
                t = self.types_map[g[1] if len(g) > 1 else "object"]
                for o in g[0]:
                    vars[o] = model.Variable(o, t, self._env)
            return None, [(vars, exp, True), (vars, exp[2], False)]
        elif self.problem.has_fluent(exp[0]):  # fluent reference
            res = [(var, exp, True)]
            for e in exp[1:]:
                # If type is received, exp declares an
                # object and is processed in build_method
                if e != "-" and not self.problem.has_type(e):
                    res.append((var, e, False))
            return None, res
        elif exp[0] in assignments:  # quantified assignment variable
            assert len(exp) == 1
            return None, [(var, exp, True)]
        elif len(exp) == 1:  # expand an element inside brackets
            return None, [(var, exp[0], False)]
        else:
            raise SyntaxError(f"Not able to handle: {exp}")

    def _parse_exp(
        self,
        var: Dict[str, model.Variable],
        exp: Union[ParseResults, str],
        assignments: Dict[str, "model.Object"] = {},
        available_params: typing.Optional[
            OrderedDict[str, "model.Type"]
        ] = None,  # If we are parsing a precondition or effect, we need to know the valid parameters
    ) -> model.FNode:
        stack = [(var, exp, False)]
        solved: List[model.FNode] = []
        while len(stack) > 0:
            var, exp, status = stack.pop()
            if status:
                if exp[0] == "-" and len(exp) == 2:  # unary minus
                    solved.append(self._em.Times(-1, solved.pop()))
                elif exp[0] in self._operators:  # n-ary operators
                    op: Callable = self._operators[exp[0]]
                    solved.append(op(*[solved.pop() for _ in exp[1:]]))
                elif exp[0] in ["exists", "forall"]:  # quantifier operators
                    q_op: Callable = (
                        self._em.Exists if exp[0] == "exists" else self._em.Forall
                    )
                    solved.append(q_op(solved.pop(), *var.values()))
                elif self.problem.has_fluent(exp[0]):  # fluent reference
                    f = self.problem.fluent(exp[0])
                    # In object is declared in the exp do not pop for the (- object)
                    args = [
                        solved.pop()
                        for e in exp[1:]
                        if e != "-" and not self.problem.has_type(e)
                    ]
                    solved.append(self._em.FluentExp(f, tuple(args)))
                elif exp[0] in assignments:  # quantified assignment variable
                    assert len(exp) == 1
                    solved.append(self._em.ObjectExp(assignments[exp[0]]))
                else:
                    raise up.exceptions.UPUnreachableCodeError
            else:
                if isinstance(exp, ParseResults):
                    node, new_stack = self._parse_exp_parse_result(
                        var, exp, assignments
                    )
                    if node:
                        solved.append(node)
                    if new_stack:
                        stack.extend(new_stack)
                elif isinstance(exp, str):
                    # Instead of having to pass the action already built, we can pass the available parameters
                    node = self._parse_exp_str(var, exp, assignments, available_params)
                    solved.append(node)
                else:
                    raise SyntaxError(f"Not able to handle: {exp}")
        assert len(solved) == 1  # sanity check
        return solved.pop()

    def _parse_effect(
        self,
        exp: Union[ParseResults, str],
        cond: Union[model.FNode, bool] = True,
        timing: typing.Optional[model.Timing] = None,
        assignments: Dict[str, "model.Object"] = {},
        available_params: typing.Optional[OrderedDict[str, "model.Type"]] = None,
    ) -> List[Tuple[model.FNode, model.FNode, Union[model.FNode, bool], str]]:
        to_add = [(exp, cond)]
        result: List[
            Tuple[model.FNode, model.FNode, Union[model.FNode, bool], str]
        ] = []
        while to_add:
            exp, cond = to_add.pop(0)
            if len(exp) == 0:
                continue  # ignore the case where the effect list is empty, e.g., `:effect ()`
            op = exp[0]
            if op == "and":
                exp = exp[1:]
                for e in exp:
                    to_add.append((e, cond))
            elif op == "when":
                cond = self._parse_exp({}, exp[1], assignments, available_params)
                to_add.append((exp[2], cond))
            elif op == "not":
                exp = exp[1]
                eff = (
                    self._parse_exp({}, exp, assignments, available_params),
                    self._em.FALSE(),
                    cond,
                    op,
                )
                result.append(eff)
                # act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "assign" or op == "increase" or op == "decrease":
                eff = (
                    self._parse_exp({}, exp[1], assignments, available_params),
                    self._parse_exp({}, exp[2], assignments, available_params),
                    cond,
                    op,
                )
                result.append(eff)
                # act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "forall":
                assert isinstance(exp, ParseResults)
                # Get the list of universal_assignments linked to this action. If it does not exist, default it to the empty list
                # assert self.universal_assignments is not None
                # assert act is not None
                # action_assignments = self.universal_assignments.setdefault(act, [])
                # action_assignments.append(exp)

                # Returning and checking elsewhere
                result.append(exp)
            else:
                eff = (
                    self._parse_exp({}, exp, assignments, available_params),
                    self._em.TRUE(),
                    cond,
                    None,
                )
                result.append(eff)
                # act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore

        return result

    def _build_durative_action(
        self,
        name: str,
        params: OrderedDict,
        duration: ParseResults,
        cond: Union[ParseResults, str],
        eff: ParseResults,
    ) -> model.DurativeAction:

        dur_act = model.DurativeAction(name, params, self._env)

        # Parse duration
        if duration[0] == "=":
            duration.pop(0)
            duration.pop(0)
            dur_act.set_fixed_duration(self._parse_exp({}, duration, {}, params))
        elif duration[0] == "and":
            upper = None
            lower = None
            for j in range(1, len(duration)):
                if duration[j][0] == ">=" and lower is None:
                    duration[j].pop(0)
                    duration[j].pop(0)
                    lower = self._parse_exp({}, duration[j], {}, params)
                elif duration[j][0] == "<=" and upper is None:
                    duration[j].pop(0)
                    duration[j].pop(0)
                    upper = self._parse_exp({}, duration[j], {}, params)
                else:
                    raise SyntaxError(
                        f"Not able to handle duration constraint of action {name}"
                    )
            if lower is None or upper is None:
                raise SyntaxError(
                    f"Not able to handle duration constraint of action {name}"
                )
            d = model.ClosedDurationInterval(lower, upper)
            dur_act.set_duration_constraint(d)
        else:
            raise SyntaxError(
                f"Not able to handle duration constraint of action {name}"
            )

        # Add conditions to action
        conditions = self._parse_condition(cond, params)
        for c in conditions:
            dur_act.add_condition(c[0], c[1])

        # Add each effect to action
        self._add_timed_effects(dur_act, params, self.universal_assignments, eff)

        # Check action cost
        self.has_actions_cost = (
            self.has_actions_cost and self._durative_action_has_cost(dur_act)
        )

        return dur_act

    def _build_action(
        self,
        name: str,
        params: OrderedDict,
        pre: model.FNode = None,
        eff: List[Tuple[model.FNode, model.FNode, Union[model.FNode, bool], str]] = [],
        durative: bool = False,
        duration: typing.Optional[List[str]] = None,
    ) -> model.Action:

        # Durative actions
        if durative:
            return self._build_durative_action(name, params, duration[0], pre, eff)

        act = model.InstantaneousAction(name, params, self._env)

        if pre:
            act.add_precondition(pre)

        for f in eff:
            if f[0] == "forall":
                assert self.universal_assignments is not None
                action_assignments = self.universal_assignments.setdefault(act, [])
                action_assignments.append(f)
            elif f[3] == "assign":
                act.add_effect(f[0], f[1], f[2])
            elif f[3] == "increase":
                act.add_increase_effect(f[0], f[1], f[2])
            elif f[3] == "decrease":
                act.add_decrease_effect(f[0], f[1], f[2])
            else:
                act.add_effect(f[0], f[1], f[2])

        # self.problem.add_action(act)
        # Do we need to do it here? it comes from _parse_problem method
        self.has_actions_cost = (
            self.has_actions_cost and self._instantaneous_action_has_cost(act)
        )
        return act

    def _parse_action(
        self, action: OrderedDict
    ) -> Tuple[str, OrderedDict, bool, List[str], List[model.Fluent]]:
        """Parses an action from the domain and return the name and the parameters"""
        action_name = action["name"]
        a_params = self._parse_params(action["params"])
        durative = False
        duration = None

        if "duration" in action:
            durative = True
            duration = action["duration"]

        res = OrderedDict()

        res["name"] = action_name
        res["params"] = a_params
        res["durative"] = durative
        res["duration"] = duration

        if "pre" in action:
            res["pre"] = self._parse_exp({}, action["pre"][0], {}, a_params)
        else:  # "pre" always exist, although it might be empty
            res["pre"] = []

        if "eff" in action:
            res["eff"] = self._parse_effect(action["eff"][0], True, None, {}, a_params)
        else:
            res["eff"] = []

        return res

    def _parse_inline(
        self,
        inline,
        # method_name: str,
        method_params: OrderedDict,  # Task and pre params of method
    ) -> model.Action:
        inline_version = 0
        # inline_name = (method_name or "") + "_inline"
        inline_name = "inline"

        # Find the first available name for the inline task
        inline_version = self.inline_version
        self.inline_version += 1

        res = OrderedDict()

        res["name"] = f"{inline_name}_{inline_version}"
        res["durative"] = False
        res["duration"] = []

        # Check for params declared in inline
        cond_params = self._get_params_in_exp(inline["cond"][0])
        eff_params = self._get_params_in_exp(inline["eff"][0])

        # Join with method_params
        res["params"] = method_params
        res["params"].update(cond_params)
        res["params"].update(eff_params)
        res["cond"] = []
        res["eff"] = []

        # Parse conditions and effects
        if "cond" in inline:
            res["cond"] = self._parse_exp({}, inline["cond"][0], {}, res["params"])

        if "eff" in inline:
            res["eff"] = self._parse_effect(
                inline["eff"][0], True, None, {}, res["params"]
            )

        # Build action
        action_model = self._build_action(
            res["name"],
            res["params"],
            res["cond"],
            res["eff"],
            res["durative"],
            res["duration"],
        )

        # Add inline to the problem
        self.problem.add_action(action_model)

        # Return subtask
        return htn.Subtask(action_model, *action_model.parameters)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def _get_params_in_exp(self, exp) -> OrderedDict:
        """Returns params found in exp"""
        params = OrderedDict()

        stack = [exp]
        while len(stack) > 0:
            exp = stack.pop()

            # Check for single strings
            # (probably won't return anything, as each string is only one word)
            if isinstance(exp, str):
                params.update(self._parse_params_and_types(exp))
            elif len(exp) >= 2:
                # Check if all elements are strings
                if all(isinstance(e, str) for e in exp):
                    params.update(self._parse_params_and_types(exp))
                else:
                    # Find params in each sub expression
                    for e in exp:
                        stack.append(e)

        return params

    def _parse_params_and_types(self, params: List[str]) -> List[str]:
        """Parses a list of parameters and returns a list of the parameters names. Only returns declared paramaters (?o - <type>)"""

        def parse_type(type: str):
            if type in self.types_map:
                return self.types_map[type]  # Must return the object, not the str
            else:
                raise ValueError(f"Type {type} not defined")

        res_params = OrderedDict()
        i = 0
        while i < len(params):
            if params[i][0] == "?":  # parameter
                if (
                    i + 1 < len(params) and params[i + 1] == "-"
                ):  # type is specified, check it
                    res_params[params[i][1:]] = parse_type(params[i + 2])
                    i += 3
                else:  # No type specified, ignore
                    i += 1
                    # In case we need to get non-specified params,
                    # change line above

            else:  # Not a param, ignore
                i += 1

        return res_params

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def _build_task(self, task: OrderedDict) -> htn.Task:
        task_name = task["name"]

        task_params = self._parse_params(task.get("params", []))

        return htn.Task(task_name, task_params)

    def _parse_method(
        self,
        method: OrderedDict,
        method_params: OrderedDict,  # Task and pre params of method
    ):
        # Parse subtasks
        ordered_subtasks = []  # List of tuple (order, list(subtask))
        subtasks_params = []

        # Get ordered subtasks
        for ordering in method.get("subtasks", []):
            subtasks = []  # List of model.Subtask
            order = ordering.get("ordering", "(")

            # ordering[0] is the order tag, rest are subtasks definitions
            for subs in ordering[1:]:
                # TODO: Add time restrictions from branch time-constraints
                time = subs.get("time_exp", None)
                if time is not None:
                    continue

                subtask_model = self._parse_subtask(subs, method_params)

                if subtask_model is not None:
                    # Add model to list
                    subtasks.append(subtask_model)

                    # Get model.Parameter for each param
                    # TODO: See parse_inline params TODOs
                    for p in subtask_model.parameters:
                        subtasks_params.append(p.parameter())

            # Append ordering to method subtasks
            ordered_subtasks.append((order, subtasks))

        return ordered_subtasks, subtasks_params

    # self.problem must have task built
    def _build_method(self, method: OrderedDict, task_name: str) -> htn.Method:
        method_name = f'{task_name}-{method["name"]}'  # Methods names are
        # not unique across tasks

        method_params = OrderedDict()

        # Get parent task params
        task_model = self.problem.get_task(task_name)
        task_params = OrderedDict({p.name: p.type for p in task_model.parameters})
        method_params.update(task_params)

        # Get precondition params
        method_preconditions = method.get("pre", [])
        for pre in method_preconditions:
            pre_params = self._get_params_in_exp(pre)
            method_params.update(pre_params)

        # Parse and build method subtasks
        subtasks, params = self._parse_method(method, method_params)

        # Get subtasks params as OrderedDict
        subtask_params = OrderedDict({p.name: p.type for p in params})
        method_params.update(subtask_params)

        # ----------------------------
        # Build model
        method_model = htn.Method(method_name, method_params)

        # Add parent task to model
        method_model.set_task(task_model)

        # Add subtasks to model
        for ordering in subtasks:
            for s in ordering[1]:
                method_model.add_subtask(s)

            if ordering[0] == "(":
                method_model.set_ordered(*ordering[1])

        # All subtasks from the next iteration have sequential order with
        # respect to the previous iteration
        if len(subtasks) >= 2:
            for i in range(1, len(subtasks)):
                # Loop through subtasks of previous ordering
                for s1 in subtasks[i - 1][1]:
                    for s2 in subtasks[i][1]:
                        method_model.set_ordered(s1, s2)

        # Add preconditions to model
        for pre in method_preconditions:
            parsed_pre = self._parse_exp({}, pre, {}, method_params)
            method_model.add_precondition(parsed_pre)

        return method_model

    def _parse_subtask(
        self,
        subtask: OrderedDict,
        method_params: OrderedDict,  # Task and pre params of method
    ):
        if "cond" in subtask.keys():  # == inline
            return self._parse_inline(subtask, method_params)

        task_name = subtask["name"]

        task: Union[htn.Task, model.Action]
        if self.problem.has_task(task_name):
            task = self.problem.get_task(task_name)
        elif self.problem.has_action(task_name):
            task = self.problem.action(task_name)
        else:
            return None
        assert isinstance(task, htn.Task) or isinstance(task, model.Action)

        # 1: Find subtask params
        subt_params = self._parse_params(subtask["params"])

        # 2: Find params of action/task invoked
        task_params = OrderedDict({p.name: p.type for p in task.parameters})

        # Set the variable names of task_params to subt_params
        params = OrderedDict()
        for s_p, t_p in zip(subt_params.items(), task_params.items()):
            params[s_p[0]] = t_p[1]

        # 3: Parse exp adding ? to each variable (somehow without ? it fails)
        parameters = [
            self._parse_exp({}, "?" + str(param), {}, params) for param in subt_params
        ]

        # Create and return Subtask
        return htn.Subtask(task, *parameters)

    # _________________________________________________________

    # TODO: Wait for python-fluents discussion
    def _parse_function_code(
        self, fluent: model.Fluent, code: str
    ) -> model.SimulatedEffect:

        # Replace ?var with var
        for param in fluent.signature:
            code = code.replace(f"?{param.name}", f"{param.name}")

        # Dont judge me
        def _inner_fun(
            problem: model.Problem,
            state: model.ROState,
            actual_params: Dict["up.model.parameter.Parameter", "up.model.fnode.FNode"],
        ):
            values = {}
            for param in fluent.signature:
                values[param.name] = state.get_value(
                    actual_params.get(param)
                ).constant_value()

            # Dont judge me 2
            # TODO: the function must return a list of values that correspond to
            # the input fluents.
            # Example:
            # def fun(problem, state, actual_params):
            #     value = state.get_value(battery_charge(actual_params.get(robot))).constant_value()
            #     return [Int(value - 10)]
            # move.set_simulated_effect(SimulatedEffect([battery_charge(robot)], fun))
            return eval(code, {}, values)

        # TODO: An action only have one simulatedEffect therefore we have to detect every code_function called
        # On the effects of the action and merge them into one simulatedEffect
        return model.SimulatedEffect([fluent(*fluent.signature)], _inner_fun)

    # NOTE: Derived are declared in :predicates, so no need to add
    # fluent to problem here, they are already in there
    def _parse_derived(self, derived: OrderedDict) -> Dict[str, List[FNode]]:

        name = derived[1][0]
        params = self._parse_params(derived[1][1])

        fluents = []
        for exp in derived.get("exp", None):  # Add fluents
            fluent = self._parse_exp({}, exp, {}, params)  # Returns FNode
            fluents.append(fluent)

        self.derived[name] = fluents

        # TODO: Wait for discussion #247 before deleting this
        # return model.Derived(name, self._tm.BoolType(), params, self._env, fluents)

    def parse_problem(
        self, domain_filename: str, problem_filename: typing.Optional[str] = None
    ) -> "model.Problem":
        """
        Takes in input a filename containing the `HPDL` domain and optionally a filename
        containing the `HPDL` problem and returns the parsed `Problem`.

        Note that if the `problem_filename` is `None`, an incomplete `Problem` will be returned.

        :param domain_filename: The path to the file containing the `HPDL` domain.
        :param problem_filename: Optionally the path to the file containing the `HPDL` problem.
        :return: The `Problem` parsed from the given HPDL domain + problem.
        :return: The `Problem` parsed from the given pddl domain + problem.
        """
        domain_res = self._pp_domain.parseFile(domain_filename)

        self.problem = self._build_problem(domain_res["name"], domain_res["features"])
        self.types_map: Dict[str, "model.Type"] = {}
        self.object_type_needed: bool = self._check_if_object_type_is_needed(domain_res)
        self.universal_assignments: Dict["model.Action", List[ParseResults]] = {}
        self.has_actions_cost = False
        ##

        for types_list in domain_res.get("types", []):
            self._parse_types(types_list)

        # Check object type is defined
        if (
            self.object_type_needed and "object" not in self.types_map
        ):  # The object type is needed, but has not been defined
            self.types_map["object"] = self._env.type_manager.UserType(
                "object", None
            )  # We manually define it.

        for p in domain_res.get("predicates", []):
            fluent = self._parse_predicate(p)
            self.problem.add_fluent(fluent)

        for f in domain_res.get("functions", []):
            func = self._parse_function(f)
            code = f.get("code", None)
            if code:
                simulated_effect = self._parse_function_code(func, code[0])
                print("effect", simulated_effect)

            self.problem.add_fluent(func)

        # Must go after functions, as they can be used in derived
        for d in domain_res.get("derived", []):
            self._parse_derived(d)

        for c in domain_res.get("constants", []):
            objects = self._parse_constant(c)
            for o in objects:
                self.problem.add_object(o)

        for task in domain_res.get("tasks", []):
            assert isinstance(self.problem, htn.HierarchicalProblem)
            task_model = self._build_task(task)
            self.problem.add_task(task_model)

        for a in domain_res.get("actions", []):
            if not "duration" in a:
                parsed_action = self._parse_action(a)
                action_model = self._build_action(
                    parsed_action["name"],
                    parsed_action["params"],
                    parsed_action["pre"],
                    parsed_action["eff"],
                    parsed_action["durative"],
                    parsed_action["duration"],
                )
                self.problem.add_action(action_model)
            else:
                action_model = self._build_durative_action(
                    a["name"],
                    self._parse_params(a["params"]),
                    a["duration"][0],
                    a["pre"][0],
                    a["eff"][0],
                )
                self.problem.add_action(action_model)

        # Methods are defined inside tasks;
        # we need to parse them after all tasks and actions have been defined
        # because _parse_subtasks() needs to be able to find them.
        for task in domain_res.get("tasks", []):
            for method in task.get("methods", []):
                method_model = self._build_method(method, task["name"])
                self.problem.add_method(method_model)

        # Parse problem
        if problem_filename is not None:
            problem_res = self._pp_problem.parseFile(problem_filename)

            self.problem.name = problem_res["name"]

            objects = problem_res.get("objects", [])
            objects = self._parse_params(objects)

            for var, kind in objects.items():
                self.problem.add_object(model.Object(var, kind, self.problem.env))

            # Add universal_assignments (forall, something-else?)
            for action, eff_list in self.universal_assignments.items():
                for eff in eff_list:
                    # Parse the variable definition part and create 2 lists, the first one with the variable names,
                    # the second one with the variable types.
                    vars_string = " ".join(eff[1])
                    vars_res = self._pp_parameters.parseString(vars_string)
                    var_names: List[str] = []
                    var_types: List["model.Type"] = []

                    for g in vars_res["params"]:
                        t = self.types_map[g[1] if len(g) > 1 else "object"]
                        for o in g[0]:
                            var_names.append(f"?{o}")
                            var_types.append(t)

                    # for each variable type, get all the objects of that type and calculate the cartesian
                    # product between all the given objects and iterate over them, changing the variable assignments
                    # in the added effect
                    for objects in product(
                        *(self.problem.objects(t) for t in var_types)
                    ):
                        assert len(var_names) == len(objects)
                        assignments = {
                            name: obj for name, obj in zip(var_names, objects)
                        }
                        if isinstance(action, model.InstantaneousAction):
                            self._add_effect(
                                action,
                                self.types_map,
                                None,
                                eff[2],
                                assignments=assignments,
                            )
                        elif isinstance(action, model.DurativeAction):
                            # TODO: There should be another way for this
                            # Create dict with params and its types
                            params = OrderedDict(
                                [(k, v.type) for k, v in action._parameters.items()]
                            )

                            self._add_timed_effects(
                                action,
                                params,
                                None,
                                eff[2],
                                assignments=assignments,
                            )
                        else:
                            raise NotImplementedError

            # TODO: customization (time format/start/horizon/unit)

            for i in problem_res.get("init", []):
                if i[0] == "=":
                    self.problem.set_initial_value(
                        self._parse_exp({}, i[1]),
                        self._parse_exp({}, i[2]),
                    )
                elif (
                    len(i) == 3 and i[0] == "at" and i[1].replace(".", "", 1).isdigit()
                ):
                    ti = model.StartTiming(Fraction(i[1]))
                    va = self._parse_exp({}, i[2])
                    if va.is_fluent_exp():
                        self.problem.add_timed_effect(ti, va, self._em.TRUE())
                    elif va.is_not():
                        self.problem.add_timed_effect(ti, va.arg(0), self._em.FALSE())
                    elif va.is_equals():
                        self.problem.add_timed_effect(ti, va.arg(0), va.arg(1))
                    else:
                        raise SyntaxError(f"Not able to handle this TIL {i}")
                else:
                    self.problem.set_initial_value(
                        self._parse_exp({}, i),
                        self._em.TRUE(),
                    )

            # HPDL task-goal is the equivalent of HDDL htn tasks
            tasknet = problem_res.get("goal", None)
            if tasknet is not None:
                subtasks, _ = self._parse_method(tasknet, self.types_map)

                # Add subtasks to task_network
                for ordering in subtasks:
                    for s in ordering[1]:
                        self.problem.task_network.add_subtask(s)

                    if ordering[0] == "(":
                        self.problem.task_network.set_ordered(*ordering[1])

                # All subtasks from the next iteration have sequential order with
                # respect to the previous iteration
                if len(ordering) >= 2:
                    for i in range(1, len(subtasks)):
                        # Loop through subtasks of previous ordering
                        for s1 in subtasks[i - 1][1]:
                            for s2 in subtasks[i][1]:
                                self.problem.task_network.set_ordered(s1, s2)

            self.has_actions_cost = (
                self.has_actions_cost and self._problem_has_actions_cost(self.problem)
            )

            optimization = problem_res.get("optimization", None)
            metric = problem_res.get("metric", None)

            if metric is not None:
                if (
                    optimization == "minimize"
                    and len(metric) == 1
                    and metric[0] == "total-time"
                ):
                    self.problem.add_quality_metric(model.metrics.MinimizeMakespan())
                else:
                    metric_exp = self._parse_exp({}, metric)
                    if (
                        self.has_actions_cost
                        and optimization == "minimize"
                        and metric_exp == self._totalcost
                    ):
                        costs = {}
                        self.problem._fluents.remove(self._totalcost.fluent())
                        self.problem._initial_value.pop(self._totalcost)
                        use_plan_length = all(
                            False for _ in self.problem.durative_actions
                        )
                        for a in self.problem.instantaneous_actions:
                            cost = None
                            for e in a.effects:
                                if e.fluent == self._totalcost:
                                    cost = e
                                    break
                            if cost is not None:
                                costs[a] = cost.value
                                a._effects.remove(cost)
                                if cost.value != 1:
                                    use_plan_length = False
                            else:
                                use_plan_length = False
                        if use_plan_length:
                            self.problem.add_quality_metric(
                                model.metrics.MinimizeSequentialPlanLength()
                            )
                        else:
                            self.problem.add_quality_metric(
                                model.metrics.MinimizeActionCosts(
                                    costs, self._em.Int(0)
                                )
                            )
                    else:
                        if optimization == "minimize":
                            self.problem.add_quality_metric(
                                model.metrics.MinimizeExpressionOnFinalState(metric_exp)
                            )
                        elif optimization == "maximize":
                            self.problem.add_quality_metric(
                                model.metrics.MaximizeExpressionOnFinalState(metric_exp)
                            )
        else:
            if len(self.universal_assignments) != 0:
                raise UPUsageError(
                    "The domain has quantified assignments. In the unified_planning library this is compatible only if the problem is given and not only the domain."
                )
        return self.problem
