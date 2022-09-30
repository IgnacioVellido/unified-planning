import typing
from collections import OrderedDict
from fractions import Fraction
from itertools import product
from typing import Callable, Dict, List, Tuple, Union, cast

import pyparsing
import unified_planning as up
import unified_planning.model.htn as htn
import unified_planning.model.walkers
from pyparsing import (
    Group,
    Keyword,
    OneOrMore,
    Optional,
    QuotedString,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    alphas,
    nestedExpr,
    nums,
    restOfLine,
)
from unified_planning import model
from unified_planning.environment import Environment, get_env
from unified_planning.exceptions import UPUsageError
from unified_planning.model import FNode, expression, problem
from unified_planning.model.expression import Expression

if pyparsing.__version__ < "3.0.0":
    from pyparsing import ParseResults
    from pyparsing import oneOf as one_of
else:
    from pyparsing import one_of
    from pyparsing.results import ParseResults


class HPDLGrammar:
    def __init__(self):
        name = Word(alphas, alphanums + "_" + "-")
        number = Word(nums + "." + "-")

        # To define subtypes
        # obj1 obj2 - parent
        name_list = Group(Group(OneOrMore(name)) + Optional(Suppress("-") + name))

        variable = Suppress("?") + name
        typed_variables = Group(
            Group(OneOrMore(variable)).setResultsName("variables")
            + Optional(Suppress("-") + name).setResultsName("type")
        )

        # Group of one or more variable and optionally their type
        parameters = Group(ZeroOrMore(typed_variables)).setResultsName("params")

        # Any predicate with/without parameters
        predicate = (
            Suppress("(")
            + Group(name.setResultsName("name") + parameters)
            + Suppress(")")
        )

        # List of predicates
        and_predicate = (
            Suppress("(")
            + "and"
            + Group(OneOrMore(predicate))  # | operation))
            + Suppress(")")
        )

        operator = one_of(
            "and or not imply >= <= > < = + - / * increase decrease         assign scale-up scale-down"
        ).setResultsName("operator")

        operation = Group(
            Suppress("(")
            + operator
            + (number | predicate).setResultsName("operand1")
            + (number | predicate).setResultsName("operand2")
            + Suppress(")")
        )

        # ----------------------------------------------------------
        # Sections
        sec_requirements = (
            Suppress("(")
            + ":requirements"
            + OneOrMore(
                one_of(
                    ":strips :typing :negative-preconditions :disjunctive-preconditions :equality :existential-preconditions :universal-preconditions :quantified-preconditions :conditional-effects :fluents :numeric-fluents :adl :durative-actions :duration-inequalities :timed-initial-literals :action-costs :hierarchy :htn-expansion :metatags :derived-predicates :negative-preconditions"
                )
            )
            + Suppress(")")
        )

        sec_types = (
            Suppress("(")
            + ":types"
            + OneOrMore(name_list).setResultsName("types")
            + Suppress(")")
        )

        # Same as sec_types
        sec_constants = (
            Suppress("(")
            + ":constants"
            # + Optional(
            + OneOrMore(name_list).setResultsName("constants")
            # )
            + Suppress(")")
        )

        sec_predicates = (
            Suppress("(")
            + ":predicates"
            + Group(OneOrMore(predicate)).setResultsName("predicates")
            + Suppress(")")
        )

        # Functions can specify -number type
        sec_functions = (
            Suppress("(")
            + ":functions"
            + Group(
                OneOrMore(predicate + Optional(Suppress("- number")))
            ).setResultsName("functions")
            + Suppress(")")
        )

        # derived evaluates an expression and returns a boolean?
        derived = Group(
            Suppress("(")
            + ":derived"
            + nestedExpr().setResultsName("pre")
            + nestedExpr().setResultsName("post")
            + Suppress(")")
        )

        # ----------------------------------------------------------
        # Actions

        action = Group(
            Suppress("(")
            + ":action"
            + name.setResultsName("name")
            + ":parameters"
            + Suppress("(")
            + parameters
            + Suppress(")")
            + Optional(":precondition" + nestedExpr().setResultsName("pre"))
            + Optional(":effect" + nestedExpr().setResultsName("eff"))
            + Suppress(")")
        )

        dur_action = Group(
            Suppress("(")
            + ":durative-action"
            + name.setResultsName("name")
            + ":parameters"
            + Suppress("(")
            + parameters
            + Suppress(")")
            + ":duration"
            + nestedExpr().setResultsName("duration")
            + ":condition"
            + nestedExpr().setResultsName("cond")
            + ":effect"
            + nestedExpr().setResultsName("eff")
            + Suppress(")")
        )

        # ----------------------------------------------------------

        # TODO: Extend to other tags
        tag_def = Group(
            Suppress("(") + ":tag" + "prettyprint" + QuotedString('"') + Suppress(")")
        ).setResultsName("inline")

        # ----------------------------------------------------------
        # HTN

        # nestedExpr in case we missed something
        inline_def = Group(
            Suppress("(")
            + ":inline"
            + (predicate | and_predicate | operation | nestedExpr()).setResultsName(
                "cond"
            )
            + (predicate | and_predicate | operation | nestedExpr()).setResultsName(
                "eff"
            )
            + Suppress(")")
        ).setResultsName("inline")

        subtask_def = Group(
            Suppress("(") + name.setResultsName("name") + parameters + Suppress(")")
        ).setResultsName("subtask")

        method = Group(
            Suppress("(")
            + ":method"
            + name.setResultsName("name")
            + ":precondition"
            + nestedExpr().setResultsName("pre")
            # TODO: Set order
            + Optional(":meta" + Suppress("(") + OneOrMore(tag_def) + Suppress(")"))
            + ":tasks"
            + Suppress("(")
            + Group(ZeroOrMore(inline_def | subtask_def)).setResultsName("subtasks")
            + Suppress(")")
            + Suppress(")")
        )

        task = Group(
            Suppress("(")
            + ":task"
            + name.setResultsName("name")
            + ":parameters"
            + Suppress("(")
            + parameters
            + Suppress(")")
            + Group(OneOrMore(method)).setResultsName("methods")
            + Suppress(")")
        )

        # ----------------------------------------------------------

        domain = (
            Suppress("(")
            + "define"
            + Suppress("(")
            + "domain"
            + name.setResultsName("name")
            + Suppress(")")
            + Optional(sec_requirements).setResultsName("features")
            + Optional(sec_types)
            + Optional(sec_constants)
            + Optional(sec_predicates)
            + Optional(sec_functions)
            + Group(ZeroOrMore(derived)).setResultsName("derived")
            + Group(ZeroOrMore(task)).setResultsName("tasks")
            + Group(ZeroOrMore(action | dur_action)).setResultsName("actions")
            + Suppress(")")
        )

        # ----------------------------------------------------------

        objects = OneOrMore(
            Group(Group(OneOrMore(name)) + Optional(Suppress("-") + name))
        ).setResultsName("objects")

        # htn_def = Group(
        #     Suppress("(")
        #     + ":htn"
        #     + Optional(":tasks" + nestedExpr().setResultsName("tasks"))
        #     + Optional(":ordering" + nestedExpr().setResultsName("ordering"))
        #     + Optional(":constraints" + nestedExpr().setResultsName("constraints"))
        #     + Suppress(")")
        # )

        metric = (Keyword("minimize") | Keyword("maximize")).setResultsName(
            "optimization"
        ) + (name | nestedExpr()).setResultsName("metric")

        goal = Group(
            Suppress("(")
            + ":tasks-goal"
            + ":tasks"
            + nestedExpr().setResultsName("goal")
            + Suppress(")")
        )

        # ----------------------------------------------------------

        problem = (
            Suppress("(")
            + "define"
            + Suppress("(")
            + "problem"
            + name.setResultsName("name")
            + Suppress(")")
            + Suppress("(")
            + ":domain"
            + name
            + Suppress(")")
            + Optional(sec_requirements)
            + Optional(Suppress("(") + ":objects" + objects + Suppress(")"))
            # + Optional(htn_def.setResultsName("htn"))
            + Suppress("(")
            + ":init"
            + ZeroOrMore(nestedExpr()).setResultsName("init")
            + Suppress(")")
            + Optional(goal)
            + Optional(Suppress("(") + ":metric" + metric + Suppress(")"))
            + Suppress(")")
        )

        # ----------------------------------------------------------

        domain.ignore(";" + restOfLine)
        problem.ignore(";" + restOfLine)

        self._domain = domain
        self._problem = problem
        self._parameters = parameters

    @property
    def domain(self):
        return self._domain

    @property
    def problem(self):
        return self._problem

    @property
    def parameters(self):
        return self._parameters


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

        # Refactor properties:
        self.problem: model.Problem = None
        self.types_map: Dict[str, "model.Type"] = {}
        self.object_type_needed: bool = False
        self.universal_assignments: Dict["model.Action", List[ParseResults]] = {}
        self.has_actions_cost: bool = False

    def _parse_exp(
        self,
        problem: model.Problem,
        act: typing.Optional[Union[model.Action, htn.Method]],
        types_map: Dict[str, model.Type],
        var: Dict[str, model.Variable],
        exp: Union[ParseResults, str],
        assignments: Dict[str, "model.Object"] = {},
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
                elif problem.has_fluent(exp[0]):  # fluent reference
                    f = problem.fluent(exp[0])
                    args = [solved.pop() for _ in exp[1:]]
                    solved.append(self._em.FluentExp(f, tuple(args)))
                elif exp[0] in assignments:  # quantified assignment variable
                    assert len(exp) == 1
                    solved.append(self._em.ObjectExp(assignments[exp[0]]))
                else:
                    raise up.exceptions.UPUnreachableCodeError
            else:
                if isinstance(exp, ParseResults):
                    if len(exp) == 0:  # empty precodition
                        solved.append(self._em.TRUE())
                    elif exp[0] == "-" and len(exp) == 2:  # unary minus
                        stack.append((var, exp, True))
                        stack.append((var, exp[1], False))
                    elif exp[0] in self._operators:  # n-ary operators
                        stack.append((var, exp, True))
                        for e in exp[1:]:
                            stack.append((var, e, False))
                    elif exp[0] in ["exists", "forall"]:  # quantifier operators
                        vars_string = " ".join(exp[1])
                        vars_res = self._pp_parameters.parseString(vars_string)
                        vars = {}
                        for g in vars_res["params"]:
                            t = types_map[g[1] if len(g) > 1 else "object"]
                            for o in g[0]:
                                vars[o] = model.Variable(o, t, self._env)
                        stack.append((vars, exp, True))
                        stack.append((vars, exp[2], False))
                    elif problem.has_fluent(exp[0]):  # fluent reference
                        stack.append((var, exp, True))
                        for e in exp[1:]:
                            stack.append((var, e, False))
                    elif exp[0] in assignments:  # quantified assignment variable
                        assert len(exp) == 1
                        stack.append((var, exp, True))
                    elif len(exp) == 1:  # expand an element inside brackets
                        stack.append((var, exp[0], False))
                    else:
                        raise SyntaxError(f"Not able to handle: {exp}")
                elif isinstance(exp, str):
                    if (
                        exp[0] == "?" and exp[1:] in var
                    ):  # variable in a quantifier expression
                        solved.append(self._em.VariableExp(var[exp[1:]]))
                    elif exp in assignments:  # quantified assignment variable
                        solved.append(self._em.ObjectExp(assignments[exp]))
                    elif exp[0] == "?":  # action parameter
                        assert act is not None
                        solved.append(self._em.ParameterExp(act.parameter(exp[1:])))
                    elif problem.has_fluent(exp):  # fluent
                        solved.append(self._em.FluentExp(problem.fluent(exp)))
                    elif problem.has_object(exp):  # object
                        solved.append(self._em.ObjectExp(problem.object(exp)))
                    else:  # number
                        n = Fraction(exp)
                        if n.denominator == 1:
                            solved.append(self._em.Int(n.numerator))
                        else:
                            solved.append(self._em.Real(n))
                else:
                    raise SyntaxError(f"Not able to handle: {exp}")
        assert len(solved) == 1  # sanity check
        return solved.pop()

    def _add_effect(
        self,
        problem: model.Problem,
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
                cond = self._parse_exp(problem, act, types_map, {}, exp[1], assignments)
                to_add.append((exp[2], cond))
            elif op == "not":
                exp = exp[1]
                eff = (
                    self._parse_exp(problem, act, types_map, {}, exp, assignments),
                    self._em.FALSE(),
                    cond,
                )
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "assign":
                eff = (
                    self._parse_exp(problem, act, types_map, {}, exp[1], assignments),
                    self._parse_exp(problem, act, types_map, {}, exp[2], assignments),
                    cond,
                )
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "increase":
                eff = (
                    self._parse_exp(problem, act, types_map, {}, exp[1], assignments),
                    self._parse_exp(problem, act, types_map, {}, exp[2], assignments),
                    cond,
                )
                act.add_increase_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "decrease":
                eff = (
                    self._parse_exp(problem, act, types_map, {}, exp[1], assignments),
                    self._parse_exp(problem, act, types_map, {}, exp[2], assignments),
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
                    self._parse_exp(problem, act, types_map, {}, exp, assignments),
                    self._em.TRUE(),
                    cond,
                )
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore

    def _add_condition(
        self,
        problem: model.Problem,
        act: model.DurativeAction,
        exp: Union[ParseResults, str],
        types_map: Dict[str, model.Type],
        vars: typing.Optional[Dict[str, model.Variable]] = None,
    ):
        to_add = [(exp, vars)]
        while to_add:
            exp, vars = to_add.pop(0)
            op = exp[0]
            if op == "and":
                for e in exp[1:]:
                    to_add.append((e, vars))
            elif op == "forall":
                vars_string = " ".join(exp[1])
                vars_res = self._pp_parameters.parseString(vars_string)
                if vars is None:
                    vars = {}
                for g in vars_res["params"]:
                    t = types_map[g[1] if len(g) > 1 else "object"]
                    for o in g[0]:
                        vars[o] = model.Variable(o, t, self._env)
                to_add.append((exp[2], vars))
            elif len(exp) == 3 and op == "at" and exp[1] == "start":
                cond = self._parse_exp(
                    problem, act, types_map, {} if vars is None else vars, exp[2]
                )
                if vars is not None:
                    cond = self._em.Forall(cond, *vars.values())
                act.add_condition(model.StartTiming(), cond)
            elif len(exp) == 3 and op == "at" and exp[1] == "end":
                cond = self._parse_exp(
                    problem, act, types_map, {} if vars is None else vars, exp[2]
                )
                if vars is not None:
                    cond = self._em.Forall(cond, *vars.values())
                act.add_condition(model.EndTiming(), cond)
            elif len(exp) == 3 and op == "over" and exp[1] == "all":
                t_all = model.OpenTimeInterval(model.StartTiming(), model.EndTiming())
                cond = self._parse_exp(
                    problem, act, types_map, {} if vars is None else vars, exp[2]
                )
                if vars is not None:
                    cond = self._em.Forall(cond, *vars.values())
                act.add_condition(t_all, cond)
            else:  # HPDL accept any exp, and considers (at start ...)
                cond = self._parse_exp(
                    problem,
                    act,
                    types_map,
                    {} if vars is None else vars,
                    exp,  # CHANGED
                )
                if vars is not None:
                    cond = self._em.Forall(cond, *vars.values())
                act.add_condition(model.StartTiming(), cond)
                # raise SyntaxError(f"Not able to handle: {exp}")

    def _add_timed_effects(
        self,
        problem: model.Problem,
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
                    problem,
                    act,
                    types_map,
                    universal_assignments,
                    eff[2],
                    timing=model.StartTiming(),
                    assignments=assignments,
                )
            elif len(eff) == 3 and op == "at" and eff[1] == "end":
                self._add_effect(
                    problem,
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
                    problem,
                    act,
                    types_map,
                    universal_assignments,
                    eff,  # CHANGED
                    timing=model.EndTiming(),
                    assignments=assignments,
                )
                # raise SyntaxError(f"Not able to handle: {eff}")

    def _parse_subtask(
        self,
        e,
        method: typing.Optional[htn.Method],
        problem: htn.HierarchicalProblem,
        types_map: Dict[str, model.Type],
    ) -> typing.Optional[htn.Subtask]:
        """Returns the Subtask corresponding to the given expression e or
        None if the expression cannot be interpreted as a subtask."""
        if len(e) == 0:
            return None
        task_name = e[0]
        print(f"task_name: {task_name}")
        task: Union[htn.Task, model.Action]
        if problem.has_task(task_name):
            task = problem.get_task(task_name)
        elif problem.has_action(task_name):
            task = problem.action(task_name)
        elif task_name == ":inline":
            # task = self._parse_inline(e, method, problem)
            return None
        else:
            return None
        assert isinstance(task, htn.Task) or isinstance(task, model.Action)

        if task_name == ":inline":
            # TODO: Get the parameters from the subtask
            return htn.Subtask(
                task,
            )

        # Remove types and '-' from the expression
        e = [part for part in e if part != "-" and part not in self.types_map]
        parameters = [
            self._parse_exp(problem, method, types_map, {}, param) for param in e[1:]
        ]
        return htn.Subtask(task, *parameters)

    def _parse_inline(
        self,
        e,
        method: typing.Optional[htn.Method],
        problem: htn.HierarchicalProblem,
    ) -> List[htn.Subtask]:
        inline_version = 0
        # inline_name = (method.name or "") + "_inline"
        inline_name = "_inline"
        # raise NotImplementedError("Inline methods are not supported yet.")
        # Find the first available name for the inline task
        # TODO: this is not very efficient; improve it
        while problem.has_action(f"{inline_name}_{inline_version}"):
            inline_version += 1

        # Let's build an action that corresponds to the inline task
        # Example dict: a [':action', 'AVATAR_MOVE_UP', ':parameters', [['a'], 'MovingAvatar'], ':precondition', ['and', ['can-move-up', '?a'], ['orientation-up', '?a']], ':effect', ['and', ['decrease', ['coordinate_x', '?a'], '1']]]
        action = OrderedDict()
        action["name"] = f"{inline_name}_{inline_version}"

        # action["params"] = [p.name for p in method.parameters]

        # The effect does not start with "and", we need to add it
        action["eff"] = [["and", e[2]]]

        return self._parse_action(
            action,
            problem,
            self.types_map,
            self.universal_assignments,
        )

    def _parse_subtasks(
        self,
        e,
        method: typing.Optional[htn.Method],
        problem: htn.HierarchicalProblem,
        types_map: Dict[str, model.Type],
    ) -> List[htn.Subtask]:
        """Returns the list of subtasks of the expression"""
        single_task = self._parse_subtask(e, method, problem, types_map)
        if single_task is not None:
            return [single_task]

        elif len(e) == 0:
            return []

        # In HPDL, we dont have the "and" keyword
        # elif e[0] == "and":
        #     return [
        #         subtask
        #         for e2 in e[1:]
        #         for subtask in self._parse_subtasks(e2, method, problem, types_map)
        #     ]
        elif len(e) >= 1:
            return [
                subtask
                for e2 in e[1:]
                for subtask in self._parse_subtasks(e2, method, problem, types_map)
            ]
        else:
            raise SyntaxError(f"Could not parse the subtasks list: {e}")

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

    # TODO Hay que modificar y añadir varias cosas en parse_problem, ver más abajo.
    # def _parse_action(
    #     self,
    #     a,
    #     problem: model.Problem,
    #     types_map: Dict[str, model.Type],
    #     universal_assignments: Dict["model.Action", List[ParseResults]],
    # ):
    #     """Parses an action from the domain and adds it to the problem"""
    #     n = a["name"]
    #     a_params = OrderedDict()
    #     for g in a.get("params", []):
    #         t = types_map[g[1] if len(g) > 1 else "object"]
    #         for p in g[0]:
    #             a_params[p] = t
    #     if "duration" in a:
    #         dur_act = model.DurativeAction(n, a_params, self._env)
    #         dur = a["duration"][0]
    #         if dur[0] == "=":
    #             dur.pop(0)
    #             dur.pop(0)
    #             dur_act.set_fixed_duration(
    #                 self._parse_exp(problem, dur_act, types_map, {}, dur)
    #             )
    #         elif dur[0] == "and":
    #             upper = None
    #             lower = None
    #             for j in range(1, len(dur)):
    #                 if dur[j][0] == ">=" and lower is None:
    #                     dur[j].pop(0)
    #                     dur[j].pop(0)
    #                     lower = self._parse_exp(problem, dur_act, types_map, {}, dur[j])
    #                 elif dur[j][0] == "<=" and upper is None:
    #                     dur[j].pop(0)
    #                     dur[j].pop(0)
    #                     upper = self._parse_exp(problem, dur_act, types_map, {}, dur[j])
    #                 else:
    #                     raise SyntaxError(
    #                         f"Not able to handle duration constraint of action {n}"
    #                     )
    #             if lower is None or upper is None:
    #                 raise SyntaxError(
    #                     f"Not able to handle duration constraint of action {n}"
    #                 )
    #             d = model.ClosedDurationInterval(lower, upper)
    #             dur_act.set_duration_constraint(d)
    #         else:
    #             raise SyntaxError(
    #                 f"Not able to handle duration constraint of action {n}"
    #             )
    #         cond = a["cond"][0]
    #         self._add_condition(problem, dur_act, cond, types_map)
    #         eff = a["eff"][0]
    #         self._add_timed_effects(
    #             problem, dur_act, types_map, universal_assignments, eff
    #         )
    #         problem.add_action(dur_act)
    #         self.has_actions_cost = (
    #             self.has_actions_cost and self._durative_action_has_cost(dur_act)
    #         )
    #         return dur_act
    #     else:
    #         act = model.InstantaneousAction(n, a_params, self._env)
    #         if "pre" in a:
    #             act.add_precondition(
    #                 self._parse_exp(problem, act, types_map, {}, a["pre"][0])
    #             )
    #         if "eff" in a:
    #             self._add_effect(
    #                 problem, act, types_map, universal_assignments, a["eff"][0]
    #             )
    #         problem.add_action(act)
    #         # Do we need to do it here? it comes from _parse_problem method
    #         self.has_actions_cost = (
    #             self.has_actions_cost and self._instantaneous_action_has_cost(act)
    #         )
    #         return act

    # TODO: Here start the refactoring of the code
    def _build_problem(self, name: str, features: List[str]) -> model.Problem:
        features = set(features)
        if ":hierarchy" in features or ":htn-expansion" in features:
            return htn.HierarchicalProblem(
                name,
                self._env,
                initial_defaults={self._tm.BoolType(): self._em.FALSE()},
            )

        return model.Problem(
            name,
            self._env,
            initial_defaults={self._tm.BoolType(): self._em.FALSE()},
        )

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
        params = OrderedDict()
        for g in params:
            param_type = self.types_map[g[1] if len(g) > 1 else "object"]
            for param_name in g[0]:
                params[param_name] = param_type
        return params

    def _parse_function(self, func: OrderedDict) -> model.Fluent:
        name = func[0]
        params = self._parse_params(func[1])
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

    def _parse_method(self, method: OrderedDict):
        pass

    def _build_method(
        self,
    ) -> htn.Method:
        pass

    def _parse_exp_str(
        self,
        var: Dict[str, model.Variable],
        exp: str,
        assignments: Dict[str, "model.Object"] = {},
    ) -> model.FNode:
        if exp[0] == "?" and exp[1:] in var:  # variable in a quantifier expression
            return self._em.VariableExp(var[exp[1:]])
        elif exp in assignments:  # quantified assignment variable
            return self._em.ObjectExp(assignments[exp])
        elif exp[0] == "?":  # action parameter
            return self._em.ParameterExp(
                model.Parameter(exp[1:], self._tm.ObjectType())
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
    ) -> Tuple[Union[model.FNode, None], Union[List[Tuple[Any, Any, Any]], None]]:
        if len(exp) == 0:  # empty precodition
            return self._em.TRUE(), None
        elif exp[0] == "-" and len(exp) == 2:  # unary minus
            return None, [(var, exp, True), (var, exp[1], False)]
        elif exp[0] in self._operators:  # n-ary operators
            res = [(var, exp, True)]
            for e in exp[1:]:
                res.append((var, e, False))
            return None, res
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
        # parameters: typing.Optional[Union[model.Action, htn.Method]],
        var: Dict[str, model.Variable],
        exp: Union[ParseResults, str],
        assignments: Dict[str, "model.Object"] = {},
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
                    args = [solved.pop() for _ in exp[1:]]
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
                    node = self._parse_exp_str(var, exp, assignments)
                    solved.append(node)
                else:
                    raise SyntaxError(f"Not able to handle: {exp}")
        assert len(solved) == 1  # sanity check
        return solved.pop()

    def _parse_effect(
        self,
        act: Union[model.InstantaneousAction, model.DurativeAction],
        universal_assignments: typing.Optional[
            Dict["model.Action", List[ParseResults]]
        ],
        exp: Union[ParseResults, str],
        cond: Union[model.FNode, bool] = True,
        timing: typing.Optional[model.Timing] = None,
        assignments: Dict[str, "model.Object"] = {},
    ) -> List[typing.Any]:
        to_add = [(exp, cond)]
        res = []
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
                cond = self._parse_exp(
                    self.problem, act, self.types_map, {}, exp[1], assignments
                )
                to_add.append((exp[2], cond))
            elif op == "not":
                exp = exp[1]
                eff = (
                    self._parse_exp(
                        self.problem, act, self.types_map, {}, exp, assignments
                    ),
                    self._em.FALSE(),
                    cond,
                )
                res.append(eff)
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "assign":
                eff = (
                    self._parse_exp(
                        self.problem, act, self.types_map, {}, exp[1], assignments
                    ),
                    self._parse_exp(
                        self.problem, act, self.types_map, {}, exp[2], assignments
                    ),
                    cond,
                )
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "increase":
                eff = (
                    self._parse_exp(
                        self.problem, act, self.types_map, {}, exp[1], assignments
                    ),
                    self._parse_exp(
                        self.problem, act, self.types_map, {}, exp[2], assignments
                    ),
                    cond,
                )
                act.add_increase_effect(*eff if timing is None else (timing, *eff))  # type: ignore
            elif op == "decrease":
                eff = (
                    self._parse_exp(
                        self.problem, act, self.types_map, {}, exp[1], assignments
                    ),
                    self._parse_exp(
                        self.problem, act, self.types_map, {}, exp[2], assignments
                    ),
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
                    self._parse_exp(
                        self.problem, act, self.types_map, {}, exp, assignments
                    ),
                    self._em.TRUE(),
                    cond,
                )
                act.add_effect(*eff if timing is None else (timing, *eff))  # type: ignore

    def _parse_subtask(self, subtask: OrderedDict):
        pass

    def _build_subtask(
        self,
    ) -> htn.Subtask:
        pass

    def _build_durative_action(
        self,
        name: str,
        params: OrderedDict,
        duration: ParseResults,
        cond: ParseResults,
        eff: ParseResults,
    ) -> model.DurativeAction:
        dur_act = model.DurativeAction(name, params, self._env)
        if duration[0] == "=":
            duration.pop(0)
            duration.pop(0)
            dur_act.set_fixed_duration(
                self._parse_exp(self.problem, dur_act, self.types_map, {}, duration)
            )
        elif duration[0] == "and":
            upper = None
            lower = None
            for j in range(1, len(duration)):
                if duration[j][0] == ">=" and lower is None:
                    duration[j].pop(0)
                    duration[j].pop(0)
                    lower = self._parse_exp(
                        self.problem, dur_act, self.types_map, {}, duration[j]
                    )
                elif duration[j][0] == "<=" and upper is None:
                    duration[j].pop(0)
                    duration[j].pop(0)
                    upper = self._parse_exp(
                        self.problem, dur_act, self.types_map, {}, duration[j]
                    )
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
        # cond = action["cond"][0]
        self._add_condition(self.problem, dur_act, cond, self.types_map)
        # eff = action["eff"][0]
        self._add_timed_effects(
            self.problem, dur_act, self.types_map, self.universal_assignments, eff
        )
        # problem.add_action(dur_act)
        self.has_actions_cost = (
            self.has_actions_cost and self._durative_action_has_cost(dur_act)
        )
        return dur_act

    def _build_action(
        self,
        name: str,
        params: OrderedDict,
        durative: bool,
        pre: List[str],
        eff: List[model.Fluent],
        duration: typing.Optional[List[str]],
    ) -> model.Action:
        if durative:
            return self._build_durative_action(name, params)

        act = model.InstantaneousAction(name, params, self._env)
        if len(pre):
            act.add_precondition(
                self._parse_exp(self.problem, act, self.types_map, {}, pre)
            )
        if len(eff):
            self._add_effect(
                self.problem, act, self.types_map, self.universal_assignments, eff
            )
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
        """Parses an action from the domain and adds it to the problem"""
        action_name = action["name"]
        a_params = self._parse_params(action["params"])
        durative = False
        duration = []

        if "duration" in action:
            durative = True
            duration = action["duration"]

        res = OrderedDict()

        res["name"] = action_name
        res["params"] = a_params
        res["durative"] = durative
        res["duration"] = duration

        if "pre" in action:
            res["pre"] = self._parse_exp(
                problem, act, self.types_map, {}, action["pre"][0]
            )
        if "eff" in action:
            self._add_effect(
                problem,
                act,
                self.types_map,
                self.universal_assignments,
                action["eff"][0],
            )

        act = model.InstantaneousAction(action_name, a_params, self._env)
        pre = [self._parse_exp(problem, act, self.types_map, {}, action["pre"][0])]

        if "pre" in action:
            act.add_precondition(
                self._parse_exp(problem, act, self.types_map, {}, action["pre"][0])
            )
        if "eff" in action:
            self._add_effect(
                problem,
                act,
                self.types_map,
                self.universal_assignments,
                action["eff"][0],
            )
        # problem.add_action(act)
        # Do we need to do it here? it comes from _parse_problem method
        self.has_actions_cost = (
            self.has_actions_cost and self._instantaneous_action_has_cost(act)
        )
        return act

    def _build_action(
        self,
        name: str,
        params: List[model.Parameter],
        preconditions: List[typing.Any],  # TODO: Decide type
        effects: List[typing.Any],  # TODO: Decide type
    ) -> model.Action:
        return model.Action(name, params, self._env)

    # _________________________________________________________

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

        # TODO Pensar si necesitamos distinguir en "features" que estamos parseando HPDL
        # Init properties of the problem
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

        # TODO   DERIVED PREDICATES Hay que añadir problem.add_derived_predicate() y esto tendría que ser en una
        #       nueva subclase de HierarchicalProblem, que podríamos llamar HPDLProblem

        # TODO  Las funciones pddl se gestionan y se añaden como un fluent especial "con un tipo real".
        #       un fluent en el upfmodel es [name, type, signature], donde signature son los parámetros.
        #       habría que
        #               o bien cambiar la definición de la clase fluent en el upfmodel
        #               o bien crear una subclase de HierarchicalProblem, que sea, HPDLProblem, CREO  que esto es lo ideal
        #                 porque un HPDLProblem tiene los mismos atributos que un HierarchicalProblem, y un conjunto
        #                 adicional como derived predicates y functions que se implementan con python.
        # TODO AÑADIR la funcion problem.add_pythonfunction(f)
        for f in domain_res.get("functions", []):
            func = self._parse_function(f)
            self.problem.add_fluent(func)

        # TODO Comprobar las constantes, que no deberían  dar problema
        for c in domain_res.get("constants", []):
            objects = self._parse_constant(c)
            for o in objects:
                self.problem.add_object(o)

        for task in domain_res.get("tasks", []):
            assert isinstance(self.problem, htn.HierarchicalProblem)
            task_model = self._build_task(task)
            self.problem.add_task(task_model)

        for a in domain_res.get("actions", []):
            self._parse_action(a)
            self._parse_action(
                a,
                self.problem,
                self.types_map,
                self.universal_assignments,
            )

        # Methods are defined inside tasks;
        # we need to parse them after all tasks and actions have been defined
        # because _parse_subtasks() needs to be able to find them.
        for task in domain_res.get("tasks", []):
            for method in task.get("methods", []):
                # assert isinstance(problem, htn.HierarchicalProblem)
                method_name = f'{task["name"]}-{method["name"]}'  # Methods names are
                # not unique across tasks

                subtasks = []
                subtasks_params = OrderedDict()

                for subs in method.get("subtasks", []):
                    subs = self._parse_subtasks(
                        subs, None, self.problem, self.types_map
                    )
                    for s in subs:
                        subtasks.append(s)
                        subtasks_params.append(s.parameters)
                        # method_model.add_subtask(s)

                task_model = self.problem.get_task(task["name"])
                task_params = OrderedDict(
                    {p.name: p.type for p in task_model.parameters}
                )

                method_model = htn.Method(method_name, task_params + subtasks_params)
                method_model.set_task(task_model)
                for s in subtasks:
                    method_model.add_subtask(s)

                method_preconditions = method.get("preconditions", [])
                for pre in method_preconditions:
                    method_model.add_precondition(pre)

                # for ord_subs in m.get("tasks", []):
                #     ord_subs = self._parse_subtasks(ord_subs, method, problem, types_map)
                #     for s in ord_subs:
                #         method.add_subtask(s)
                #     method.set_ordered(*ord_subs)
                self.problem.add_method(method_model)

        if problem_filename is not None:
            problem_res = self._pp_problem.parseFile(problem_filename)

            self.problem.name = problem_res["name"]

            for g in problem_res.get("objects", []):
                t = self.types_map[g[1] if len(g) > 1 else "object"]
                for o in g[0]:
                    self.problem.add_object(model.Object(o, t, self.problem.env))

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
                                self.problem,
                                action,
                                self.types_map,
                                None,
                                eff[2],
                                assignments=assignments,
                            )
                        elif isinstance(action, model.DurativeAction):
                            self._add_timed_effects(
                                self.problem,
                                action,
                                self.types_map,
                                None,
                                eff[2],
                                assignments=assignments,
                            )
                        else:
                            raise NotImplementedError

            tasknet = problem_res.get("htn", None)
            if tasknet is not None:
                assert isinstance(self.problem, htn.HierarchicalProblem)
                tasks = self._parse_subtasks(
                    tasknet["tasks"][0], None, self.problem, self.types_map
                )
                for task in tasks:
                    self.problem.task_network.add_subtask(task)
                if len(tasknet["ordering"][0]) != 0:
                    raise SyntaxError(
                        "Ordering not supported in the initial task network"
                    )
                if len(tasknet["constraints"][0]) != 0:
                    raise SyntaxError(
                        "Constraints not supported in the initial task network"
                    )

            for i in problem_res.get("init", []):
                if i[0] == "=":
                    self.problem.set_initial_value(
                        self._parse_exp(self.problem, None, self.types_map, {}, i[1]),
                        self._parse_exp(self.problem, None, self.types_map, {}, i[2]),
                    )
                elif (
                    len(i) == 3 and i[0] == "at" and i[1].replace(".", "", 1).isdigit()
                ):
                    ti = model.StartTiming(Fraction(i[1]))
                    va = self._parse_exp(self.problem, None, self.types_map, {}, i[2])
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
                        self._parse_exp(self.problem, None, self.types_map, {}, i),
                        self._em.TRUE(),
                    )

            if "goal" in problem_res:
                self.problem.add_goal(
                    self._parse_exp(
                        self.problem, None, self.types_map, {}, problem_res["goal"][0]
                    )
                )
            elif not isinstance(self.problem, htn.HierarchicalProblem):
                raise SyntaxError("Missing goal section in problem file.")

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
                    metric_exp = self._parse_exp(
                        self.problem, None, self.types_map, {}, metric
                    )
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
