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

        variable = Optional(Suppress("?")) + name  # Optional for goals
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
            + OneOrMore(name_list).setResultsName("constants")
            + Suppress(")")
        )

        sec_predicates = (
            Suppress("(")
            + ":predicates"
            + Group(OneOrMore(predicate)).setResultsName("predicates")
            + Suppress(")")
        )

        # Python function
        # Fails in curly brackets inside python_code, but as far as I know they
        # are only used in dictionaries who were introduced in Python 3.7, which
        # SIADEX does not support
        python_code = pyparsing.Regex("{(.|\n)*?}")

        # Functions can specify -number type
        sec_functions = (
            Suppress("(")
            + ":functions"
            + Group(
                OneOrMore(
                    Group(  # Added group
                        predicate
                        + Optional(Suppress("- number"))
                        + Optional(python_code.setResultsName("python"))
                    )
                )
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
            + nestedExpr().setResultsName("pre")  # CHANGED: PDDLReader uses "cond"
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
            + nestedExpr().setResultsName("cond")
            + nestedExpr().setResultsName("eff")
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
            + Optional(":meta" + Suppress("(") + OneOrMore(tag_def) + Suppress(")"))
            + ":tasks"
            + Suppress("(")
            + Group(
                ZeroOrMore(
                    Group(
                        # Ordering is defined with [] or ()
                        Optional("[", default="(").setResultsName("ordering")
                        + OneOrMore(inline_def | subtask_def)
                        + Suppress(Optional("]"))
                    )
                )
            ).setResultsName("subtasks")
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
            + Suppress("(")
            + Group(
                ZeroOrMore(
                    Group(
                        # Ordering is defined with [] or ()
                        Optional("[", default="(").setResultsName("ordering")
                        + OneOrMore(subtask_def)
                        + Suppress(Optional("]"))
                    )
                )
            ).setResultsName("subtasks")
            + Suppress(")")
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
            + Suppress("(")
            + ":init"
            + ZeroOrMore(nestedExpr()).setResultsName("init")
            + Suppress(")")
            + Optional(
                goal.setResultsName("goal")
            )  # TODO: It isn't optional in HPDL, right?
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
