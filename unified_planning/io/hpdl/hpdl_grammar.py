import pyparsing
from pyparsing import (
    Group,
    Keyword,
    OneOrMore,
    Optional,
    QuotedString,
    SkipTo,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    alphas,
    nestedExpr,
    restOfLine,
)

if pyparsing.__version__ < "3.0.0":
    from pyparsing import oneOf as one_of
else:
    from pyparsing import one_of


class HPDLGrammar:
    def __init__(self):
        name = Word(alphas, alphanums + "_" + "-")

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
            + ZeroOrMore(name_list).setResultsName("constants")
            + Suppress(")")
        )

        sec_predicates = (
            Suppress("(")
            + ":predicates"
            + Group(OneOrMore(predicate)).setResultsName("predicates")
            + Suppress(")")
        )

        # Fails if curly brackets inside python_function, but as far as I know they
        # are only used in dictionaries who were introduced in Python 3.7, which
        # SIADEX does not support
        python_function = (
            Suppress("{") + SkipTo("}") + Suppress("}")
        )
        # Functions can specify -number type
        sec_functions = (
            Suppress("(")
            + ":functions"
            + Group(
                OneOrMore(
                    Group(
                        predicate
                        + Optional(Suppress("- number"))
                        + Optional(python_function.setResultsName("code"))
                    )
                )
            ).setResultsName("functions")
            + Suppress(")")
        )

        # derived evaluates a logical expression with given arguments
        # planner changes occurrences of "pre" with "exp"
        derived = Group(
            Suppress("(")
            + ":derived"
            + predicate
            + nestedExpr().setResultsName("exp")
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
        # HPDL

        inline_def = Group(
            Suppress("(")
            + ":inline"
            + nestedExpr().setResultsName("cond")
            + nestedExpr().setResultsName("eff")
            + Suppress(")")
        ).setResultsName("inline")

        tag_def = Group(
            Suppress("(") + ":tag" + "prettyprint" + QuotedString('"') + Suppress(")")
        ).setResultsName("inline")

        # ----------------------------------------------------------
        # HTN

        subtask_def = Group(
            Suppress("(") + name.setResultsName("name") + parameters + Suppress(")")
        ).setResultsName("subtask")

        time_rest_subt = Group(
            Suppress("(")
            + nestedExpr().setResultsName("time_exp")
            + subtask_def
            + Suppress(")")
        )

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
                        # Ordering is defined with parallel [] or sequential in absence
                        # Can't use () as it will collide with time constraints
                        Optional("[", default="(").setResultsName("ordering")
                        + OneOrMore(inline_def | subtask_def | time_rest_subt)
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
        # Problem

        objects = OneOrMore(
            Group(Group(OneOrMore(name)) + Optional(Suppress("-") + name))
        ).setResultsName("objects")

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
                        # Ordering is defined with parallel [] or sequential ()
                        Optional("[", default="(").setResultsName("ordering")
                        + OneOrMore(subtask_def)
                        + Suppress(Optional("]"))
                    )
                )
            ).setResultsName("subtasks")
            + Suppress(")")
            + Suppress(")")
        )

        # Time customization
        sec_customization = Group(
            Suppress("(")
            + ":customization"
            + OneOrMore(nestedExpr())
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
            + Optional(sec_customization.setResultsName("customization"))
            + Optional(sec_requirements)
            + Optional(Suppress("(") + ":objects" + objects + Suppress(")"))
            + Suppress("(")
            + ":init"
            + ZeroOrMore(nestedExpr()).setResultsName("init")
            + Suppress(")")
            + Optional(
                goal.setResultsName("goal")
            )
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
