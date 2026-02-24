# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver

from pyomo.environ import (
    SolverFactory,
    Objective,
    maximize,
    TerminationCondition,
    value,
    Var,
    Integers,
    Constraint,
    ConcreteModel,
)
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models
from pyomo.core.base.suffix import Suffix


def _select_first_available(solver_names):
    for name in solver_names:
        if SolverFactory(name).available(exception_flag=False):
            return name
    return None


# NOTE: We intentionally avoid `appsi_highs` here. GDPopt's subproblem clone
# attaches an IMPORT `dual` suffix, and APPSI's legacy wrapper will attempt to
# import duals whenever `model.dual.import_enabled()` is true. HiGHS does not
# provide valid duals for MIP/MILP in general, which can raise at load time.
# Additionally, APPSI HiGHS raises if asked to load solutions when no feasible
# solution exists (e.g., unbounded/infeasible), which breaks the unbounded test.
_MIP_SOLVER = _select_first_available(('appsi_highs', 'glpk', 'cbc'))
_NLP_SOLVER = 'ipopt'

_ENUMERATE_SOLVERS_AVAILABLE = _MIP_SOLVER is not None and SolverFactory(
    _NLP_SOLVER
).available(exception_flag=False)


def _disable_dual_suffix_import(solver, subproblem, subproblem_util_block):
    # GDPopt's subproblem clone always attaches an IMPORT `dual` suffix so OA
    # solvers can read duals for cut generation. Enumerate does not generate OA
    # cuts, and many MIP solvers (including HiGHS) will not provide valid duals
    # for MIP/MILP solves. Disable import to avoid NoDualsError during solution
    # loading.
    dual = getattr(subproblem, 'dual', None)
    if isinstance(dual, Suffix) and dual.import_enabled():
        dual.direction = Suffix.LOCAL


@unittest.skipUnless(
    _ENUMERATE_SOLVERS_AVAILABLE,
    'Required solvers not available (need appsi_highs/glpk/cbc + Ipopt for NLPs)',
)
class TestGDPoptEnumerate(unittest.TestCase):
    def test_solve_two_term_disjunction(self):
        m = models.makeTwoTermDisj()
        m.obj = Objective(expr=m.x, sense=maximize)

        results = SolverFactory('gdpopt.enumerate').solve(
            m,
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, 9)
        self.assertEqual(results.problem.upper_bound, 9)

        self.assertEqual(value(m.x), 9)
        self.assertTrue(value(m.d[0].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))

    def modify_two_term_disjunction(self, m):
        # Make first disjunct feasible
        m.a.setlb(0)
        # Discrete variable
        m.y = Var(domain=Integers, bounds=(2, 4))
        m.d[1].c3 = Constraint(expr=m.x <= 6)
        m.d[0].c2 = Constraint(expr=m.y + m.a - 5 <= 2)

        m.obj = Objective(expr=-m.x - m.y)

    def test_solve_GDP_iterate_over_discrete_variables(self):
        m = models.makeTwoTermDisj()
        self.modify_two_term_disjunction(m)

        results = SolverFactory('gdpopt.enumerate').solve(
            m,
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            force_subproblem_nlp=True,
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 6)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, -11)
        self.assertEqual(results.problem.upper_bound, -11)

        self.assertEqual(value(m.x), 9)
        self.assertEqual(value(m.y), 2)
        self.assertTrue(value(m.d[0].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))

    def test_solve_GDP_do_not_iterate_over_discrete_variables(self):
        m = models.makeTwoTermDisj()
        self.modify_two_term_disjunction(m)

        results = SolverFactory('gdpopt.enumerate').solve(
            m,
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, -11)
        self.assertEqual(results.problem.upper_bound, -11)

        self.assertEqual(value(m.x), 9)
        self.assertEqual(value(m.y), 2)
        self.assertTrue(value(m.d[0].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))

    def test_solve_GDP_iterate_over_Boolean_variables(self):
        m = models.makeLogicalConstraintsOnDisjuncts()

        results = SolverFactory('gdpopt.enumerate').solve(
            m,
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            force_subproblem_nlp=True,
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 16)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, 8)
        self.assertEqual(results.problem.upper_bound, 8)

        self.assertTrue(value(m.d[2].indicator_var))
        self.assertTrue(value(m.d[3].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))
        self.assertFalse(value(m.d[4].indicator_var))
        self.assertEqual(value(m.x), 8)
        # We don't know what values they take, but they have to be different
        self.assertNotEqual(value(m.Y[1]), value(m.Y[2]))

    def test_solve_GDP_do_not_iterate_over_Boolean_variables(self):
        m = models.makeLogicalConstraintsOnDisjuncts()

        results = SolverFactory('gdpopt.enumerate').solve(
            m,
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 4)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, 8)
        self.assertEqual(results.problem.upper_bound, 8)

        self.assertTrue(value(m.d[2].indicator_var))
        self.assertTrue(value(m.d[3].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))
        self.assertFalse(value(m.d[4].indicator_var))
        self.assertEqual(value(m.x), 8)
        # We don't know what values they take, but they have to be different
        self.assertNotEqual(value(m.Y[1]), value(m.Y[2]))

    def test_stop_at_iteration_limit(self):
        m = models.makeLogicalConstraintsOnDisjuncts()

        results = SolverFactory('gdpopt.enumerate').solve(
            m,
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            iterlim=4,
            force_subproblem_nlp=True,
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 4)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxIterations
        )

    def test_unbounded_GDP(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
        m.o = Objective(expr=m.z)

        results = SolverFactory('gdpopt.enumerate').solve(
            m,
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            mip_solver_args={'load_solutions': False},
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 1)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.unbounded
        )
        self.assertEqual(results.problem.lower_bound, -float('inf'))
        self.assertEqual(results.problem.upper_bound, -float('inf'))


@unittest.skipUnless(
    _ENUMERATE_SOLVERS_AVAILABLE,
    'Required solvers not available (need appsi_highs/glpk/cbc + Ipopt for NLPs)',
)
class TestGDPoptEnumerate_ipopt_tests(unittest.TestCase):
    def test_infeasible_GDP(self):
        m = models.make_infeasible_gdp_model()

        results = SolverFactory('gdpopt.enumerate').solve(
            m,
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertEqual(results.problem.lower_bound, float('inf'))

    def test_algorithm_specified_to_solve(self):
        m = models.twoDisj_twoCircles_easy()

        results = SolverFactory('gdpopt').solve(
            m,
            algorithm='enumerate',
            mip_solver=_MIP_SOLVER,
            nlp_solver=_NLP_SOLVER,
            tee=True,
            call_before_subproblem_solve=_disable_dual_suffix_import,
        )

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertAlmostEqual(results.problem.lower_bound, 9)
        self.assertAlmostEqual(results.problem.upper_bound, 9)

        self.assertAlmostEqual(value(m.x), 2)
        self.assertAlmostEqual(value(m.y), 7)
        self.assertTrue(value(m.upper_circle.indicator_var))
        self.assertFalse(value(m.lower_circle.indicator_var))
