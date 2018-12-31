"""This module provides functions for cut generation."""
from __future__ import division

from math import copysign, fabs

from pyomo.contrib.gdpopt.util import time_code, constraints_in_True_disjuncts
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc
from pyomo.core import (Block, ConstraintList, NonNegativeReals, VarList,
                        minimize, value)
from pyomo.core.base.symbolic import differentiate
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet


def add_subproblem_cuts(subprob_result, solve_data, config):
    if config.strategy == "LOA":
        return add_outer_approximation_cuts(subprob_result, solve_data, config)
    elif config.strategy == "GLOA":
        return add_affine_cuts(subprob_result, solve_data, config)


def add_outer_approximation_cuts(nlp_result, solve_data, config):
    """Add outer approximation cuts to the linear GDP model."""
    with time_code(solve_data.timing, 'OA cut generation'):
        m = solve_data.linear_GDP
        GDPopt = m.GDPopt_utils
        sign_adjust = -1 if solve_data.objective_sense == minimize else 1

        # copy values over
        for var, val in zip(GDPopt.variable_list, nlp_result.var_values):
            if val is not None and not var.fixed:
                var.value = val

        # TODO some kind of special handling if the dual is phenomenally small?
        config.logger.debug('Adding OA cuts.')

        counter = 0
        if not hasattr(GDPopt, 'jacobians'):
            GDPopt.jacobians = ComponentMap()
        for constr, dual_value in zip(GDPopt.constraint_list,
                                      nlp_result.dual_values):
            if dual_value is None or constr.body.polynomial_degree() in (1, 0):
                continue

            # Determine if the user pre-specified that OA cuts should not be
            # generated for the given constraint.
            parent_block = constr.parent_block()
            ignore_set = getattr(parent_block, 'GDPopt_ignore_OA', None)
            config.logger.debug('Ignore_set %s' % ignore_set)
            if (ignore_set and (constr in ignore_set or
                                constr.parent_component() in ignore_set)):
                config.logger.debug(
                    'OA cut addition for %s skipped because it is in '
                    'the ignore set.' % constr.name)
                continue

            config.logger.debug(
                "Adding OA cut for %s with dual value %s"
                % (constr.name, dual_value))

            # Cache jacobians
            jacobians = GDPopt.jacobians.get(constr, None)
            if jacobians is None:
                constr_vars = list(EXPR.identify_variables(constr.body))
                jac_list = differentiate(constr.body, wrt_list=constr_vars)
                jacobians = ComponentMap(zip(constr_vars, jac_list))
                GDPopt.jacobians[constr] = jacobians

            # Create a block on which to put outer approximation cuts.
            oa_utils = parent_block.component('GDPopt_OA')
            if oa_utils is None:
                oa_utils = parent_block.GDPopt_OA = Block(
                    doc="Block holding outer approximation cuts "
                    "and associated data.")
                oa_utils.GDPopt_OA_cuts = ConstraintList()
                oa_utils.GDPopt_OA_slacks = VarList(
                    bounds=(0, config.max_slack),
                    domain=NonNegativeReals, initialize=0)

            # TODO add OA cut corresponding to objective

            oa_cuts = oa_utils.GDPopt_OA_cuts
            slack_var = oa_utils.GDPopt_OA_slacks.add()
            oa_cuts.add(
                expr=copysign(1, sign_adjust * dual_value) * (
                    value(constr.body) + sum(
                        value(jacobians[var]) * (var - value(var))
                        for var in jacobians)) + slack_var <= 0)
            counter += 1

        config.logger.info('Added %s OA cuts' % counter)


def add_affine_cuts(nlp_result, solve_data, config):
    m = solve_data.linear_GDP
    config.logger.info("Adding affine cuts.")
    GDPopt = m.GDPopt_utils
    for var, val in zip(GDPopt.variable_list, nlp_result.var_values):
        if val is not None and not var.fixed:
            var.value = val

    for constr in constraints_in_True_disjuncts(m, config):
        # for constr in GDPopt.working_nonlinear_constraints:

        if constr.body.polynomial_degree() in (1, 0):
            continue

        # if constr.body.polynomial_degree() in (1, 0):
        #     continue

        # TODO check that constraint is on active Disjunct

        vars_in_constr = list(
            EXPR.identify_variables(constr.body))
        if any(var.value is None for var in vars_in_constr):
            continue  # a variable has no values

        # mcpp stuff
        mc_eqn = mc(constr.body)
        ccSlope = mc_eqn.subcc()
        cvSlope = mc_eqn.subcv()
        ccStart = mc_eqn.concave()
        cvStart = mc_eqn.convex()
        ub_int = min(constr.upper, mc_eqn.upper()) if constr.has_ub() else mc_eqn.upper()
        lb_int = max(constr.lower, mc_eqn.lower()) if constr.has_lb() else mc_eqn.lower()

        parent_block = constr.parent_block()
        # Create a block on which to put outer approximation cuts.
        aff_utils = parent_block.component('GDPopt_aff')
        if aff_utils is None:
            aff_utils = parent_block.GDPopt_aff = Block(
                doc="Block holding affine constraints")
            aff_utils.GDPopt_aff_cons = ConstraintList()
        aff_cuts = aff_utils.GDPopt_aff_cons
        concave_cut = sum(ccSlope[var] * (var - var.value)
                          for var in vars_in_constr
                          ) + ccStart >= lb_int
        convex_cut = sum(cvSlope[var] * (var - var.value)
                         for var in vars_in_constr
                         ) + cvStart <= ub_int
        aff_cuts.add(expr=concave_cut)
        aff_cuts.add(expr=convex_cut)


def add_integer_cut(var_values, solve_data, config, feasible=False):
    """Add an integer cut to the linear GDP model."""
    m = solve_data.linear_GDP
    GDPopt = m.GDPopt_utils
    var_value_is_one = ComponentSet()
    var_value_is_zero = ComponentSet()
    for var, val in zip(GDPopt.variable_list, var_values):
        if not var.is_binary():
            continue
        if var.fixed:
            if val is not None and var.value != val:
                # val needs to be None or match var.value. Otherwise, we have a
                # contradiction.
                raise ValueError(
                    "Fixed variable %s has value %s != "
                    "provided value of %s." % (var.name, var.value, val))
            val = var.value
        # TODO we can also add a check to skip binary variables that are not an
        # indicator_var on disjuncts.
        if fabs(val - 1) <= config.integer_tolerance:
            var_value_is_one.add(var)
        elif fabs(val) <= config.integer_tolerance:
            var_value_is_zero.add(var)
        else:
            raise ValueError(
                'Binary %s = %s is not 0 or 1' % (var.name, val))

    if not (var_value_is_one or var_value_is_zero):
        # if no remaining binary variables, then terminate algorithm.
        config.logger.info(
            'Adding integer cut to a model without binary variables. '
            'Model is now infeasible.')
        if solve_data.objective_sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
        return False

    int_cut = (sum(1 - v for v in var_value_is_one) +
               sum(v for v in var_value_is_zero)) >= 1

    if not feasible:
        config.logger.info('Adding integer cut')
        GDPopt.integer_cuts.add(expr=int_cut)
    else:
        backtracking_enabled = (
            "disabled" if GDPopt.no_backtracking.active else "allowed")
        config.logger.info(
            'Registering explored configuration. '
            'Backtracking is currently %s.' % backtracking_enabled)
        GDPopt.no_backtracking.add(expr=int_cut)
