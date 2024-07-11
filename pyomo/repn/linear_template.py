#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from copy import deepcopy
from itertools import chain

from pyomo.common.collections import ComponentSet
from pyomo.common.numeric_types import native_types

import pyomo.core.expr as expr
import pyomo.repn.linear as linear
import pyomo.repn.util as util

from pyomo.core.base import NumericLabeler
from pyomo.core.expr import ExpressionType
from pyomo.repn.linear import LinearRepn

_CONSTANT = util.ExprType.CONSTANT
_VARIABLE = util.ExprType.VARIABLE
_LINEAR = util.ExprType.LINEAR

code_type = deepcopy.__class__


class LinearTemplateRepn(LinearRepn):
    __slots__ = ("linear_sum",)

    def __init__(self):
        super().__init__()
        self.linear_sum = []

    def __str__(self):
        return (
            f"LinearTemplateRepn(mult={self.multiplier}, const={self.constant}, "
            f"linear={self.linear}, linear_sum={self.linear_sum}, "
            f"nonlinear={self.nonlinear})"
        )

    def walker_exitNode(self):
        if self.nonlinear is not None:
            return _GENERAL, self
        elif self.linear or self.linear_sum:
            return _LINEAR, self
        else:
            return _CONSTANT, self.multiplier * self.constant

    def duplicate(self):
        ans = super().duplicate()
        ans.linear_sum = [(r[0].duplicate(),) + r[1:] for r in self.linear_sum]
        return ans

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use a QuadraticRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

        """
        super().append(other)
        _type, other = other
        if getattr(other, 'linear_sum', None):
            mult = other.multiplier
            if not mult:
                return
            if mult != 1:
                for term in other.linear_sum:
                    term[0].multiplier *= mult
            self.linear_sum.extend(other.linear_sum)

    def _build_evaluator(
        self,
        smap,
        expr_cache,
        multiplier,
        repetitions,
        remove_fixed_vars,
        check_duplicates,
    ):
        ans = []
        multiplier *= self.multiplier
        constant = self.constant
        if constant.__class__ not in native_types or constant:
            constant *= multiplier
            if not repetitions or (
                constant.__class__ not in native_types and constant.is_expression_type()
            ):
                ans.append('const += ' + constant.to_string(smap=smap))
                constant = 0
            else:
                constant *= repetitions
        for k, coef in list(self.linear.items()):
            coef *= multiplier
            if coef.__class__ not in native_types and coef.is_expression_type():
                coef = coef.to_string(smap=smap)
            elif coef:
                coef = repr(coef)
            else:
                continue

            indent = ''
            if k in expr_cache:
                k = expr_cache[k]
                if k.__class__ not in native_types and k.is_expression_type():
                    ans.append('v = ' + k.to_string(smap=smap))
                    k = 'v'
                    if remove_fixed_vars:
                        ans.append('if v.__class__ is tuple:')
                        ans.append('    const += v[0] * {coef}')
                        ans.append('    v = None')
                        ans.append('else:')
                        indent = '    '
                    elif not check_duplicates:
                        # Directly substitute the expression into the
                        # 'linear[vid] = coef below
                        k = ans.pop()[4:]
            if check_duplicates:
                ans.append(indent + f'if {k} in linear:')
                ans.append(indent + f'    linear[{k}] += {coef}')
                ans.append(indent + 'else:')
                ans.append(indent + f'    linear[{k}] = {coef}')
            else:
                ans.append(indent + f'linear_indices.append({k})')
                ans.append(indent + f'linear_data.append({coef})')
        for subrepn, subindices, subsets in self.linear_sum:
            ans.extend(
                '    ' * i
                + f"for {','.join(smap.getSymbol(i) for i in _idx)} in "
                + (
                    _set.to_string(smap=smap)
                    if _set.is_expression_type()
                    else smap.getSymbol(_set)
                )
                + ":"
                for i, (_idx, _set) in enumerate(zip(subindices, subsets))
            )
            try:
                subrep = 1
                for _set in subsets:
                    subrep *= len(_set)
            except:
                subrep = 0
            subans, subconst = subrepn._build_evaluator(
                smap,
                expr_cache,
                multiplier,
                repetitions * subrep,
                remove_fixed_vars,
                check_duplicates,
            )
            indent = '    ' * (len(subsets))
            ans.extend(indent + line for line in subans)
            constant += subconst
        return ans, constant

    def compile(
        self,
        env,
        smap,
        expr_cache,
        args,
        remove_fixed_vars=False,
        check_duplicates=False,
    ):
        ans, constant = self._build_evaluator(
            smap, expr_cache, 1, 1, remove_fixed_vars, check_duplicates
        )
        if not ans:
            return constant
        indent = '\n    '
        if not constant and ans and ans[0].startswith('const +='):
            # Convert initial "const +=" to "const ="
            ans[0] = ''.join(ans[0].split('+', 1))
        else:
            ans.insert(0, 'const = ' + repr(constant))
        fcn_body = indent.join(ans[1:])
        if 'const' not in fcn_body:
            # No constants in the expression.  Move the initial const
            # term to the return value and avoid declaring the local
            # variable
            ans = ['return ' + ans[0].split('=', 1)[1]]
            if fcn_body:
                ans.insert(0, fcn_body)
        else:
            ans = [ans[0], fcn_body, 'return const']
        if check_duplicates:
            ans.insert(0, f"def build_expr(linear, {', '.join(args)}):")
        else:
            ans.insert(
                0, f"def build_expr(linear_indices, linear_data, {', '.join(args)}):"
            )
        ans = indent.join(ans)
        # build the function in the env namespace, then remove and
        # return the compiled function.  The function's globals will
        # still be bound to env
        exec(ans, env)
        return env.pop('build_expr')


class LinearTemplateBeforeChildDispatcher(linear.LinearBeforeChildDispatcher):
    def record_var(self, visitor, var):
        # Note: the following is mostly a copy of
        # LinearBeforeChildDispatcher.record_var, but with extra
        # hanlding to update the env in the same loop
        var_comp = var.parent_component()
        # Double-check that the component has not already been processed
        # (through an individual var data)
        name = visitor.symbolmap.getSymbol(var_comp)
        if name in visitor.env:
            return
        ve = visitor.env[name] = {}

        # We always add all indices to the var_map at once so that
        # we can honor deterministic ordering of unordered sets
        # (because the user could have iterated over an unordered
        # set when constructing an expression, thereby altering the
        # order in which we would see the variables)
        vm = visitor.var_map
        _iter = var_comp.items(visitor.sorter)
        for idx, v in _iter:
            # if v.fixed:
            #    ve[idx] = (v.value,)
            #    continue
            vid = id(v)
            vm[vid] = v
            ve[idx] = vid

    def _before_indexed_var(self, visitor, child):
        if child not in visitor.indexed_vars:
            visitor.before_child_dispatcher.record_var(visitor, child)
            visitor.indexed_vars.add(child)
        return False, (_VARIABLE, child)

    def _before_indexed_param(self, visitor, child):
        if child not in visitor.indexed_params:
            visitor.indexed_params.add(child)
            name = visitor.symbolmap.getSymbol(child)
            visitor.env[name] = child.extract_values()
        return False, (_CONSTANT, child)

    def _before_indexed_component(self, visitor, child):
        visitor.env[visitor.symbolmap.getSymbol(child)] = child
        return False, (_CONSTANT, child)

    def _before_index_template(self, visitor, child):
        symb = visitor.symbolmap.getSymbol(child)
        visitor.env[symb] = 0
        visitor.expr_cache[id(child)] = child
        return False, (_CONSTANT, child)

    def _before_component(self, visitor, child):
        visitor.env[visitor.symbolmap.getSymbol(child)] = child
        return False, (_CONSTANT, child)

    def _before_named_expression(self, visitor, child):
        raise NotImplementedError()


def _handle_getitem(visitor, node, comp, *args):
    expr = comp[1][tuple(arg[1] for arg in args)]
    if comp[0] is _CONSTANT:
        return (_CONSTANT, expr)
    elif comp[0] is _VARIABLE:
        # Because we are passing up an id() and not the expression
        # itself, we need to cache the expression that we just created
        # to preserve a reference to it and prevent deallocation / GC
        visitor.expr_cache[id(expr)] = expr
        ans = visitor.Result()
        ans.linear[id(expr)] = 1
        return (_LINEAR, ans)


def _handle_templatesum(visitor, node, comp, *args):
    ans = visitor.Result()
    if comp[0] == _LINEAR:
        ans.linear_sum.append((comp[1], node.template_iters(), [a[1] for a in args]))
        return _LINEAR, ans
    else:
        raise DeveloperError()


def define_exit_node_handlers(_exit_node_handlers=None):
    if _exit_node_handlers is None:
        _exit_node_handlers = {}
    linear.define_exit_node_handlers(_exit_node_handlers)

    _exit_node_handlers[expr.GetItemExpression] = {None: _handle_getitem}
    _exit_node_handlers[expr.TemplateSumExpression] = {None: _handle_templatesum}

    return _exit_node_handlers


class LinearTemplateRepnVisitor(linear.LinearRepnVisitor):
    Result = LinearTemplateRepn
    before_child_dispatcher = LinearTemplateBeforeChildDispatcher()
    exit_node_dispatcher = linear.ExitNodeDispatcher(
        util.initialize_exit_node_dispatcher(define_exit_node_handlers())
    )

    def __init__(
        self, subexpression_cache, var_map, var_order, sorter, remove_fixed_vars=False
    ):
        super().__init__(subexpression_cache, var_map, var_order, sorter)
        self.indexed_vars = set()
        self.indexed_params = set()
        self.expr_cache = {}
        self.env = {}
        self.symbolmap = expr.SymbolMap(NumericLabeler('x'))
        self.expanded_templates = {}
        self.remove_fixed_vars = remove_fixed_vars

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if node.__class__ is expr.TemplateSumExpression:
            return node.template_args(), []
        if node.__class__ in linear.sum_like_expression_types:
            return node.args, self.Result()
        else:
            return node.args, []

    def expand_expression(self, obj, template_info):
        env = self.env
        try:
            body, lb, ub = self.expanded_templates[id(template_info)]
        except KeyError:
            smap = self.symbolmap
            expr, indices = template_info
            args = [smap.getSymbol(i) for i in indices]
            if expr.is_expression_type(ExpressionType.RELATIONAL):
                lb, body, ub = obj.normalize_constraint()
                if body is not None:
                    body = self.walk_expression(body).compile(
                        env, smap, self.expr_cache, args, False
                    )
                if lb is not None:
                    lb = self.walk_expression(lb).compile(
                        env, smap, self.expr_cache, args, True
                    )
                if ub is not None:
                    ub = self.walk_expression(ub).compile(
                        env, smap, self.expr_cache, args, True
                    )
            elif expr is not None:
                lb = ub = None
                body = self.walk_expression(expr).compile(
                    env, smap, self.expr_cache, args, False
                )
            else:
                body = lb = ub = None
            self.expanded_templates[id(template_info)] = body, lb, ub

        linear_indices = []
        linear_data = []
        index = obj.index()
        if index.__class__ is not tuple:
            if index is None and not obj.parent_component().is_indexed():
                index = ()
            else:
                index = (index,)
        if lb.__class__ is code_type:
            lb = lb(linear_indices, linear_data, *index)
            if linear_indices:
                raise RuntimeError(f"Constraint {obj} has non-fixed lower bound")
        if ub.__class__ is code_type:
            ub = ub(linear_indices, linear_data, *index)
            if linear_indices:
                raise RuntimeError(f"Constraint {obj} has non-fixed upper bound")
        return (
            body(linear_indices, linear_data, *index),
            linear_indices,
            linear_data,
            lb,
            ub,
        )


def pyomo_create_model(N, M, P):
    import random

    random.seed(1000)

    model = ConcreteModel()
    model.N = Param(within=PositiveIntegers, initialize=N)
    model.M = Param(within=PositiveIntegers, initialize=M)
    model.P = Param(within=RangeSet(1, model.N), initialize=P)
    model.Locations = RangeSet(1, model.N)
    model.Customers = RangeSet(1, model.M)
    model.d = Param(
        model.Locations,
        model.Customers,
        initialize=lambda n, m, model: random.uniform(1.0, 2.0),
        within=Reals,
    )
    model.x = Var(model.Locations, model.Customers, bounds=(0.0, 1.0))
    model.y = Var(model.Locations, within=Binary)

    @model.Objective()
    def obj(model):
        return sum(
            model.d[n, m] * model.x[n, m]
            for n in model.Locations
            for m in model.Customers
        )

    @model.Constraint(model.Customers)
    def single_x(model, m):
        return sum(model.x[n, m] for n in model.Locations) == 1.0

    @model.Constraint(model.Locations, model.Customers)
    def bound_y(model, n, m):
        return model.x[n, m] <= model.y[n]

    @model.Constraint()
    def num_facilities(model):
        return sum(model.y[n] for n in model.Locations) - model.P == 0.0

    return model


if __name__ == '__main__':
    import logging
    import gc
    import cProfile
    import pstats
    from pyomo.common.timing import report_timing, TicTocTimer
    from pyomo.environ import *
    import pyomo.core.base.constraint as _c
    import pyomo.core.base.objective as _o

    import gurobipy
    import scipy.sparse

    report_timing(level=logging.DEBUG)

    timing_logger = logging.getLogger('pyomo.common.timing.create_model')
    timer = TicTocTimer(logger=timing_logger)

    gc.disable()

    pr = cProfile.Profile()
    profile_count = 0

    if 1:
        _c.TEMPLATIZE_CONSTRAINTS = True
        _o.TEMPLATIZE_OBJECTIVES = True

    m = pyomo_create_model(640, 640, 1)
    timer.toc("Created model")

    if False:
        comps = []
        if _c.TEMPLATIZE_CONSTRAINTS:
            comps.extend((m.bound_y, m.single_x, m.num_facilities))
        if _o.TEMPLATIZE_OBJECTIVES:
            comps.append(m.obj)
        if comps:
            visitor = LinearTemplateRepnVisitor({}, {}, {}, None)

            for comp in comps:
                e, i = next(iter(comp._data.values())).template_expr()
                print(comp.name, e)
                if e.is_expression_type(ExpressionType.RELATIONAL):
                    for arg in e.args:
                        print(visitor.walk_expression(arg))
                else:
                    print(visitor.walk_expression(e))

    # print(SolverFactory('gurobi_direct_v2').solve(m, tee=True))
    import pyomo.contrib.solver.factory

    if profile_count:
        pr.enable()
    # r = SolverFactory('gurobi').solve(m, tee=True)
    r = pyomo.contrib.solver.factory.SolverFactory('gurobi_direct').solve(m, tee=True)
    if profile_count:
        pr.disable()

    timer.toc("Solved model")
    r.display()

    if profile_count:
        profile_count = abs(profile_count)
        ps = pstats.Stats(pr)
        # ps = ps.sort_stats('time','calls')
        ps = ps.sort_stats('cumtime', 'calls')
        ps.print_stats(profile_count)
        ps.print_callers(profile_count)
        ps.print_callees(profile_count)
