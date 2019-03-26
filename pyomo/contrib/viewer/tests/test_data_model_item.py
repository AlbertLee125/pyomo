##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# This software is distributed under the 3-clause BSD License.
##############################################################################
"""
Test data model items for QTreeView. These tests should work even without PyQt.
"""

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.contrib.viewer.model_browser import ComponentDataItem

class TestDataModelItem(unittest.TestCase):
    def setUp(self):
        # Borrowed this test model from the trust region tests
        m = ConcreteModel()
        m.z = Var(range(3), domain=Reals, initialize=2.)
        m.x = Var(range(4), initialize=2.)
        m.x[1] = 1.0
        m.x[2] = 0.0
        m.x[3] = None

        m.b1 = Block()
        m.b1.e1 = Expression(expr=m.x[0] + m.x[1])
        m.b1.e2 = Expression(expr=m.x[0]/m.x[2])
        m.b1.e3 = Expression(expr=m.x[3]*m.x[1])
        m.b1.e4 = Expression(expr=log(m.x[2]))
        m.b1.e5 = Expression(expr=log(m.x[2] - 2))

        def blackbox(a,b):
            return sin(a-b)
        self.bb = ExternalFunction(blackbox)

        m.obj = Objective(
            expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
                + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
            )
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + self.bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
        m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))
        m.c3 = Constraint(expr=m.x[1] == 3)
        m.c4 = Constraint(expr=0 == 3/m.x[2])
        m.c5 = Constraint(expr=0 == log(m.x[2]))
        m.c6 = Constraint(expr=0 == log(m.x[2]-4))
        m.c7 = Constraint(expr=0 == log(m.x[3]))
        m.p1 = Param(mutable=True, initialize=1)
        m.c8 = Constraint(expr = m.x[1] <= 1/m.p1)
        m.p1 = 0
        self.m = m.clone()

    def test_expr_calc(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.b1.e1)
        cdi.calculate()
        assert(abs(cdi.get("value")-3) < 0.0001)

    def test_expr_calc_div0(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.b1.e2)
        cdi.calculate()
        assert(cdi.get("value") == "Divide_by_0")

    def test_expr_calc_log0(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.b1.e4)
        cdi.calculate()
        assert(cdi.get("value") == None)

    def test_expr_calc_log_neg(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.b1.e5)
        cdi.calculate()
        assert(cdi.get("value") == None)

    def test_expr_calc_value_None(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.b1.e3)
        cdi.calculate()
        assert(cdi.get("value") == None)

    def test_cons_calc(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.c3)
        cdi.calculate()
        assert(abs(cdi.get("residual") - 2) < 0.0001)

    def test_cons_calc_div0(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.c4)
        cdi.calculate()
        assert(cdi.get("value") == "Divide_by_0")

    def test_cons_calc_log0(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.c5)
        cdi.calculate()
        assert(cdi.get("value") == None)

    def test_cons_calc_log_neg(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.c6)
        cdi.calculate()
        assert(cdi.get("value") == None)

    def test_cons_calc_value_None(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.c7)
        cdi.calculate()
        assert(cdi.get("value") == None)

    def test_cons_calc_upper_div0(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.c8)
        cdi.calculate()
        # the ui lists the upper and lower attributes as ub and lb
        # this was originally so I could easily combine variables and
        # constarints in the same view, but I split them up, so may want
        # to reconsider that choise in the future. This is to remind myself
        # why I'm getting "ub" and not "upper"
        assert(cdi.get("ub") == "Divide_by_0")

    def test_var_get_value(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.x[1])
        assert(abs(cdi.get("value") - 1) < 0.0001)

    def test_var_get_bounds(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.x[1])
        self.m.x[1].setlb(0)
        self.m.x[1].setub(10)
        assert(abs(cdi.get("lb") - 0) < 0.0001)
        assert(abs(cdi.get("ub") - 10) < 0.0001)

    def test_var_set_bounds(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.x[1])
        cdi.set("lb", 2)
        cdi.set("ub", 8)
        assert(abs(cdi.get("lb") - 2) < 0.0001)
        assert(abs(cdi.get("ub") - 8) < 0.0001)

    def test_var_fixed_bounds(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.x[1])
        cdi.set("fixed", True)
        assert(cdi.get("fixed"))
        cdi.set("fixed", False)
        assert(not cdi.get("fixed"))

    def test_degrees_of_freedom(self):
        import pyomo.contrib.viewer.report as rpt
        # this should hit everything in report.  It only exists to calculate
        # degrees of freedom for display in the ui
        assert(rpt.degrees_of_freedom(self.m)==0)
