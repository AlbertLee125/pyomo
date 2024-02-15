#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *

model = AbstractModel()

model.A = Set(initialize=['A1', 'A2', 'A3'])
model.B = Set(initialize=['B1', 'B2', 'B3'])

model.M = Param(model.A)
model.N = Param(model.A, model.B)

instance = model.create_instance('table2.dat')
instance.pprint()
