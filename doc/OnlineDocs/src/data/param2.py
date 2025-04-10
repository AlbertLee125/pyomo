#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *

model = AbstractModel()

# @decl
model.A = Set()
model.B = Param(model.A)
# @decl

instance = model.create_instance('param2.dat')

keys = instance.B.keys()
for key in sorted(keys):
    print(str(key) + " " + str(value(instance.B[key])))
