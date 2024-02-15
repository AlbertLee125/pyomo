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

model.Z = Set(initialize=[('A1', 'B1', 1), ('A2', 'B2', 2), ('A3', 'B3', 3)])
# model.Z = Set(dimen=3)
model.D = Param(model.Z)

instance = model.create_instance('ABCD2.dat')

print('Z ' + str(sorted(list(instance.Z.data()))))
print('D')
for key in sorted(instance.D.keys()):
    print(name(instance.D, key) + " " + str(value(instance.D[key])))
