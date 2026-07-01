"""
demo_ufo.py
-----------
Demonstration of the pyomo_ufo solver interface.

Run from the UFO root directory:

    cd /home/petra/work_tacr/UFO
    python examples/demo_ufo.py
"""

import sys
import os

# Allow running from the UFO root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyomo.environ as pyo
import pyomo_ufo  # registers 'ufo' with SolverFactory


def separator(title):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print('='*60)


def check(name, got, expected, tol=1e-4):
    ok = abs(got - expected) < tol
    mark = '✓' if ok else '✗'
    print(f'  {mark} {name}: {got:.8f}  (expected ≈ {expected})')
    return ok


# ---------------------------------------------------------------
# Problem 1: Unconstrained Rosenbrock
# ---------------------------------------------------------------
separator('1. Unconstrained Rosenbrock')

m = pyo.ConcreteModel()
m.x = pyo.Var([1, 2], initialize={1: -1.2, 2: 1.0})
m.obj = pyo.Objective(
    expr=100*(m.x[2] - m.x[1]**2)**2 + (m.x[1] - 1)**2
)

opt = pyo.SolverFactory('ufo')
res = opt.solve(m)

print(f'  Status:  {res.solver.status}')
print(f'  TC:      {res.solver.termination_condition}')
print(f'  Message: {res.solver.termination_message}')
check('x[1]', pyo.value(m.x[1]), 1.0)
check('x[2]', pyo.value(m.x[2]), 1.0)
check('obj',  pyo.value(m.obj),  0.0)


# ---------------------------------------------------------------
# Problem 2: Box-constrained minimization
# ---------------------------------------------------------------
separator('2. Box-constrained  (min (x-1.5)^2 + (y-1.5)^2, 0 ≤ x,y ≤ 1)')

m2 = pyo.ConcreteModel()
m2.x = pyo.Var([1, 2], bounds=(0.0, 1.0), initialize=0.5)
m2.obj = pyo.Objective(expr=(m2.x[1] - 1.5)**2 + (m2.x[2] - 1.5)**2)

res2 = opt.solve(m2)
print(f'  Status:  {res2.solver.status}  ({res2.solver.termination_message})')
check('x[1]', pyo.value(m2.x[1]), 1.0)
check('x[2]', pyo.value(m2.x[2]), 1.0)
check('obj',  pyo.value(m2.obj),  0.5)


# ---------------------------------------------------------------
# Problem 3: Linear equality constraint
# min  x^2 + y^2   s.t.  x + y = 1
# Solution: x = y = 0.5, obj = 0.5
# ---------------------------------------------------------------
separator('3. Linear equality constraint  (x + y = 1)')

m3 = pyo.ConcreteModel()
m3.x = pyo.Var([1, 2], initialize=0.8)
m3.obj = pyo.Objective(expr=m3.x[1]**2 + m3.x[2]**2)
m3.c1 = pyo.Constraint(expr=m3.x[1] + m3.x[2] == 1.0)

res3 = opt.solve(m3)
print(f'  Status:  {res3.solver.status}  ({res3.solver.termination_message})')
check('x[1]', pyo.value(m3.x[1]), 0.5)
check('x[2]', pyo.value(m3.x[2]), 0.5)
check('obj',  pyo.value(m3.obj),  0.5)


# ---------------------------------------------------------------
# Problem 4: Linear inequality constraint
# min  x^2 + y^2   s.t.  x + y >= 1
# Solution: x = y = 0.5, obj = 0.5
# ---------------------------------------------------------------
separator('4. Linear inequality constraint  (x + y >= 1)')

m4 = pyo.ConcreteModel()
m4.x = pyo.Var([1, 2], initialize=0.8)
m4.obj = pyo.Objective(expr=m4.x[1]**2 + m4.x[2]**2)
m4.c1 = pyo.Constraint(expr=m4.x[1] + m4.x[2] >= 1.0)

res4 = opt.solve(m4)
print(f'  Status:  {res4.solver.status}  ({res4.solver.termination_message})')
check('x[1]', pyo.value(m4.x[1]), 0.5)
check('x[2]', pyo.value(m4.x[2]), 0.5)
check('obj',  pyo.value(m4.obj),  0.5)


# ---------------------------------------------------------------
# Problem 5: Nonlinear constraint
# min  x + y   s.t.  x^2 + y^2 >= 1,  x >= 0,  y >= 0
# Solution: x = y = 1/sqrt(2) ≈ 0.7071,  obj ≈ 1.4142
# ---------------------------------------------------------------
separator('5. Nonlinear inequality constraint  (x^2 + y^2 >= 1)')

m5 = pyo.ConcreteModel()
m5.x = pyo.Var([1, 2], bounds=(0.0, None), initialize=1.0)
m5.obj = pyo.Objective(expr=m5.x[1] + m5.x[2])
m5.c1 = pyo.Constraint(expr=m5.x[1]**2 + m5.x[2]**2 >= 1.0)

res5 = opt.solve(m5)
print(f'  Status:  {res5.solver.status}  ({res5.solver.termination_message})')
import math
check('x[1]', pyo.value(m5.x[1]), 1/math.sqrt(2))
check('x[2]', pyo.value(m5.x[2]), 1/math.sqrt(2))
check('obj',  pyo.value(m5.obj),  math.sqrt(2))


# ---------------------------------------------------------------
# Problem 6: Maximization
# max  x*y   s.t.  x + y = 10,  x,y >= 0
# Solution: x = y = 5, obj = 25
# ---------------------------------------------------------------
separator('6. Maximization  (max x*y,  x+y=10)')

m6 = pyo.ConcreteModel()
m6.x = pyo.Var([1, 2], bounds=(0.0, None), initialize=3.0)
m6.obj = pyo.Objective(expr=m6.x[1] * m6.x[2], sense=pyo.maximize)
m6.c1 = pyo.Constraint(expr=m6.x[1] + m6.x[2] == 10.0)

res6 = opt.solve(m6)
print(f'  Status:  {res6.solver.status}  ({res6.solver.termination_message})')
check('x[1]', pyo.value(m6.x[1]), 5.0)
check('x[2]', pyo.value(m6.x[2]), 5.0)
check('obj',  pyo.value(m6.obj),  25.0)

# ---------------------------------------------------------------
# Problem 7: Two-sided range constraint
# min  (x1-2)^2 + (x2-2)^2   s.t.  1 <= x1 + x2 <= 3
# Unconstrained min (2,2) has x1+x2=4 > 3, so the upper bound is
# active: project (2,2) onto x1+x2=3 -> (1.5, 1.5), obj = 0.5
# ---------------------------------------------------------------
separator('7. Two-sided range constraint  (1 <= x1+x2 <= 3)')

m7 = pyo.ConcreteModel()
m7.x = pyo.Var([1, 2], initialize=0.0)
m7.obj = pyo.Objective(expr=(m7.x[1] - 2)**2 + (m7.x[2] - 2)**2)
m7.c1 = pyo.Constraint(expr=pyo.inequality(1.0, m7.x[1] + m7.x[2], 3.0))

res7 = opt.solve(m7)
print(f'  Status:  {res7.solver.status}  ({res7.solver.termination_message})')
check('x[1]', pyo.value(m7.x[1]), 1.5)
check('x[2]', pyo.value(m7.x[2]), 1.5)
check('obj',  pyo.value(m7.obj),  0.5)


# ---------------------------------------------------------------
# Problem 8: Nonlinear equality constraint
# min  (x1-1)^2 + (x2-1)^2   s.t.  x1^2 + x2^2 = 1
# Closest point on the unit circle to (1,1): x1 = x2 = 1/sqrt(2),
# obj = 2*(1 - 1/sqrt(2))^2 ≈ 0.171573
# ---------------------------------------------------------------
separator('8. Nonlinear equality constraint  (x1^2 + x2^2 = 1)')

m8 = pyo.ConcreteModel()
m8.x = pyo.Var([1, 2], initialize=0.6)
m8.obj = pyo.Objective(expr=(m8.x[1] - 1)**2 + (m8.x[2] - 1)**2)
m8.c1 = pyo.Constraint(expr=m8.x[1]**2 + m8.x[2]**2 == 1.0)

res8 = opt.solve(m8)
print(f'  Status:  {res8.solver.status}  ({res8.solver.termination_message})')
check('x[1]', pyo.value(m8.x[1]), 1 / math.sqrt(2))
check('x[2]', pyo.value(m8.x[2]), 1 / math.sqrt(2))
check('obj',  pyo.value(m8.obj),  2 * (1 - 1 / math.sqrt(2))**2)


# ---------------------------------------------------------------
# Problem 9: Two-sided box bounds, unconstrained min outside the box
# min  (x1-3)^2 + (x2+1)^2   s.t.  -1 <= x1,x2 <= 1
# Unconstrained min (3,-1) is outside the box -> clamp to (1,-1),
# obj = (1-3)^2 + (-1+1)^2 = 4
# ---------------------------------------------------------------
separator('9. Two-sided box bounds  (-1 <= x <= 1)')

m9 = pyo.ConcreteModel()
m9.x = pyo.Var([1, 2], bounds=(-1.0, 1.0), initialize=0.0)
m9.obj = pyo.Objective(expr=(m9.x[1] - 3)**2 + (m9.x[2] + 1)**2)

res9 = opt.solve(m9)
print(f'  Status:  {res9.solver.status}  ({res9.solver.termination_message})')
check('x[1]', pyo.value(m9.x[1]),  1.0)
check('x[2]', pyo.value(m9.x[2]), -1.0)
check('obj',  pyo.value(m9.obj),   4.0)


# ---------------------------------------------------------------
# Problem 10: Complementarity / binary variable
# min  (y - 0.7)^2   s.t.  y*(1-y) = 0,  0 <= y <= 1
# The constraint forces y in {0, 1}; the closer one to 0.7 is y=1,
# obj = (1 - 0.7)^2 = 0.09
# ---------------------------------------------------------------
separator('10. Complementarity  (y*(1-y)=0 forces y in {0,1})')

m10 = pyo.ConcreteModel()
m10.y = pyo.Var(bounds=(0.0, 1.0), initialize=0.8)
m10.obj = pyo.Objective(expr=(m10.y - 0.7)**2)
m10.c1 = pyo.Constraint(expr=m10.y * (1 - m10.y) == 0.0)

res10 = opt.solve(m10)
print(f'  Status:  {res10.solver.status}  ({res10.solver.termination_message})')
check('y',   pyo.value(m10.y),   1.0)
check('obj', pyo.value(m10.obj), 0.09)


# ---------------------------------------------------------------
# Problem 11: Pure linear program (objective stays MODEL='FL')
# max  x1 + 2*x2   s.t.  x1 + x2 <= 4,  0 <= x1,x2 <= 3
# Push x2 to its bound 3, then x1 = 4-3 = 1 -> (1, 3), obj = 7
# ---------------------------------------------------------------
separator('11. Linear program  (max x1+2x2, x1+x2<=4, 0<=x<=3)')

m11 = pyo.ConcreteModel()
m11.x = pyo.Var([1, 2], bounds=(0.0, 3.0), initialize=1.0)
m11.obj = pyo.Objective(expr=m11.x[1] + 2 * m11.x[2], sense=pyo.maximize)
m11.c1 = pyo.Constraint(expr=m11.x[1] + m11.x[2] <= 4.0)

res11 = opt.solve(m11)
print(f'  Status:  {res11.solver.status}  ({res11.solver.termination_message})')
check('x[1]', pyo.value(m11.x[1]), 1.0)
check('x[2]', pyo.value(m11.x[2]), 3.0)
check('obj',  pyo.value(m11.obj),  7.0)


print('\n' + '='*60)
print('  Done.')
print('='*60)
