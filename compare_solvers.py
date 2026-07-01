"""
compare_solvers.py
------------------
Cross-check the pyomo_ufo (UFO) solver against a reference solver (IPOPT) on
the same set of models used in demo_ufo.py.

Each model is defined by a builder function so a fresh, unsolved copy can be
handed to every solver. For each problem we print the objective and the key
variable values from both solvers side by side, plus the analytically known
optimum, so discrepancies are easy to spot.

    uv run compare_solvers.py

Requires a working IPOPT on PATH (pyomo SolverFactory('ipopt')).
"""

import glob
import math
import os

import pyomo.environ as pyo
import pyomo_ufo  # noqa: F401 — registers 'ufo' with SolverFactory


# ---------------------------------------------------------------------------
# IPOPT setup
# ---------------------------------------------------------------------------
# This IPOPT build defaults to the HSL linear solver (libhsl.so), which is not
# installed, so it aborts with a library-loading failure and silently returns
# the starting point. Force the bundled MUMPS solver and make sure the
# directory holding libcoinmumps.so is reachable at run time.
_MUMPS_DIRS = ['/dir/to/install/lib', '/usr/local/lib',
               '/usr/lib/x86_64-linux-gnu']


def _ensure_mumps_on_path():
    """Prepend the directory containing libcoinmumps.so to LD_LIBRARY_PATH."""
    for d in _MUMPS_DIRS:
        if glob.glob(os.path.join(d, 'libcoinmumps.so*')):
            cur = os.environ.get('LD_LIBRARY_PATH', '')
            if d not in cur.split(':'):
                os.environ['LD_LIBRARY_PATH'] = f'{d}:{cur}' if cur else d
            return d
    return None


def _make_solver(name):
    opt = pyo.SolverFactory(name)
    if name == 'ipopt':
        opt.options['linear_solver'] = 'mumps'
    return opt


_ensure_mumps_on_path()


# ---------------------------------------------------------------------------
# Model builders: each returns (model, vars_of_interest, expected)
#   vars_of_interest : list of (label, Var) to report
#   expected         : dict {label -> value, 'obj' -> value}
# ---------------------------------------------------------------------------

def p1_rosenbrock():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize={1: -1.2, 2: 1.0})
    m.obj = pyo.Objective(expr=100*(m.x[2] - m.x[1]**2)**2 + (m.x[1] - 1)**2)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': 1.0, 'x2': 1.0, 'obj': 0.0}


def p2_box():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], bounds=(0.0, 1.0), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x[1] - 1.5)**2 + (m.x[2] - 1.5)**2)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': 1.0, 'x2': 1.0, 'obj': 0.5}


def p3_lin_eq():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=0.8)
    m.obj = pyo.Objective(expr=m.x[1]**2 + m.x[2]**2)
    m.c1 = pyo.Constraint(expr=m.x[1] + m.x[2] == 1.0)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': 0.5, 'x2': 0.5, 'obj': 0.5}


def p4_lin_ineq():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=0.8)
    m.obj = pyo.Objective(expr=m.x[1]**2 + m.x[2]**2)
    m.c1 = pyo.Constraint(expr=m.x[1] + m.x[2] >= 1.0)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': 0.5, 'x2': 0.5, 'obj': 0.5}


def p5_nl_ineq():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], bounds=(0.0, None), initialize=1.0)
    m.obj = pyo.Objective(expr=m.x[1] + m.x[2])
    m.c1 = pyo.Constraint(expr=m.x[1]**2 + m.x[2]**2 >= 1.0)
    s = 1 / math.sqrt(2)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': s, 'x2': s, 'obj': math.sqrt(2)}


def p6_max():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], bounds=(0.0, None), initialize=3.0)
    m.obj = pyo.Objective(expr=m.x[1] * m.x[2], sense=pyo.maximize)
    m.c1 = pyo.Constraint(expr=m.x[1] + m.x[2] == 10.0)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': 5.0, 'x2': 5.0, 'obj': 25.0}


def p7_range():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x[1] - 2)**2 + (m.x[2] - 2)**2)
    m.c1 = pyo.Constraint(expr=pyo.inequality(1.0, m.x[1] + m.x[2], 3.0))
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': 1.5, 'x2': 1.5, 'obj': 0.5}


def p8_nl_eq():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=0.6)
    m.obj = pyo.Objective(expr=(m.x[1] - 1)**2 + (m.x[2] - 1)**2)
    m.c1 = pyo.Constraint(expr=m.x[1]**2 + m.x[2]**2 == 1.0)
    s = 1 / math.sqrt(2)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': s, 'x2': s, 'obj': 2*(1 - s)**2}


def p9_box2():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], bounds=(-1.0, 1.0), initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x[1] - 3)**2 + (m.x[2] + 1)**2)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': 1.0, 'x2': -1.0, 'obj': 4.0}


def p10_complementarity():
    m = pyo.ConcreteModel()
    m.y = pyo.Var(bounds=(0.0, 1.0), initialize=0.8)
    m.obj = pyo.Objective(expr=(m.y - 0.7)**2)
    m.c1 = pyo.Constraint(expr=m.y * (1 - m.y) == 0.0)
    return m, [('y', m.y)], {'y': 1.0, 'obj': 0.09}


def p11_lp():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], bounds=(0.0, 3.0), initialize=1.0)
    m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2], sense=pyo.maximize)
    m.c1 = pyo.Constraint(expr=m.x[1] + m.x[2] <= 4.0)
    return m, [('x1', m.x[1]), ('x2', m.x[2])], {'x1': 1.0, 'x2': 3.0, 'obj': 7.0}


PROBLEMS = [
    ('1. Rosenbrock (unconstrained)',        p1_rosenbrock),
    ('2. Box-constrained',                   p2_box),
    ('3. Linear equality',                   p3_lin_eq),
    ('4. Linear inequality',                 p4_lin_ineq),
    ('5. Nonlinear inequality',              p5_nl_ineq),
    ('6. Maximization (x*y)',                p6_max),
    ('7. Two-sided range constraint',        p7_range),
    ('8. Nonlinear equality',                p8_nl_eq),
    ('9. Two-sided box bounds',              p9_box2),
    ('10. Complementarity (binary)',         p10_complementarity),
    ('11. Linear program',                   p11_lp),
]

SOLVERS = ['ufo', 'ipopt']
TOL = 1e-4


def solve_one(solver_name, builder):
    """Build a fresh model, solve it, return (obj, {label: value}, message)."""
    model, vois, _ = builder()
    opt = _make_solver(solver_name)
    try:
        res = opt.solve(model)
        msg = str(res.solver.termination_condition)
    except Exception as e:
        return None, {}, f'ERROR: {type(e).__name__}: {e}'
    try:
        obj = pyo.value(model.obj)
    except Exception:
        obj = None
    values = {}
    for label, var in vois:
        v = pyo.value(var, exception=False)
        values[label] = v
    return obj, values, msg


def fmt(v):
    return f'{v:.6f}' if isinstance(v, float) else str(v)


def main():
    for title, builder in PROBLEMS:
        _, vois, expected = builder()
        labels = [lbl for lbl, _ in vois]

        print('\n' + '=' * 72)
        print(f'  {title}')
        print('=' * 72)

        # Column header
        cols = labels + ['obj']
        print(f'  {"solver":<10}' + ''.join(f'{c:>14}' for c in cols)
              + '   termination')

        # Reference (analytic) row
        print(f'  {"expected":<10}'
              + ''.join(f'{fmt(expected.get(c)):>14}' for c in cols))

        results = {}
        for sname in SOLVERS:
            obj, values, msg = solve_one(sname, builder)
            results[sname] = (obj, values)
            row = ''.join(f'{fmt(values.get(c) if c != "obj" else obj):>14}'
                          for c in cols)
            print(f'  {sname:<10}{row}   {msg}')

        # Agreement check between the two solvers (objective)
        o_ufo = results.get('ufo', (None,))[0]
        o_ref = results.get('ipopt', (None,))[0]
        if isinstance(o_ufo, float) and isinstance(o_ref, float):
            agree = abs(o_ufo - o_ref) <= TOL * (1 + abs(o_ref))
            mark = 'MATCH' if agree else 'DIFFER'
            print(f'  -> ufo vs ipopt objective: {mark} '
                  f'(|Δ|={abs(o_ufo - o_ref):.2e})')
        else:
            print('  -> ufo vs ipopt objective: N/A (a solver failed)')

    print('\n' + '=' * 72)
    print('  Done.')
    print('=' * 72)


if __name__ == '__main__':
    main()
