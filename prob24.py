"""prob24 - Pyomo version of prob24.ufo

Sparse minimization with nonlinear equality constraints
(Luksan-Vlcek sparse test problem collection).

  minimize    f(x) = sum_{j=1..NF} [ (3 - 2*x_j)*x_j + 1 - x_{j-1} - x_{j+1} ]^2
              (Broyden tridiagonal function; x_0 and x_{NF+1} terms omitted)

  subject to  c_K(x) = 0,  for K = 3 .. NF-2   (NC equality constraints)
              c_K(x) = 8*x_K*(x_K^2 - x_{K-1}) - 2*(1 - x_K)
                       + 4*(x_K - x_{K+1}^2) + x_{K-1}^2
                       - x_{K-2} + x_{K+1} - x_{K+2}^2

  start       x_j = -1.0
"""

import pyomo.environ as pyo

NF = 100          # number of variables
NC = NF - 4       # number of constraints (= 96); K runs 3 .. NF-2

# ---- Objective: Broyden tridiagonal least-squares -------------------------
def _obj_rule(m):
    total = 0.0
    for j in m.J:
        wa = (3.0 - 2.0 * m.x[j]) * m.x[j] + 1.0
        if j > 1:
            wa -= m.x[j - 1]
        if j < NF:
            wa -= m.x[j + 1]
        total += wa ** 2
    return total


# ---- Constraints: c_K(x) = 0 for K = 3 .. NF-2 ----------------------------
def _con_rule(m, k):
    return (
        8.0 * m.x[k] * (m.x[k] ** 2 - m.x[k - 1])
        - 2.0 * (1.0 - m.x[k])
        + 4.0 * (m.x[k] - m.x[k + 1] ** 2)
        + m.x[k - 1] ** 2
        - m.x[k - 2]
        + m.x[k + 1]
        - m.x[k + 2] ** 2
        == 0.0
    )


def build_model():
    """Return a fresh, unsolved prob24 model (variables start at -1.0)."""
    model = pyo.ConcreteModel(name="prob24")
    model.J = pyo.RangeSet(1, NF)                 # variables x_1 .. x_NF
    model.x = pyo.Var(model.J, initialize=-1.0)
    model.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)
    model.K = pyo.RangeSet(3, NF - 2)             # constraints c_3 .. c_{NF-2}
    model.con = pyo.Constraint(model.K, rule=_con_rule)
    return model


# Module-level model so `from prob24 import model` keeps working.
model = build_model()


def _max_constraint_violation(m):
    """Largest |c_K(x)| over all constraints (all are equalities == 0)."""
    worst = 0.0
    for k in m.K:
        body = pyo.value(m.con[k].body)
        lb = m.con[k].lower
        ub = m.con[k].upper
        viol = 0.0
        if lb is not None:
            viol = max(viol, pyo.value(lb) - body)
        if ub is not None:
            viol = max(viol, body - pyo.value(ub))
        worst = max(worst, viol)
    return worst


def _solve_and_summarize(solver_name, make_solver):
    """Solve a fresh model with solver_name; return (obj, max_viol, tc, x)."""
    m = build_model()
    opt = make_solver(solver_name)
    res = opt.solve(m)
    tc = str(res.solver.termination_condition)
    obj = pyo.value(m.obj)
    viol = _max_constraint_violation(m)
    x = [pyo.value(m.x[j]) for j in m.J]
    return obj, viol, tc, x


def _compare():
    """Solve prob24 with both UFO and IPOPT and print a cross-check."""
    # Reuse the IPOPT/MUMPS environment setup from compare_solvers.py
    # (this build's default HSL linear solver is not installed).
    from compare_solvers import _make_solver  # noqa: F401 — also sets LD path

    print(f"prob24: NF={NF} variables, NC={NC} nonlinear equality constraints\n")
    results = {}
    for name in ("ufo", "ipopt"):
        try:
            obj, viol, tc, x = _solve_and_summarize(name, _make_solver)
            results[name] = (obj, x)
            print(f"  {name:>6}: obj={obj:.10g}  max_con_viol={viol:.2e}  [{tc}]")
        except Exception as e:      # noqa: BLE001 — report and continue
            results[name] = None
            print(f"  {name:>6}: FAILED ({type(e).__name__}: {e})")

    o_ufo = results.get("ufo")
    o_ipopt = results.get("ipopt")
    if o_ufo and o_ipopt:
        d_obj = abs(o_ufo[0] - o_ipopt[0])
        d_x = max(abs(a - b) for a, b in zip(o_ufo[1], o_ipopt[1]))
        agree = d_obj <= 1e-4 * (1 + abs(o_ipopt[0]))
        print(f"\n  -> objective |Δ|={d_obj:.3e}  max|Δx|={d_x:.3e}  "
              f"({'MATCH' if agree else 'DIFFER (likely different local optima)'})")
    else:
        print("\n  -> comparison N/A (a solver failed)")


if __name__ == "__main__":
    import argparse

    import pyomo_ufo  # noqa: F401 — registers the 'ufo' solver

    parser = argparse.ArgumentParser(description="Solve prob24 with UFO.")
    parser.add_argument(
        "--compare", action="store_true",
        help="also solve with IPOPT and compare objective / feasibility",
    )
    args = parser.parse_args()

    if args.compare:
        _compare()
    else:
        solver = pyo.SolverFactory("ufo")
        results = solver.solve(model)

        print(results)
        print("Objective =", pyo.value(model.obj))
        print("x =", [pyo.value(model.x[j]) for j in model.J])
