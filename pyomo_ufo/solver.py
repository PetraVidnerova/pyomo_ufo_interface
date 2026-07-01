"""
solver.py
---------
Pyomo solver plugin that wraps the UFO optimization system.

Register with Pyomo's SolverFactory by importing this module (or the
``pyomo_ufo`` package)::

    import pyomo_ufo
    opt = pyo.SolverFactory('ufo')
    results = opt.solve(model, tee=True)

Options
~~~~~~~
Set via ``opt.options`` or keyword arguments to ``solve()``:

    ufo_dir    : str  – path to UFO installation directory
                        (default: auto-detect relative to this file)
    keepfiles  : bool – keep generated P.UFO / P.F / P.OUT in a temp dir
                        (default: False)
    tee        : bool – stream UFO stdout to console (default: False)
    model_kind : str  – UFO $MODEL directive: 'FL' for linear objective,
                        'FF' for general nonlinear. Default 'FL'.
"""

import os
import subprocess
from pathlib import Path

import pyomo.environ as pyo
from pyomo.opt import (
    OptSolver,
    SolverFactory,
    SolverResults,
    SolverStatus,
    TerminationCondition,
)
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.problem import ProblemSense

from .writer import UFOWriter
from .parser import UFOOutputParser

# Default UFO directory: two levels up from this file → ufo/ subdirectory
_DEFAULT_UFO_DIR = str(Path(__file__).parent.parent / 'ufo')

# Files / patterns that must be symlinked into the working directory
_REQUIRED_FILES = ['ufobel', 'libufo.a']
_REQUIRED_GLOB = ['*.I']

# Preferred gfortran binary (libufo.a was recompiled with gfortran-9)
_GFORTRAN = 'gfortran-9'

# Directory containing libgfortran.so.3 (needed by ufobel binary)
_LIBGFORTRAN3_DIRS = [
    '/home/petra/miniconda3/pkgs/libgcc-7.2.0-h69d50b8_2/lib',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/lib64',
    '/usr/local/lib',
]


def _find_libgfortran3():
    """Return path to a directory containing libgfortran.so.3, or None."""
    for d in _LIBGFORTRAN3_DIRS:
        if Path(d, 'libgfortran.so.3').exists():
            return d
    return None


@SolverFactory.register('ufo', doc='UFO universal function optimizer (Luksan et al., CAS)')
class UFOSolver(OptSolver):
    """Pyomo solver plugin for the UFO optimization system."""

    def __init__(self, **kwds):
        kwds['type'] = 'ufo'
        super().__init__(**kwds)

    def available(self, exception_flag=True):
        """Return True if the UFO ufobel binary is reachable."""
        ufo_dir = self._get_ufo_dir()
        ufobel = Path(ufo_dir) / 'ufobel'
        if ufobel.exists() and os.access(ufobel, os.X_OK):
            return True
        if exception_flag:
            raise RuntimeError(
                f"UFO binary 'ufobel' not found or not executable in '{ufo_dir}'. "
                "Set opt.options['ufo_dir'] to the UFO installation directory."
            )
        return False

    def license_is_valid(self):
        return True

    def warm_start_capable(self):
        return False

    def version(self):
        return (1, 0, 0, 0)

    # ------------------------------------------------------------------
    # Core solve logic — override solve() to bypass file-based _presolve
    # ------------------------------------------------------------------

    def solve(self, model, **kwds):
        """Solve the Pyomo model and return SolverResults."""
        # Merge kwds into options temporarily
        tee = kwds.pop('tee', self.options.get('tee', False))
        kwds.pop('keepfiles', None)  # ignored, files always kept in cwd
        lout = int(kwds.pop('lout', self.options.get('lout', 2)))
        model_kind = kwds.pop('model_kind',
                              self.options.get('model_kind', 'FL'))
        scale_rows = bool(kwds.pop('scale_rows',
                                   self.options.get('scale_rows', False)))
        ufo_dir = Path(kwds.pop('ufo_dir', self._get_ufo_dir()))

        self.available(exception_flag=True)

        # 1. Generate .ufo file
        writer = UFOWriter(lout=lout, model_kind=model_kind,
                           scale_rows=scale_rows)
        ufo_text, meta = writer.write(model)

        # 2. Set up working directory (use cwd, not temp)
        tmpdir = os.getcwd()
        try:
            self._setup_workdir(tmpdir, ufo_dir, ufo_text,
                                dat_text=meta.get('dat_text', ''),
                                map_text=meta.get('map_text', ''))

            # 3. Run pipeline
            self._run_pipeline(tmpdir, tee=tee)

            # 4. Read and parse P.OUT
            out_path = Path(tmpdir) / 'P.OUT'
            if not out_path.exists():
                raise RuntimeError(
                    "UFO produced no P.OUT output file. "
                    f"Temp directory: {tmpdir}"
                )
            out_text = out_path.read_text()
            if tee:
                print(out_text)

            parsed = UFOOutputParser().parse(out_text)

            # If UFO didn't print X=... to P.OUT (typical for large NF),
            # read the solution vector from P.SOL written by $SET(OUTPUT).
            if not parsed.get('variables'):
                sol_path = Path(tmpdir) / 'P.SOL'
                if sol_path.exists():
                    parsed['variables'] = self._parse_sol_file(sol_path)

            # P.SOL is the *final* iterate, which may be post-divergence
            # garbage (UFO blows up after touching the LP-vertex region).
            # The FMODELF trace contains every iterate UFO evaluated; pick
            # the best feasible one and prefer it over P.SOL.
            best = self._pick_best_feasible_iterate(tmpdir, meta)
            if best is not None:
                best_f, best_x, max_viol = best
                import logging as _lg
                _lg.getLogger('__main__').info(
                    f"UFO trace: best feasible iterate F={best_f:.6g} "
                    f"max_viol={max_viol:.3g} (TOLC={meta['tolc']:.1e})"
                )
                parsed['variables'] = best_x
                parsed['objective'] = best_f

        except Exception:
            raise

        return self._build_results(model, parsed, meta)

    @staticmethod
    def _parse_sol_file(path):
        """Parse P.SOL lines of the form '<index> <value>' → list of floats."""
        values = []
        for line in Path(path).read_text().splitlines():
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                idx = int(tokens[0])
            except ValueError:
                continue
            try:
                val = float(tokens[1].replace('D', 'E').replace('d', 'e'))
            except ValueError:
                continue
            # Extend with zeros if indices are not contiguous starting at 1
            while len(values) < idx - 1:
                values.append(0.0)
            if len(values) == idx - 1:
                values.append(val)
            else:
                values[idx - 1] = val
        return values

    @staticmethod
    def _pick_best_feasible_iterate(workdir, meta):
        """
        Read P.TRACE (one record per FMODELF call: F=..., index/value pairs,
        then '#END'), evaluate the linear constraint violation for each
        iterate, and return (F, X, max_viol) for the best-objective iterate
        (smallest F when minimizing, largest F when maximizing) whose max
        violation is at most TOLC. None if nothing qualifies.

        The feasibility test only covers LINEAR constraints (from the CSR data
        in meta); it cannot see nonlinear constraints. Applying it when
        nonlinear constraints are present would treat every iterate as
        feasible and could select an infeasible low-objective point, silently
        overriding UFO's converged (feasible) solution. So we bail out in that
        case and trust UFO's P.OUT / P.SOL result instead.
        """
        trace_path = Path(workdir) / 'P.TRACE'
        if not trace_path.exists():
            return None
        if not all(k in meta for k in ('icg', 'jcg', 'cgvals',
                                        'linear_meta', 'tolc', 'n_vars')):
            return None

        # Nonlinear constraints are not represented in linear_meta, so the
        # violation check below would be blind to them — don't override.
        if meta.get('n_constraints', 0) > meta.get('n_linear', 0):
            return None

        nf = meta['n_vars']
        icg = meta['icg']
        jcg = meta['jcg']
        cgvals = meta['cgvals']
        linear_meta = meta['linear_meta']
        tolc = meta['tolc']
        maximize = meta.get('sense') == 'maximize'

        def _flt(s):
            return float(s.replace('D', 'E').replace('d', 'e'))

        def _max_violation(x):
            mv = 0.0
            for kc, (_ic, eff_lb, eff_ub, kind) in enumerate(linear_meta):
                start = icg[kc] - 1
                end = icg[kc + 1] - 1
                body = 0.0
                for m in range(start, end):
                    body += cgvals[m] * x[jcg[m] - 1]
                if kind == 'eq':
                    v = abs(body - eff_lb)
                elif kind == 'lb':
                    v = max(0.0, eff_lb - body)
                elif kind == 'ub':
                    v = max(0.0, body - eff_ub)
                else:  # 'range'
                    v = max(max(0.0, eff_lb - body), max(0.0, body - eff_ub))
                if v > mv:
                    mv = v
            return mv

        best = None  # (F, X, max_viol)
        cur_f = None
        cur_x = [0.0] * nf
        with trace_path.open('r') as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('F='):
                    cur_f = _flt(line[2:].strip())
                    cur_x = [0.0] * nf
                elif line == '#END':
                    if cur_f is None:
                        continue
                    mv = _max_violation(cur_x)
                    if mv <= tolc:
                        better = (best is None
                                  or (cur_f > best[0] if maximize
                                      else cur_f < best[0]))
                        if better:
                            best = (cur_f, list(cur_x), mv)
                    cur_f = None
                else:
                    tokens = line.split()
                    if len(tokens) < 2:
                        continue
                    try:
                        i = int(tokens[0])
                        v = _flt(tokens[1])
                    except ValueError:
                        continue
                    if 1 <= i <= nf:
                        cur_x[i - 1] = v
        return best

    def _build_results(self, model, p, meta):
        """Construct a Pyomo SolverResults from parsed UFO output."""
        results = SolverResults()
        results.solver.name = 'UFO'
        results.solver.version = '.'.join(str(x) for x in self.version())

        if p['status'] == 'ok':
            results.solver.status = SolverStatus.ok
        elif p['status'] == 'warning':
            results.solver.status = SolverStatus.warning
        else:
            results.solver.status = SolverStatus.error

        tc_map = {
            'optimal':       TerminationCondition.optimal,
            'maxIterations': TerminationCondition.maxIterations,
            'maxTimeLimit':  TerminationCondition.maxTimeLimit,
            'other':         TerminationCondition.other,
        }
        results.solver.termination_condition = tc_map.get(
            p['termination_condition'], TerminationCondition.other
        )
        results.solver.termination_message = p['termination_message']
        if p['wall_time']:
            results.solver.wallclock_time = self._parse_wall_time(p['wall_time'])

        results.problem.name = getattr(model, 'name', 'unknown')
        results.problem.number_of_variables = meta['n_vars']
        results.problem.number_of_constraints = meta['n_constraints']
        results.problem.number_of_objectives = 1
        results.problem.sense = (
            ProblemSense.minimize if meta['sense'] == 'minimize'
            else ProblemSense.maximize
        )

        soln = results.solution.add()
        # Pyomo's ModelSolutions.add_solution branches on `solution._cuid`
        # (CUID-keyed vs string-keyed). We use string keys (var.name), so
        # declare _cuid as None to select that path instead of raising.
        soln._cuid = None
        soln.status = (
            SolutionStatus.optimal
            if p['status'] == 'ok' and p['termination_condition'] == 'optimal'
            else SolutionStatus.other
        )

        # Set variable values on the model
        ordered_vars = meta['ordered_vars']
        parsed_vars = p.get('variables', [])
        if parsed_vars and len(parsed_vars) == len(ordered_vars):
            for var, val in zip(ordered_vars, parsed_vars):
                var.set_value(val)
                soln.variable[var.name] = {'Value': val}
        else:
            # No usable solution vector: skip var dict rather than crashing
            # on uninitialized Pyomo vars.
            for var in ordered_vars:
                val = var.value
                if val is not None:
                    soln.variable[var.name] = {'Value': val}

        # Objective — only trust UFO's reported value. Don't fall back to
        # pyo.value(obj): if UFO failed, P.SOL may be all-zero and vars that
        # only appear in the deleted objective/constraints remain unset,
        # which would log misleading "uninitialized VarData" errors.
        obj = self._get_active_objective(model)
        if p['objective'] is not None:
            soln.objective[obj.name] = {'Value': p['objective']}

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ufo_dir(self):
        return str(self.options.get('ufo_dir', _DEFAULT_UFO_DIR))

    def _get_active_objective(self, model):
        for obj_comp in model.component_objects(pyo.Objective, active=True):
            for idx in obj_comp:
                return obj_comp[idx]
            return obj_comp

    def _setup_workdir(self, workdir, ufo_dir, ufo_text, dat_text='',
                       map_text=''):
        """Symlink required UFO files into workdir and write P.UFO."""
        workdir = Path(workdir)
        ufo_dir = Path(ufo_dir)

        # Symlink ufobel and libufo.a (skip if already present)
        for fname in _REQUIRED_FILES:
            src = ufo_dir / fname
            dst = workdir / fname
            if src.exists() and not dst.exists():
                os.symlink(src, dst)

        # Symlink all .I template files (skip if already present)
        for src in ufo_dir.glob('*.I'):
            dst = workdir / src.name
            if not dst.exists():
                os.symlink(src, dst)

        # Write the problem file
        (workdir / 'P.UFO').write_text(ufo_text)

        # Write constraint data file (read by Fortran at runtime)
        if dat_text:
            (workdir / 'CDATA.IN').write_text(dat_text)

        if map_text:
            (workdir / 'P.UFO.MAP').write_text(map_text, encoding='utf-8')

    def _run_pipeline(self, workdir, tee=False):
        """Run: ufobel → gfortran P.F -lufo → ./p"""
        workdir = str(workdir)

        # Wipe stale P.TRACE so each run starts with a clean iterate log.
        trace_path = Path(workdir) / 'P.TRACE'
        if trace_path.exists():
            trace_path.unlink()

        # Build environment with libgfortran.so.3 on LD_LIBRARY_PATH if needed
        env = os.environ.copy()
        # lg3_dir = _find_libgfortran3()
        # if lg3_dir:
        #     existing = env.get('LD_LIBRARY_PATH', '')
        #     env['LD_LIBRARY_PATH'] = (lg3_dir + ':' + existing) if existing else lg3_dir

        capture = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if tee:
            capture = {}  # let stdout/stderr flow to console

        # Step 1: ufobel (preprocessor). ufobel returns rc=0 even after
        # `FATAL ERROR — EXECUTION ABORTED`, which would silently leave a
        # stale P.F in place. Detect the abort by scanning stdout.
        r = subprocess.run(
            ['./ufobel'], cwd=workdir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        ufobel_out = r.stdout.decode(errors='replace')
        ufobel_err = r.stderr.decode(errors='replace')
        if r.returncode != 0 or 'FATAL ERROR' in ufobel_out or 'EXECUTION ABORTED' in ufobel_out:
            raise RuntimeError(
                f"ufobel aborted (rc={r.returncode}):\n"
                f"--- stdout ---\n{ufobel_out}\n"
                f"--- stderr ---\n{ufobel_err}"
            )

        # Step 2: compile P.F  (-no-pie needed: libufo.a built without -fPIC)
        r = subprocess.run(
            [_GFORTRAN, 'P.F', '-std=legacy', '-w', '-no-pie', '-fno-automatic',
             '-o', 'p', '-L.', '-lufo'],
            cwd=workdir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if r.returncode != 0:
            pf = (Path(workdir) / 'P.F').read_text() if (Path(workdir) / 'P.F').exists() else ''
            err = r.stderr.decode(errors='replace')
            raise RuntimeError(
                f"gfortran compilation failed (rc={r.returncode}):\n{err}\n"
                f"--- Generated P.F ---\n{pf}"
            )

        # Step 3: run optimizer
        r = subprocess.run(['./p'], cwd=workdir, env=env, **capture)
        if r.returncode != 0 and not (Path(workdir) / 'P.OUT').exists():
            err = r.stderr.decode(errors='replace') if hasattr(r, 'stderr') else ''
            raise RuntimeError(f"UFO execution failed (rc={r.returncode}):\n{err}")

    @staticmethod
    def _parse_wall_time(s):
        """Convert 'h:mm:ss.ss' to total seconds (float)."""
        try:
            parts = s.split(':')
            if len(parts) == 3:
                h, m, sec = parts
                return int(h) * 3600 + int(m) * 60 + float(sec)
            return 0.0
        except Exception:
            return 0.0
