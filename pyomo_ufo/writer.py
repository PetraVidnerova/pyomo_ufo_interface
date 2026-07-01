"""
writer.py
---------
Convert a Pyomo ConcreteModel (or solved AbstractModel instance) into a UFO
input file (.ufo format) suitable for the ``ufobel`` preprocessor.

Supported problem types
~~~~~~~~~~~~~~~~~~~~~~~
- Unconstrained NLP
- Box-constrained NLP
- NLP with linear equality/inequality constraints
- NLP with nonlinear equality/inequality constraints

All problems use ``$MODEL='FF'`` (general nonlinear objective).

Usage::

    from pyomo_ufo.writer import UFOWriter
    writer = UFOWriter()
    ufo_text, meta = writer.write(model)
    # meta['ordered_vars'] is the list of VarData in the order X(1)..X(NF)
"""

import io
import math

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables, polynomial_degree
from pyomo.core import value as pyo_value
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.repn import generate_standard_repn
from .fortran_expr import to_fortran, _fortran_const


_VALID_MODEL_KINDS = ('FL', 'FF')


class UFOWriter:
    """
    Translate a Pyomo model to a UFO ``.ufo`` input file.

    Parameters
    ----------
    lout : int
        UFO output level (0=minimal, 2=standard, 4=verbose). Default 2.
    model_kind : str
        UFO ``$MODEL`` directive: ``'FL'`` for linear objective (LP/QP),
        ``'FF'`` for general nonlinear objective (NLP). Default ``'FL'``.
    scale_rows : bool
        If True, divide each linear constraint row (coefficients and bounds)
        by its max magnitude so no row exceeds 1.0. If False, write the model
        in original units (all scales = 1.0). Default ``False``.
    """

    def __init__(self, lout=2, model_kind='FL', scale_rows=False):
        if model_kind not in _VALID_MODEL_KINDS:
            raise ValueError(
                f"model_kind must be one of {_VALID_MODEL_KINDS}, got {model_kind!r}"
            )
        self.lout = lout
        self.model_kind = model_kind
        self.scale_rows = scale_rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, model):
        """
        Generate UFO input for *model*.

        Parameters
        ----------
        model : Pyomo ConcreteModel

        Returns
        -------
        ufo_text : str
            Complete contents of the ``.ufo`` file.
        meta : dict
            ``ordered_vars``  – list of VarData in Fortran order (X(1)..X(NF))
            ``var_map``       – ``{id(v): 'X(i)'}``
            ``n_vars``        – NF
            ``n_constraints`` – NC
            ``n_linear``      – NCL
            ``n_equality``    – NCE
            ``sense``         – 'minimize' or 'maximize'
        """
        buf = io.StringIO()

        # ---- Objective -----------------------------------------------
        obj = self._get_objective(model)
        sense = 'minimize' if obj.sense == pyo.minimize else 'maximize'

        # ---- Variables -----------------------------------------------
        ordered_vars = self._collect_vars(model, obj)
        self._last_ordered_vars = ordered_vars
        nf = len(ordered_vars)
        var_map = {id(v): f'X({i+1})' for i, v in enumerate(ordered_vars)}

        # ---- Constraints ---------------------------------------------
        constraints = self._collect_constraints(model)
        linear_cons, nonlinear_cons = self._classify_constraints(constraints)
        nc = len(linear_cons) + len(nonlinear_cons)
        nx = nf # fix pro jednoduchost 
        ncl = len(linear_cons)
        nce = sum(1 for *_, kind in linear_cons + nonlinear_cons
                  if kind == 'eq')

        # MODEL='FL' declares the objective LINEAR: its gradient is the
        # constant GF vector written in $INPUT and no FMODELF block is
        # emitted. That is only valid when the objective really is linear.
        # A nonlinear objective under FL is silently mistranslated (the
        # quadratic/nonlinear part is dropped, GF becomes the constant linear
        # part or 0), so UFO optimizes the wrong function. Detect a nonlinear
        # objective and upgrade FL -> FF, which emits FMODELF with the full
        # expression (gradient by UFO's finite differences).
        obj_degree = polynomial_degree(obj.expr)
        obj_is_nonlinear = obj_degree is None or obj_degree > 1

        # UFO's constrained methods (selected whenever there are general
        # nonlinear constraints) evaluate the objective *model function*
        # FMODF during the iteration, even when the objective itself is
        # linear. MODEL='FL' emits no FMODELF block, so ufobel aborts in
        # UP1FF5.I with "FMODF MACRO NOT DEFINED". Upgrade FL -> FF when
        # nonlinear constraints are present: the FF path emits FMODELF plus
        # the analytical gradient GMODELF, so a linear objective is still
        # handled exactly (no finite-difference fallback, no wrong F=0).
        if self.model_kind == 'FL' and (nonlinear_cons or obj_is_nonlinear):
            reason = ('nonlinear objective'
                      if obj_is_nonlinear
                      else f'{len(nonlinear_cons)} nonlinear constraint(s)')
            import logging as _lg
            _lg.getLogger('__main__').info(
                f"UFO writer: {reason} present; "
                "upgrading objective model FL -> FF (FMODELF required)."
            )
            self.model_kind = 'FF'

        # Build sparse Jacobian in CSR format (ICG/JCG/CG)
        var_id_to_idx = {id(v): i for i, v in enumerate(ordered_vars)}
        icg = []     # row pointers (1-based), length nc+1
        jcg = []     # column indices (1-based)
        cgvals = []  # CG values
        linear_meta = []  # [(ic, eff_lb, eff_ub, kind)]

        for kc, (name, body, lb, ub, kind) in enumerate(linear_cons):
            repn = generate_standard_repn(body)
            const = float(repn.constant) if repn.constant else 0.0
            icg.append(len(jcg) + 1)
            if repn.linear_vars:
                entries = []
                for coef, var in zip(repn.linear_coefs, repn.linear_vars):
                    idx = var_id_to_idx.get(id(var))
                    if idx is not None and float(coef) != 0.0:
                        entries.append((idx + 1, float(coef)))
                entries.sort()  # CSR requires sorted column indices
                for col, val in entries:
                    jcg.append(col)
                    cgvals.append(val)
            ic = self._ic_flag(kind)
            eff_lb = (lb - const) if lb is not None else None
            eff_ub = (ub - const) if ub is not None else None
            linear_meta.append((ic, eff_lb, eff_ub, kind))

        for k, (name, body, lb, ub, kind) in enumerate(nonlinear_cons):
            icg.append(len(jcg) + 1)
            vlist = list(identify_variables(body, include_fixed=False))
            cols = sorted(set(
                var_id_to_idx[id(v)] + 1
                for v in vlist if id(v) in var_id_to_idx
            ))
            for col in cols:
                jcg.append(col)
                cgvals.append(0.0)

        icg.append(len(jcg) + 1)  # ICG(NC+1) sentinel
        mc = len(jcg)

        # Row scaling for LINEAR constraints: divide each row's coefficients
        # and RHS by max(|coef|, |rhs|) so no row has magnitude > 1. Without
        # this, mixed units (power in kW, energy in MWh, prices in Kc/MWh)
        # produce penalty gradients spanning ~5 orders of magnitude, which
        # breaks UFO's VL-LI3 interior-point L-BFGS step. Nonlinear rows
        # (y*(1-y)=0, start*(1-start)=0) are already unit-scale.
        _scaled = 0
        _max_row_norm_before = 0.0
        # Per-row divisor actually applied; 1.0 means the row was left alone.
        # Recorded for the P.UFO.MAP sidecar so users can reconcile UFO
        # residuals with Pyomo constraint slack.
        linear_scale = [1.0] * len(linear_cons)
        for kc in range(len(linear_cons)) if self.scale_rows else ():
            start = icg[kc] - 1
            end = icg[kc + 1] - 1
            ic, eff_lb, eff_ub, kind = linear_meta[kc]
            row_max = 0.0
            for m in range(start, end):
                a = abs(cgvals[m])
                if a > row_max:
                    row_max = a
            if eff_lb is not None and abs(eff_lb) > row_max:
                row_max = abs(eff_lb)
            if eff_ub is not None and abs(eff_ub) > row_max:
                row_max = abs(eff_ub)
            if row_max > _max_row_norm_before:
                _max_row_norm_before = row_max
            if row_max <= 1.0:
                continue
            inv = 1.0 / row_max
            for m in range(start, end):
                cgvals[m] *= inv
            if eff_lb is not None:
                eff_lb *= inv
            if eff_ub is not None:
                eff_ub *= inv
            linear_meta[kc] = (ic, eff_lb, eff_ub, kind)
            linear_scale[kc] = inv
            _scaled += 1
        if _scaled:
            import logging as _lg
            _lg.getLogger('__main__').info(
                f"UFO writer: scaled {_scaled}/{len(linear_cons)} linear rows "
                f"(max pre-scaling row norm = {_max_row_norm_before:.3g})"
            )

        externalize = mc > self._MAX_INLINE_CG
        externalize_nl = (bool(nonlinear_cons)
                          and len(nonlinear_cons) > self._MAX_INLINE_NL)
        needs_data_file = externalize or externalize_nl

        # ---- Box bounds ---------------------------------------------
        # nx = 0
        # for v  in ordered_vars:
        #     if v.has_lb() != v.has_ub():
        #         nx += 1
        
        has_lb = any(v.has_lb() for v in ordered_vars)
        has_ub = any(v.has_ub() for v in ordered_vars)
        if has_lb and has_ub:
            kbf = 2
        elif has_lb or has_ub:
            kbf = 1
        else:
            kbf = 0

        # ---- Constraint bound type ----------------------------------
        if nc == 0:
            kbc = 0
        else:
            has_one_sided = any(
                kind in ('lb', 'ub')
                for *_, kind in linear_cons + nonlinear_cons
            )
            has_two_sided = any(
                kind == 'range'
                for *_, kind in linear_cons + nonlinear_cons
            )
            kbc = 2 if has_two_sided else 1

        # ==============================================================
        # Write .ufo file
        # ==============================================================
        w = buf.write

        w('$REM UFO input generated by pyomo_ufo\n')

        # Short Pyomo<->UFO mapping summary. Full table lives in P.UFO.MAP.
        w(f'$REM Pyomo<->UFO map: NF={nf}, NC={nc} '
          f'(NCL={ncl}, nonlinear={len(nonlinear_cons)})\n')
        w('$REM See sidecar P.UFO.MAP for full X(i)/KC=k -> Pyomo name table\n')

        # I is used by compressed DO-loop emission in $SET(INPUT) (and by
        # the OUTPUT block). K is used only by the CDATA.IN reader.
        w("$ADD(INTEGER,'\\I')\n")
        if needs_data_file:
            w("$ADD(INTEGER,'\\K')\n")

        # ---- $SET(INPUT) -------------------------------------------
        obj_repn = generate_standard_repn(obj.expr)
        
        self._write_input_block(w, ordered_vars, var_map, nf,
                                linear_cons, nc, kbf, kbc, nonlinear_cons,
                                obj_repn, 
                                icg=icg, jcg=jcg, cgvals=cgvals, mc=mc,
                                linear_meta=linear_meta,
                                externalize=externalize,
                                externalize_nl=externalize_nl)

        # ---- $SET(FMODELF) — required when MODEL='FF' --------------
        if self.model_kind == 'FF':
            self._write_fmodelf_block(w, obj.expr, var_map)
            # Analytical gradient for a linear objective. Without this,
            # UFO finite-differences the gradient (nf F-evals per iter),
            # which is what caused MFV TOO SMALL for this problem.
            obj_is_linear = (
                obj_repn.nonlinear_expr is None
                and not (getattr(obj_repn, 'quadratic_vars', None) or [])
            )
            if obj_is_linear:
                w('$SET(GMODELF)\n')
                self._write_gmodelf_block(w, obj_repn, var_map, nf)
                w('$ENDSET\n')

        # ---- $SET(FMODELCS) (nonlinear constraints only) -----------
        if nonlinear_cons:
            self._write_fmodelcs_block(w, nonlinear_cons,
                                       linear_cons, var_map)

        # ---- $SET(OUTPUT) — dump solution X to P.SOL ---------------
        # UFO doesn't print X=... to P.OUT for large NF, so write it
        # ourselves via the OUTPUT hook which runs after the optimizer.
        w('$SET(OUTPUT)\n')
        w("  OPEN(91,FILE='P.SOL',STATUS='UNKNOWN')\n")
        w('  DO I=1,NF\n')
        w('  WRITE(91,*) I,X(I)\n')
        w('  ENDDO\n')
        w('  CLOSE(91)\n')
        w('$ENDSET\n')

        # ---- Control directives ------------------------------------
        w(f'$NF={nf}\n')
        if nx > 0:
            w(f'$NX={nx}\n')
        if nc > 0:
            w(f'$NC={nc}\n')
            w("$JACC='S'\n")
            w(f'$MC={mc}\n')
            #w("$FORM='SI'\n")
            # MMAX sizes the A/JA arrays used by UXSPCT for sparse-factor
            # fill-in. Initial NNZ is 10*(nf+nc); LDL fill-in can be many
            # times larger, which triggered UXSPCT(43) LACK OF SPACE.
            w(f'$MMAX={max(1_000_000, 100 * (nf + nc))}\n')
        if ncl > 0:
            w(f'$NCL={ncl}\n')
        # if nce > 0:
        #     w(f'$NCE={nce}\n')
        w(f'$KBF={kbf}\n')
        if nc > 0:
            w(f'$KBC={kbc}\n')
        # CD = constrained-descent; handles MODEL='FF' with general NC
        # and box bounds. Default VL (limited-memory VM) runs penalty
        # iterations that explode the objective and ignore XL/XU.
        if self.model_kind == 'FF' and nc > 0:
            w("$CLASS='CD'\n")
        w(f"$MODEL='{self.model_kind}'\n")
        w("$NZ=50000\n")
        w("$NAU=50000\n")
        w("$NZA=50000\n")
        # MD = max # of dense rows in UCISD1. Default is 10, but this
        # problem produces 118 dense rows once row-scaling has been
        # applied. 0 tells ufobel to auto-size.
        w("$MD=0\n")
        # TOLC = feasibility tolerance. UFO's default is ~1e-6 which is
        # tighter than this problem can achieve with L-BFGS + penalty
        # (we converge to C~2.4e-3 at iter ~39 and then diverge if forced
        # to push tighter). Relaxing stops UFO at the feasible optimum.
        w("$TOLC='5.0$P-3'\n")

    
        if sense == 'maximize':
            w('$IEXT=1\n')
        # if use_analytical_grad:
        #     w('$KDF=1\n')
        # MFV raised from 20*nf to 2000*nf: with finite-difference gradients
        # each outer iteration costs ~nf F-evals, so 20*nf only bought ~20
        # outer iterations — far short of what the penalty loop needs.
        w(f'$MFV={max(100000, 2000 * nf)}\n')
        w(f'$MIT={max(5000, 10 * nf)}\n')
        w(f'$LOUT={self.lout}\n')
        w('$BATCH\n')
        w('$STANDARD\n')


        # ---- Data file (sparse Jacobian and/or bilinear pairs) ----
        dat_parts = []
        if externalize and mc > 0:
            lines = [str(nc + 1)]  # number of ICG entries
            for ptr in icg:
                lines.append(str(ptr))
            lines.append(str(mc))  # number of nonzero entries
            for m in range(mc):
                lines.append(f'{jcg[m]} {cgvals[m]:.15e}')
            dat_parts.append('\n'.join(lines))
        if hasattr(self, '_bilinear_pairs') and self._bilinear_pairs:
            lines = [str(len(self._bilinear_pairs))]
            for (idx,) in self._bilinear_pairs:
                lines.append(str(idx))
            dat_parts.append('\n'.join(lines))
            self._bilinear_pairs = None
        dat_text = '\n'.join(dat_parts) + '\n' if dat_parts else ''

        map_text = self._build_map_text(
            ordered_vars, linear_cons, linear_meta, linear_scale,
            nonlinear_cons, nf, nc, ncl)

        meta = {
            'ordered_vars': ordered_vars,
            'var_map': var_map,
            'n_vars': nf,
            'n_constraints': nc,
            'n_linear': ncl,
            'n_equality': nce,
            'sense': sense,
            'dat_text': dat_text,
            'map_text': map_text,
            # Linear constraint matrix in scaled CSR form (post row scaling).
            # Used by the solver to pick the best feasible iterate from
            # P.TRACE when UFO ends in a divergent state.
            'icg': list(icg),
            'jcg': list(jcg),
            'cgvals': list(cgvals),
            'linear_meta': list(linear_meta),
            'tolc': 5e-3,
        }
        return buf.getvalue(), meta

    @staticmethod
    def _safe_name(name):
        # Tab/newline in a Pyomo name would corrupt the TSV-shaped sidecar.
        return str(name).replace('\t', ' ').replace('\n', ' ')

    @staticmethod
    def _build_map_text(ordered_vars, linear_cons, linear_meta, linear_scale,
                        nonlinear_cons, nf, nc, ncl):
        """
        Assemble the P.UFO.MAP sidecar text: header, X(i) -> Pyomo var name,
        KC=k -> (IC, kind, scale, Pyomo constraint name).
        """
        lines = [
            '# UFO <-> Pyomo name mapping for P.UFO',
            '# Generated by pyomo_ufo.UFOWriter',
            f'# Variables: NF={nf}',
            f'# Constraints: NC={nc} (linear NCL={ncl}, '
            f'nonlinear={len(nonlinear_cons)})',
            '#',
            '# Variables (X(i) -> pyomo name):',
        ]
        for i, v in enumerate(ordered_vars, start=1):
            lines.append(f'X({i})\t{UFOWriter._safe_name(v.name)}')
        lines.append('#')
        lines.append('# Constraints (KC -> IC, kind, scale, pyomo name):')
        for kc, ((name, _body, _lb, _ub, kind),
                (ic, _elb, _eub, _kind2)) in enumerate(
                zip(linear_cons, linear_meta), start=1):
            scale = linear_scale[kc - 1]
            lines.append(
                f'KC={kc}\tIC={ic}\tkind={kind}\tscale={scale!r}\t'
                f'{UFOWriter._safe_name(name)}')
        offset = len(linear_cons)
        for k, (name, _body, _lb, _ub, kind) in enumerate(nonlinear_cons):
            kc = offset + k + 1
            ic = UFOWriter._ic_flag(kind)
            lines.append(
                f'KC={kc}\tIC={ic}\tkind={kind}\tscale=1.0\t'
                f'{UFOWriter._safe_name(name)}')
        return '\n'.join(lines) + '\n'

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_objective(self, model):
        objectives = list(model.component_objects(pyo.Objective, active=True))
        if len(objectives) == 0:
            raise ValueError("Model has no active objective.")
        if len(objectives) > 1:
            raise ValueError("UFO solver supports only a single objective.")
        obj = objectives[0]
        # Return the (scalar) objective data object
        for idx in obj:
            return obj[idx]
        return obj  # ScalarObjective

    def _collect_vars(self, model, obj):
        """
        Return an ordered list of all active, non-fixed VarData objects.

        Order: variables appear in objective first, then in constraints,
        deduplication preserving first-seen order.
        """
        seen = {}  # id(v) -> v, preserving insertion order

        def _add(expr):
            for v in identify_variables(expr, include_fixed=False):
                if id(v) not in seen:
                    seen[id(v)] = v

        _add(obj.expr)
        for con in model.component_objects(pyo.Constraint, active=True):
            for idx in con:
                c = con[idx]
                if not c.active:
                    continue
                _add(c.body)

        return list(seen.values())

    def _collect_constraints(self, model):
        """Return list of (name, body_expr, lb, ub) for all active constraints."""
        cons = []
        for con_comp in model.component_objects(pyo.Constraint, active=True):
            for idx in con_comp:
                c = con_comp[idx]
                if not c.active:
                    continue
                lb = pyo_value(c.lower) if c.lower is not None else None
                ub = pyo_value(c.upper) if c.upper is not None else None
                name = f'{con_comp.name}[{idx}]' if idx is not None else con_comp.name
                cons.append((name, c.body, lb, ub))
        return cons

    def _classify_constraints(self, constraints):
        """
        Split constraints into linear and nonlinear.

        Each entry is (name, body_expr, lb, ub, kind)
        where kind ∈ {'eq', 'lb', 'ub', 'range'}.
        """
        linear = []
        nonlinear = []

        for name, body, lb, ub in constraints:
            deg = polynomial_degree(body)
            kind = self._bound_kind(lb, ub)
            entry = (name, body, lb, ub, kind)
            if deg is not None and deg <= 1:
                linear.append(entry)
            else:
                nonlinear.append(entry)

        return linear, nonlinear

    @staticmethod
    def _bound_kind(lb, ub):
        if lb is not None and ub is not None:
            if abs(lb - ub) < 1e-14:
                return 'eq'
            return 'range'
        if lb is not None:
            return 'lb'
        return 'ub'

    # ------------------------------------------------------------------
    # Block writers
    # ------------------------------------------------------------------

    # Max non-zero CG entries before externalizing to data file
    _MAX_INLINE_CG = 500

    def _write_input_block(self, w, ordered_vars, var_map, nf,
                           linear_cons, nc, kbf, kbc=0, 
                           nonlinear_cons=None, obj_repn=None,
                           icg=None, jcg=None, cgvals=None, mc=0,
                           linear_meta=None,
                           externalize=False, externalize_nl=False):
        w('$SET(INPUT)\n')

        # Starting point — use Pyomo-provided values, or 0 if missing.
        # The previous pure-Python projection onto linear equalities was
        # disabled: its Gaussian elimination on a near-singular 2551x2551
        # A*A^T produced garbage starting values (e.g. 642.857... for 700+
        # vars) that drove UFO's L-BFGS to explode.
        x0_vals = []
        for v in ordered_vars:
            val = v.value
            x0_vals.append(float(val) if val is not None else 0.0)

        x_entries = [
            (i + 1, (('X', _fortran_const(x0)),))
            for i, x0 in enumerate(x0_vals) if x0 != 0.0
        ]
        self._emit_indexed(w, x_entries,
                           singleton=lambda i, p: (
                               ''.join(f'  {n}({i}) = {v}\n' for n, v in p)))

        # Box bounds
        if kbf > 0:
            box_entries = []
            for i, v in enumerate(ordered_vars):
                ix = self._ix_flag(v)
                if ix == 0:
                    continue
                payload = [('IX', str(ix))]
                if v.has_lb():
                    payload.append(('XL', _fortran_const(pyo_value(v.lb))))
                if v.has_ub():
                    payload.append(('XU', _fortran_const(pyo_value(v.ub))))
                box_entries.append((i + 1, tuple(payload)))
            self._emit_indexed(w, box_entries,
                               singleton=lambda i, p: '  ' + '; '.join(
                                   f'{n}({i}) = {v}' for n, v in p) + '\n')

        # Constraint setup: IC, CL, CU
        con_entries = []
        if linear_cons and linear_meta:
            for kc, (ic, eff_lb, eff_ub, kind) in enumerate(linear_meta):
                payload = [('IC', str(ic))]
                payload.extend(self._build_bounds_payload(
                    eff_lb, eff_ub, kind, kbc))
                con_entries.append((kc + 1, tuple(payload)))

        if nonlinear_cons:
            offset = len(linear_cons)
            for k, (name, body, lb, ub, kind) in enumerate(nonlinear_cons):
                kc = offset + k + 1
                ic = self._ic_flag(kind)
                payload = [('IC', str(ic))]
                payload.extend(self._build_bounds_payload(lb, ub, kind, kbc))
                con_entries.append((kc, tuple(payload)))

        self._emit_indexed(w, con_entries,
                           singleton=lambda i, p: (
                               ''.join(f'  {n}({i}) = {v}\n' for n, v in p)))

        # Sparse Jacobian (CSR): ICG, JCG, CG arrays
        if nc > 0 and icg is not None and not externalize:
            for k in range(nc + 1):
                w(f'  ICG({k+1})={icg[k]}\n')
            for m in range(mc):
                w(f'  JCG({m+1})={jcg[m]}\n')
                if cgvals[m] != 0.0:
                    w(f'  CG({m+1})={_fortran_const(cgvals[m])}\n')

        # Read externalized data from CDATA.IN
        if externalize or externalize_nl:
            w("  OPEN(99,FILE='CDATA.IN',STATUS='OLD')\n")
            if externalize:
                w("  READ(99,*) K\n")
                w("  DO 9985 I=1,K\n")
                w("  READ(99,*) ICG(I)\n")
                w("9985 CONTINUE\n")
                w("  READ(99,*) K\n")
                w("  DO 9986 I=1,K\n")
                w("  READ(99,*) JCG(I),CG(I)\n")
                w("9986 CONTINUE\n")
            if externalize_nl:
                w("  READ(99,*) K\n")
                w("  DO 9987 I=1,K\n")
                w("  READ(99,*) BL1(I)\n")
                w("9987 CONTINUE\n")
            w("  CLOSE(99)\n")

        # GF(i) in INPUT is the constant gradient of a linear objective
        # (MODEL='FL'). For MODEL='FF' the objective is defined in FMODELF
        # and GF is either computed by finite differences or in GMODELF.
        if self.model_kind == 'FL':
            self._write_gmodelf_block(w, obj_repn, var_map, nf)
        w('$ENDSET\n')

    def _write_fmodelf_block(self, w, obj_expr, var_map):
        w('$SET(FMODELF)\n')
        expr_str = to_fortran(obj_expr, var_map)
        if len(expr_str) <= 500:
            w(f'  FF = {expr_str}\n')
        else:
            # Long expression: use chunked accumulation via repn
            repn = generate_standard_repn(obj_expr)
            const = float(repn.constant) if repn.constant else 0.0
            w(f'  FF = {_fortran_const(const)}\n')
            if repn.linear_vars:
                for coef, var in zip(repn.linear_coefs, repn.linear_vars):
                    vn = var_map.get(id(var))
                    if vn is None:
                        continue
                    c = float(coef)
                    if c < 0:
                        w(f'  FF=FF+({_fortran_const(c)})*{vn}\n')
                    else:
                        w(f'  FF=FF+{_fortran_const(c)}*{vn}\n')
            if hasattr(repn, 'quadratic_vars') and repn.quadratic_vars:
                for coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
                    vn1 = var_map.get(id(v1), '?')
                    vn2 = var_map.get(id(v2), '?')
                    w(f'  FF = FF + {_fortran_const(float(coef))}*{vn1}*{vn2}\n')
            if repn.nonlinear_expr is not None:
                nl_str = to_fortran(repn.nonlinear_expr, var_map)
                w(f'  FF = FF + ({nl_str})\n')
        # Append the iterate (F, X) to P.TRACE so the Python side can pick
        # the best feasible iterate even if UFO ends in a divergent state.
        w("  OPEN(94,FILE='P.TRACE',STATUS='UNKNOWN',POSITION='APPEND')\n")
        w("  WRITE(94,'(A,1PE24.16)') 'F=', FF\n")
        w('  DO I=1,NF\n')
        w("  WRITE(94,'(I8,1X,1PE24.16)') I, X(I)\n")
        w('  ENDDO\n')
        w("  WRITE(94,'(A)') '#END'\n")
        w('  CLOSE(94)\n')
        w('$ENDSET\n')

    def _write_gmodelf_block(self, w, repn, var_map, nf):
        """Write analytical gradient of the objective (GMODELF block)."""
        #w('$SET(GMODELF)\n')
        # Build coefficient map: var index -> coefficient
        grad = {}
        if repn.linear_vars:
            for coef, var in zip(repn.linear_coefs, repn.linear_vars):
                vn = var_map.get(id(var))
                if vn is not None:
                    # Extract index from 'X(i)'
                    idx = int(vn[2:-1])
                    grad[idx] = float(coef)
        # TODO in future: nonlinear part of gradient objective (currently not supported, so zero entries will be written)
        # Write all GF entries (zero for vars not in objective)
        gf_entries = [
            (i, (('GF', _fortran_const(grad.get(i, 0.0))),))
            for i in range(1, nf + 1)
        ]
        self._emit_indexed(w, gf_entries,
                           singleton=lambda i, p: (
                               ''.join(f'  {n}({i})={v}\n' for n, v in p)))
        #w('$ENDSET\n')

    # Max nonlinear constraints before switching to data-driven approach
    _MAX_INLINE_NL = 50

    def _classify_nl_shape(self, body, var_id_to_idx):
        """
        Verify a nonlinear constraint body has the binary-complementarity
        shape ``y*(1-y) == y - y**2`` and return the 1-based variable index.
        Raises NotImplementedError for any other shape.
        """
        repn = generate_standard_repn(body)
        if repn.nonlinear_expr is not None:
            raise NotImplementedError(
                f'Unsupported nonlinear constraint (non-quadratic): {body}'
            )
        const = float(repn.constant) if repn.constant else 0.0
        qvars = list(repn.quadratic_vars) if repn.quadratic_vars else []
        qcoefs = [float(c) for c in (repn.quadratic_coefs or [])]
        lvars = list(repn.linear_vars) if repn.linear_vars else []
        lcoefs = [float(c) for c in (repn.linear_coefs or [])]

        if abs(const) > 1e-14 or len(qvars) != 1 or len(lvars) != 1:
            raise NotImplementedError(
                f'Unsupported nonlinear constraint shape: {body}'
            )
        v1, v2 = qvars[0]
        if (id(v1) == id(v2) and id(lvars[0]) == id(v1)
                and abs(lcoefs[0] - 1.0) < 1e-14
                and abs(qcoefs[0] + 1.0) < 1e-14):
            idx = var_id_to_idx.get(id(v1), 0)
            if idx == 0:
                raise NotImplementedError(
                    f'Variable not in index map for constraint: {body}'
                )
            return idx

        raise NotImplementedError(
            f'Unsupported nonlinear constraint shape '
            f'(expected y*(1-y)): {body}'
        )

    def _write_fmodelcs_block(self, w, nonlinear_cons, linear_cons, var_map):
        offset = len(linear_cons)
        if len(nonlinear_cons) <= self._MAX_INLINE_NL:
            # Small number: use IF/ELSEIF branches
            w('$SET(FMODELC)\n')
            w('  IF(KC.LE.0)THEN\n')
            for k, (name, body, lb, ub, kind) in enumerate(nonlinear_cons):
                kc = offset + k + 1
                w(f'  ELSEIF(KC.EQ.{kc})THEN\n')
                w(f'    FC={to_fortran(body, var_map)}\n')
            w('  ENDIF\n')
            w('$ENDSET\n')
        else:
            # Large number: data-driven. Every nonlinear constraint is
            # binary complementarity y*(1-y)==0, so we only need the
            # variable index per constraint.
            var_id_to_idx = {id(v): i + 1
                             for i, v in enumerate(
                                 self._last_ordered_vars)}
            self._bilinear_pairs = []
            for name, body, lb, ub, kind in nonlinear_cons:
                idx = self._classify_nl_shape(body, var_id_to_idx)
                self._bilinear_pairs.append((idx,))

            n_nl = len(nonlinear_cons)
            w(f"$ADD(INTEGER,'\\BL1({n_nl})')\n")
            w('$SET(FMODELC)\n')
            w(f'  FC=X(BL1(KC-{offset}))'
              f'-X(BL1(KC-{offset}))*X(BL1(KC-{offset}))\n')
            w('$ENDSET\n')

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ix_flag(v):
        """Return UFO IX flag for variable bound type."""
        has_lb = v.has_lb()
        has_ub = v.has_ub()
        if has_lb and has_ub:
            lb, ub = pyo_value(v.lb), pyo_value(v.ub)
            if abs(lb - ub) < 1e-14:
                return 5  # fixed
            return 3  # two-sided
        if has_lb:
            return 1
        if has_ub:
            return 2
        return 0  # free

    @staticmethod
    def _ic_flag(kind):
        """Return UFO IC flag for constraint type."""
        return {'eq': 5, 'lb': 1, 'ub': 2, 'range': 3}.get(kind, 1)

    # Minimum run length to collapse into a DO loop. A DO/ENDDO pair costs
    # 2 extra lines, so 4 is the break-even for single-assignment payloads.
    _MIN_DO_RUN = 4

    @staticmethod
    def _group_runs(entries):
        """
        Group (i, payload) pairs into maximal runs where i increments by 1
        and payload is identical. Yields lists (runs) in order.
        """
        run = []
        for entry in entries:
            i, p = entry
            if run and i == run[-1][0] + 1 and p == run[-1][1]:
                run.append(entry)
            else:
                if run:
                    yield run
                run = [entry]
        if run:
            yield run

    def _emit_indexed(self, w, entries, singleton):
        """
        Emit a sequence of indexed array assignments, collapsing runs of
        consecutive indices with identical payload into a DO loop.

        entries: iterable of (i, payload) where payload is a tuple of
                 (array_name, fortran_value_str) pairs.
        singleton: callable(i, payload) -> str, rendering one entry when
                   it is not part of a compressible run.
        """
        for run in self._group_runs(entries):
            if len(run) >= self._MIN_DO_RUN:
                i0, i1 = run[0][0], run[-1][0]
                w(f'  DO I={i0},{i1}\n')
                for name, val in run[0][1]:
                    w(f'  {name}(I)={val}\n')
                w('  ENDDO\n')
            else:
                for i, payload in run:
                    w(singleton(i, payload))

    @staticmethod
    def _build_bounds_payload(lb, ub, kind, kbc):
        """
        Return list of (array_name, fortran_value_str) pairs corresponding
        to the CL/CU assignments that _write_bounds would emit. Mirrors the
        branching in _write_bounds exactly.
        """
        out = []
        if kind == 'eq':
            if lb is not None:
                out.append(('CL', _fortran_const(lb)))
        elif kind == 'lb':
            if lb is not None:
                out.append(('CL', _fortran_const(lb)))
        elif kind == 'ub':
            if ub is not None:
                if kbc >= 2:
                    out.append(('CU', _fortran_const(ub)))
                else:
                    out.append(('CL', _fortran_const(ub)))
        elif kind == 'range':
            if lb is not None:
                out.append(('CL', _fortran_const(lb)))
            if ub is not None:
                out.append(('CU', _fortran_const(ub)))
        return out

    @staticmethod
    def _write_bounds(w, kc, lb, ub, kind, kbc):
        """
        Write CL/CU lines for constraint kc.

        UFO convention:
        - IC=5 (eq):    CL(kc) = lb  (CU alias CL, only one needed)
        - IC=1 (lb):    CL(kc) = lb
        - IC=2 (ub):    CL(kc) = ub  when kbc=1 (CU is aliased to CL)
                        CU(kc) = ub  when kbc=2 (CU is separate array)
        - IC=3 (range): CL(kc) = lb; CU(kc) = ub  (only when kbc=2)
        """
        if kind == 'eq':
            if lb is not None:
                w(f'  CL({kc}) = {_fortran_const(lb)}\n')
        elif kind == 'lb':
            if lb is not None:
                w(f'  CL({kc}) = {_fortran_const(lb)}\n')
        elif kind == 'ub':
            if ub is not None:
                if kbc >= 2:
                    w(f'  CU({kc}) = {_fortran_const(ub)}\n')
                else:
                    # CU is aliased to CL for KBC=1
                    w(f'  CL({kc}) = {_fortran_const(ub)}\n')
        elif kind == 'range':
            if lb is not None:
                w(f'  CL({kc}) = {_fortran_const(lb)}\n')
            if ub is not None:
                w(f'  CU({kc}) = {_fortran_const(ub)}\n')

    def _project_onto_equalities(self, x0_vals, eq_cons, ordered_vars):
        """
        Project starting point x0 onto the feasible hyperplane defined by
        linear equality constraints, using the minimum-norm perturbation:

            x_proj = x0 + A^T (A A^T)^{-1} (b - A x0)

        Falls back silently to the original x0 if projection fails.
        """
        try:
            n = len(x0_vals)
            A_rows, b_vals = [], []
            for (name, body, lb, ub, kind) in eq_cons:
                coeffs, const = self._extract_linear(body, ordered_vars)
                A_rows.append(coeffs)
                rhs = lb if lb is not None else ub
                b_vals.append(rhs - const)
            m = len(A_rows)

            # residual = b - A * x0
            residual = [
                b_vals[k] - sum(A_rows[k][j] * x0_vals[j] for j in range(n))
                for k in range(m)
            ]
            if max(abs(r) for r in residual) < 1e-10:
                return x0_vals  # already feasible

            # Build A*A^T  (m × m)
            AAt = [
                [sum(A_rows[i][j] * A_rows[k][j] for j in range(n)) for k in range(m)]
                for i in range(m)
            ]
            delta = self._solve_system(AAt, residual)
            if delta is None:
                return x0_vals  # singular system

            # x_proj = x0 + A^T * delta
            x_proj = list(x0_vals)
            for j in range(n):
                for i in range(m):
                    x_proj[j] += A_rows[i][j] * delta[i]
            return x_proj
        except Exception:
            return x0_vals

    @staticmethod
    def _solve_system(A_mat, b_vec):
        """
        Solve A*x = b by Gaussian elimination with partial pivoting.
        Returns solution list or None if the system is (near-)singular.
        """
        m = len(b_vec)
        # Build augmented matrix
        aug = [A_mat[i][:] + [b_vec[i]] for i in range(m)]
        for col in range(m):
            # Partial pivot
            max_row = max(range(col, m), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if abs(aug[col][col]) < 1e-14:
                return None
            # Eliminate column
            pivot = aug[col][col]
            for row in range(m):
                if row == col:
                    continue
                factor = aug[row][col] / pivot
                for j in range(col, m + 1):
                    aug[row][j] -= factor * aug[col][j]
        return [aug[i][m] / aug[i][i] for i in range(m)]

    @staticmethod
    def _extract_linear(body, ordered_vars):
        """
        Extract linear coefficients from a degree-0/1 Pyomo expression.

        Returns (coeffs, const) where coeffs[i] is the coefficient of
        ordered_vars[i] and const is the constant term.
        """
        var_id_to_idx = {id(v): i for i, v in enumerate(ordered_vars)}
        repn = generate_standard_repn(body)
        coeffs = [0.0] * len(ordered_vars)
        if repn.linear_vars:
            for coef, var in zip(repn.linear_coefs, repn.linear_vars):
                idx = var_id_to_idx.get(id(var))
                if idx is not None:
                    coeffs[idx] = float(coef)
        const = float(repn.constant) if repn.constant else 0.0
        return coeffs, const
