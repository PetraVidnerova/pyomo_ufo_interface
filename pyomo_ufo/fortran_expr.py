"""
fortran_expr.py
---------------
Convert a Pyomo expression tree into a Fortran double-precision expression string.

Usage::

    from pyomo_ufo.fortran_expr import to_fortran

    var_map = {id(model.x[1]): 'X(1)', id(model.x[2]): 'X(2)'}
    s = to_fortran(model.obj.expr, var_map)
    # e.g. '((X(1)-1.0D0)**2 + 1.0D2*(X(2)-X(1)**2)**2)'
"""

from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import (
    SumExpression,
    LinearExpression,
    MonomialTermExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    NegationExpression,
    UnaryFunctionExpression,
    AbsExpression,
    MaxExpression,
    MinExpression,
)
from pyomo.core.expr.numvalue import NumericConstant, value as pyo_value
from pyomo.core.expr.relational_expr import InequalityExpression, EqualityExpression

# Support both old (pre-6.7.2) and new naming
try:
    from pyomo.core.base.var import VarData
except ImportError:
    from pyomo.core.base.var import _GeneralVarData as VarData
try:
    from pyomo.core.base.param import ParamData
except ImportError:
    from pyomo.core.base.param import _ParamData as ParamData

# Mapping from Pyomo unary function names to Fortran intrinsics
_UNARY_FORTRAN = {
    'exp':   'EXP',
    'log':   'LOG',
    'log10': 'LOG10',
    'sin':   'SIN',
    'cos':   'COS',
    'tan':   'TAN',
    'asin':  'ASIN',
    'acos':  'ACOS',
    'atan':  'ATAN',
    'sinh':  'SINH',
    'cosh':  'COSH',
    'tanh':  'TANH',
    'sqrt':  'SQRT',
    'abs':   'ABS',
    'ceil':  'CEILING',
    'floor': 'FLOOR',
}


def _fortran_const(v, integer_ok=False):
    """Format a numeric value as a Fortran double-precision literal.

    If ``integer_ok`` is True and the value is a small integer, emit it as a
    plain integer literal (useful for exponents: ``**2`` rather than ``**2.0D0``).
    """
    v_float = float(v)
    if integer_ok and v_float == int(v_float) and abs(v_float) < 1e9:
        return str(int(v_float))
    if v_float == int(v_float) and abs(v_float) < 1e15:
        iv = int(v_float)
        if iv == 0:
            return '0.0D0'
        # Use scientific notation for large integers to keep it readable
        if abs(iv) >= 10000:
            s = f'{v_float:.15E}'
            mantissa, exp = s.split('E')
            exp_int = int(exp)
            # Trim trailing zeros from mantissa
            mantissa = mantissa.rstrip('0').rstrip('.')
            if '.' not in mantissa:
                mantissa += '.0'
            return f'{mantissa}D{exp_int:+03d}'
        return f'{float(iv):.1f}D0'
    else:
        s = f'{v_float:.15E}'
        mantissa, exp = s.split('E')
        exp_int = int(exp)
        mantissa = mantissa.rstrip('0').rstrip('.')
        if '.' not in mantissa:
            mantissa += '.0'
        if exp_int == 0:
            return f'{mantissa}D0'
        return f'{mantissa}D{exp_int:+03d}'


class _FortranVisitor(StreamBasedExpressionVisitor):
    """
    Walk a Pyomo expression tree and return a Fortran expression string.

    Parameters
    ----------
    var_map : dict
        Maps ``id(var_data)`` → Fortran variable name string, e.g. ``'X(1)'``.
    """

    def __init__(self, var_map):
        super().__init__()
        self._var_map = var_map

    # ------------------------------------------------------------------
    # StreamBasedExpressionVisitor callbacks
    # ------------------------------------------------------------------

    def initializeWalker(self, expr):
        # Always walk the full tree
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, result
        return True, None

    def beforeChild(self, node, child, child_idx):
        """Detect leaves: variables, parameters, and numeric constants."""
        # Pyomo VarData
        if isinstance(child, VarData):
            name = self._var_map.get(id(child))
            if name is None:
                raise KeyError(
                    f"Variable {child.name} not found in var_map. "
                    "Ensure all variables are registered."
                )
            return False, name

        # Fixed variable or parameter — evaluate numerically
        if isinstance(child, ParamData):
            return False, _fortran_const(pyo_value(child))

        # Numeric constant
        if isinstance(child, NumericConstant):
            return False, _fortran_const(child.value)

        # Plain Python int/float
        if isinstance(child, (int, float)):
            return False, _fortran_const(child)

        # Expression node — descend
        return True, None

    def exitNode(self, node, data):
        """Combine children results into a Fortran string for this node."""

        if isinstance(node, NegationExpression):
            return f'(-{data[0]})'

        if isinstance(node, AbsExpression):
            return f'ABS({data[0]})'

        if isinstance(node, SumExpression):
            # data may contain '+' or '-' terms; just join with ' + '
            # but NegationExpression children already have their sign
            return '(' + ' + '.join(data) + ')'

        if isinstance(node, LinearExpression):
            # LinearExpression: constant + sum of coef*var
            # children are: [constant_or_zero, coef1, var1, coef2, var2, ...]
            # Actually LinearExpression stores .constant, .linear_coefs, .linear_vars
            terms = []
            if node.constant != 0:
                terms.append(_fortran_const(node.constant))
            for coef, var in zip(node.linear_coefs, node.linear_vars):
                vname = self._var_map.get(id(var))
                if vname is None:
                    raise KeyError(f"Variable {var.name} not found in var_map.")
                c = float(coef)
                if c == 1.0:
                    terms.append(vname)
                elif c == -1.0:
                    terms.append(f'(-{vname})')
                else:
                    terms.append(f'{_fortran_const(c)}*{vname}')
            if not terms:
                return '0.0D0'
            return '(' + ' + '.join(terms) + ')'

        if isinstance(node, MonomialTermExpression):
            # coef * var
            coef_str, var_str = data
            # coef_str is already a Fortran string (constant or expression)
            # If coef is 1, omit
            if coef_str == '1.0D0':
                return var_str
            if coef_str == '-1.0D0':
                return f'(-{var_str})'
            return f'({coef_str}*{var_str})'

        if isinstance(node, ProductExpression):
            return f'({data[0]}*{data[1]})'

        if isinstance(node, DivisionExpression):
            return f'({data[0]}/{data[1]})'

        if isinstance(node, PowExpression):
            base, exp = data
            # If exponent is a plain numeric constant, emit as integer if possible
            exp_node = node.args[1]
            if isinstance(exp_node, (int, float, NumericConstant)):
                ev = float(pyo_value(exp_node))
                exp = _fortran_const(ev, integer_ok=True)
            return f'({base}**{exp})'

        if isinstance(node, UnaryFunctionExpression):
            fname = node.getname()
            fortran_fn = _UNARY_FORTRAN.get(fname)
            if fortran_fn is None:
                raise NotImplementedError(
                    f"Unary function '{fname}' has no Fortran mapping. "
                    f"Supported: {list(_UNARY_FORTRAN)}"
                )
            return f'{fortran_fn}({data[0]})'

        if isinstance(node, MaxExpression):
            # Reduce: MAX(a, b, c) → MAX(a, MAX(b, c))
            result = data[-1]
            for d in reversed(data[:-1]):
                result = f'MAX({d}, {result})'
            return result

        if isinstance(node, MinExpression):
            result = data[-1]
            for d in reversed(data[:-1]):
                result = f'MIN({d}, {result})'
            return result

        # Fallback: try to evaluate numerically (e.g. fixed expressions)
        try:
            return _fortran_const(pyo_value(node))
        except Exception:
            raise NotImplementedError(
                f"Unsupported expression node type: {type(node).__name__}"
            )

    def finalizeResult(self, result):
        return result


def to_fortran(expr, var_map):
    """
    Convert a Pyomo expression to a Fortran double-precision expression string.

    Parameters
    ----------
    expr : Pyomo expression
    var_map : dict
        ``{id(var_data): 'X(i)'}`` for all variables appearing in ``expr``.

    Returns
    -------
    str
        A Fortran expression string suitable for inline use in generated code.
    """
    # Handle plain numeric values
    if isinstance(expr, (int, float)):
        return _fortran_const(expr)
    if isinstance(expr, NumericConstant):
        return _fortran_const(expr.value)

    visitor = _FortranVisitor(var_map)
    return visitor.walk_expression(expr)
