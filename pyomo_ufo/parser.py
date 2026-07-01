"""
parser.py
---------
Parse a UFO P.OUT output file and return a structured Python dict.

UFO P.OUT structure::

    CLASS = VM - LG1   UPDATE = B   MODEL = FF   HESF = D     NF =      2
        NIT=    3 NFV=    9 NFG=    9  F=  1.000000000     G=0.000D+00
        ...
       0 NIT=   13 NFV=   15 NFG=   15  GRAD TOL  F= -1.000000000     G=0.986D-07
     FF = -0.1000000000D+01
     X  =  0.1000000000D+01  0.2000000000D+01
     TIME= 0:00:00.00

Usage::

    from pyomo_ufo.parser import UFOOutputParser
    result = UFOOutputParser().parse(text)
"""

import re


def _parse_fortran_float(s):
    """Convert a Fortran D-exponent float string to Python float."""
    return float(s.replace('D', 'E').replace('d', 'e'))


# Map UFO termination messages to Pyomo TerminationCondition names
_TERMINATION_MAP = {
    'GRAD TOL':  'optimal',
    'FV BOUND':  'optimal',
    'FV   TOL':  'optimal',
    'RMAX=0':    'optimal',   # zero step size = already at optimum
    'OPTIMUM':   'optimal',   # LP-class termination
    'ITER MAX':  'maxIterations',
    'TIME MAX':  'maxTimeLimit',
}


class UFOOutputParser:
    """Parse the text content of a UFO P.OUT file."""

    def parse(self, text):
        """
        Parse UFO output text and return a result dict.

        Parameters
        ----------
        text : str
            Full contents of P.OUT.

        Returns
        -------
        dict with keys:
            status              : 'ok' | 'warning' | 'error'
            termination_condition : 'optimal' | 'maxIterations' | 'maxTimeLimit' | 'other'
            termination_message : str (raw UFO message)
            objective           : float or None
            variables           : list of float (may be empty for large NF)
            iterations          : int
            n_feval             : int
            n_geval             : int
            wall_time           : str (e.g. '0:00:00.12')
            return_code         : int
            header              : dict (CLASS, UPDATE, MODEL, HESF, NF)
        """
        lines = text.splitlines()

        result = {
            'status': 'error',
            'termination_condition': 'other',
            'termination_message': '',
            'objective': None,
            'variables': [],
            'iterations': 0,
            'n_feval': 0,
            'n_geval': 0,
            'wall_time': '',
            'return_code': -1,
            'header': {},
        }

        self._parse_header(lines, result)
        self._parse_iterations(lines, result)
        self._parse_termination(lines, result)
        self._parse_results(lines, result)

        # Recognize UFO internal errors (e.g. singular Jacobian) that leave
        # the termination line with a NaN F= value and no mapped message.
        # These appear as UXSGLE / UXSGFD / similar diagnostics.
        for line in lines:
            s = line.strip()
            if s.startswith('UXS') and (':' in s):
                result['status'] = 'error'
                result['termination_condition'] = 'other'
                result['termination_message'] = s
                result['objective'] = None
                break

        return result

    # ------------------------------------------------------------------

    def _parse_header(self, lines, result):
        """Parse the first header line: CLASS, UPDATE, MODEL, HESF, NF."""
        if not lines:
            return
        # e.g. " CLASS = VM - LG1   UPDATE = B   MODEL = FF   HESF = D     NF =      5"
        m = re.search(
            r'CLASS\s*=\s*(\S+)\s*-\s*(\S+)\s+'
            r'UPDATE\s*=\s*(\S+)\s+'
            r'MODEL\s*=\s*(\S+)\s+'
            r'HESF\s*=\s*(\S+)\s+'
            r'NF\s*=\s*(\d+)',
            lines[0]
        )
        if m:
            result['header'] = {
                'class':  m.group(1),
                'method': m.group(2),
                'update': m.group(3),
                'model':  m.group(4),
                'hesf':   m.group(5),
                'nf':     int(m.group(6)),
            }

    def _parse_iterations(self, lines, result):
        """Scan iteration lines to extract last NIT, NFV, NFG values."""
        # Matches both unconstrained and constrained iteration lines:
        # "    NIT=    3 NFV=    9 NFG=    9  F= ..."
        # "    NIC=  0 NIT=   6 NFV=   81 NFG=    0 F=..."
        nit_re = re.compile(r'NIT\s*=\s*(\d+)')
        nfv_re = re.compile(r'NFV\s*=\s*(\d+)')
        nfg_re = re.compile(r'NFG\s*=\s*(\d+)')

        for line in lines:
            if 'NIT=' in line and 'NFV=' in line:
                m = nit_re.search(line)
                if m:
                    result['iterations'] = int(m.group(1))
                m = nfv_re.search(line)
                if m:
                    result['n_feval'] = int(m.group(1))
                m = nfg_re.search(line)
                if m:
                    result['n_geval'] = int(m.group(1))

    def _parse_termination(self, lines, result):
        """Find the termination line (leading 0) and set status/condition."""
        # Termination line pattern: leading integer (return code), then NIT=...
        # e.g. "   0 NIT=   13 NFV=   15 NFG=   15  GRAD TOL  F= ..."
        term_re = re.compile(r'^\s*(\d+)\s+NIT\s*=')

        for line in lines:
            m = term_re.match(line)
            if m:
                rc = int(m.group(1))
                result['return_code'] = rc
                result['status'] = 'ok' if rc == 0 else 'warning'

                # Identify termination message
                for msg, cond in _TERMINATION_MAP.items():
                    if msg in line:
                        result['termination_condition'] = cond
                        result['termination_message'] = msg
                        break
                else:
                    result['termination_condition'] = 'other'
                    # Try to extract message between NFG=... and F=
                    m2 = re.search(r'NFG\s*=\s*\d+\s+(.*?)\s+F\s*=', line)
                    if m2:
                        result['termination_message'] = m2.group(1).strip()
                    else:
                        result['termination_message'] = line.strip()
                break

    def _parse_results(self, lines, result):
        """Parse the objective value, X vector, and wall time from summary lines."""
        # Objective value: "FF = value" or "F  = value"
        obj_re = re.compile(r'^\s*(FF|F)\s*=\s*([0-9\.\-\+DEde]+)')
        # X values start line
        x_start_re = re.compile(r'^\s*X\s*=\s*(.*)')
        # Continuation line for X (6 spaces indent, no '=' )
        x_cont_re = re.compile(r'^\s{5,}([0-9\.\-\+DEde\s]+)$')
        # Time line
        time_re = re.compile(r'TIME\s*=\s*(\S+)')

        collecting_x = False
        x_tokens = []

        for line in lines:
            # Objective
            m = obj_re.match(line)
            if m and not collecting_x:
                try:
                    result['objective'] = _parse_fortran_float(m.group(2))
                except ValueError:
                    pass
                continue

            # Start of X values
            m = x_start_re.match(line)
            if m:
                collecting_x = True
                x_tokens += m.group(1).split()
                continue

            # Continuation of X values
            if collecting_x:
                if time_re.search(line):
                    collecting_x = False
                    # Parse time on same line
                    mt = time_re.search(line)
                    if mt:
                        result['wall_time'] = mt.group(1)
                    break
                # Check if this looks like a continuation line (all numeric tokens)
                stripped = line.strip()
                if stripped and re.match(r'^[0-9\.\-\+DEde\s]+$', stripped):
                    x_tokens += stripped.split()
                else:
                    collecting_x = False

            # Time line (when not after X)
            m = time_re.search(line)
            if m and not collecting_x:
                result['wall_time'] = m.group(1)

        # Convert X tokens to floats
        if x_tokens:
            try:
                result['variables'] = [_parse_fortran_float(t) for t in x_tokens]
            except ValueError as e:
                result['variables'] = []
