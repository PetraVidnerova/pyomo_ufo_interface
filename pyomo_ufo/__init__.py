"""
pyomo_ufo
---------
Pyomo solver interface for the UFO universal function optimizer.

Importing this package registers the 'ufo' solver with Pyomo's SolverFactory::

    import pyomo.environ as pyo
    import pyomo_ufo

    opt = pyo.SolverFactory('ufo')
    results = opt.solve(model, tee=True)
"""

import pyomo.environ  # noqa: F401 — triggers Pyomo plugin infrastructure

from .solver import UFOSolver  # noqa: F401 — registers 'ufo' with SolverFactory

__all__ = ['UFOSolver']
__version__ = '0.1.0'
