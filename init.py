"""
VCNLS Solver Package

Variable Coefficient Nonlinear Schrödinger System Solver

Implementation of the Galerkin finite element method with Crank-Nicolson 
time integration for solving coupled nonlinear Schrödinger equations 
with variable coefficients.

Based on the theoretical framework from:
"Variable coefficient coupled nonlinear Schrödinger equation" - 
Numerical method using finite elements and Newton's method.
"""
# vcnls_solver/__init__.py
try:
    from .config import VCNLSConfig
    from .matrices import VCNLSMatrices
    from .newton_solver import NewtonSolver
    from .time_stepping import VCNLSIntegrator
    from .io_utils import VCNLSVisualizer
except ImportError:
    # Fallback para cuando se importa directamente
    from config import VCNLSConfig
    from matrices import VCNLSMatrices
    from newton_solver import NewtonSolver
    from time_stepping import VCNLSIntegrator
    from io_utils import VCNLSVisualizer

__all__ = ['VCNLSConfig', 'VCNLSMatrices', 'NewtonSolver', 
           'VCNLSIntegrator', 'VCNLSVisualizer']

from .config import VCNLSConfig
from .matrices import VCNLSMatrices
from .newton_solver import NewtonSolver
from .time_stepping import VCNLSIntegrator
from .io_utils import VCNLSVisualizer


__author__ = "Juan Pablo"
__all__ = [
    "VCNLSConfig",
    "VCNLSMatrices",
    "NewtonSolver",
    "VCNLSIntegrator",
    "VCNLSVisualizer"
]