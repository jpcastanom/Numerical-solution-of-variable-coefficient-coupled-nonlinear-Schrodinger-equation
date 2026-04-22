"""
VCNLS Configuration Module

Defines the configuration class for the Variable Coefficient Nonlinear 
Schrödinger (VCNLS) solver using dataclasses.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class VCNLSConfig:
    """
    Configuration class for the VCNLS solver.
    
    This class contains all parameters needed for the numerical solution
    of the coupled nonlinear Schrödinger system with variable coefficients.
    
    The system solved is (from Section 0.1 of the document):
        iψ_t = -a(t)ψ_xx - i d(t)ψ + i g(t)ψ_x + h(t)(|φ|² + |ψ|²)ψ
        iφ_t = -a(t)φ_xx - i d(t)φ + i g(t)φ_x + h(t)(|φ|² + |ψ|²)φ
    
    With boundary conditions (Neumann homogeneous):
        ψ_x = φ_x = 0 at x = x_L and x = x_R
    """
    
    # Spatial domain parameters
    xL: float = -20.0           # Left boundary (x_L in document)
    xR: float = 60.0            # Right boundary (x_R in document)
    N: int = 400                # Number of spatial intervals (N in document)
                                # Number of nodes = N + 1
    
    # Temporal domain parameters
    T: float = 10.0             # Final time
    k: float = 0.001            # Time step (k in document, Section 0.2)
    
    # Time-dependent coefficients (functions of time as in document)
    a: Callable[[float], float] = lambda t: 0.5       
    # Dispersion management parameter a(t) (Equation 1, 5, 6)
    
    d: Callable[[float], float] = lambda t: 0.0       
    # Gain/loss term d(t) (Equation 1, 5, 6)
    
    g: Callable[[float], float] = lambda t: 0.0       
    # Transport coefficient g(t) (Equation 5, 6)
    
    h_coeff: Callable[[float], float] = lambda t: 1.0  
    # Nonlinear management h(t) (Equation 1, 5, 6)
    # Note: named h_coeff to avoid conflict with spatial step h
    
    # Newton's method parameters (Section 0.3)
    epsilon: float = 1e-12      # Convergence tolerance ε
    max_iter: int = 50          # Maximum Newton iterations
    
    # Output options
    save_every: int = 1       # Save solution every n time steps
    verbose: bool = True        # Print progress information
    
    @property
    def h(self) -> float:
        """
        Spatial step size h = (x_R - x_L) / N.
        
        From Section 0.1.1: "the interval [x_L, x_R] is discretized by 
        uniform (N+1) grid points... the grid spacing is given by 
        h = (x_R - x_L) / N"
        """
        return (self.xR - self.xL) / self.N
    
    @property
    def n_steps(self) -> int:
        """Total number of time steps = T / k"""
        return int(self.T / self.k)
    
    @property
    def n_nodes(self) -> int:
        """Total number of spatial nodes = N + 1"""
        return self.N + 1