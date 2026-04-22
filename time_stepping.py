"""
VCNLS Time Stepping Module

Implements the Crank-Nicolson time integration scheme as described
in Section 0.2 (Full discretization) of the document.
"""

import os
import numpy as np
from matrices import VCNLSMatrices
from newton_solver import NewtonSolver

if os.environ.get('TQDM_NOTEBOOK'):
    from tqdm.notebook import tqdm
else:
    from tqdm.auto import tqdm


class VCNLSIntegrator:
    """
    Time integrator for the VCNLS system using Crank-Nicolson scheme.
    """

    def __init__(self, config, use_gpu: bool = True):
        self.cfg      = config
        self.matrices = VCNLSMatrices(config, use_gpu=use_gpu)
        self.newton   = NewtonSolver(self.matrices, config)
        self.times_saved  = []
        self.states_saved = []

    def set_initial_condition(self, psi0, phi0):
        x  = np.linspace(self.cfg.xL, self.cfg.xR, self.cfg.n_nodes)
        V0 = np.zeros(4 * self.cfg.n_nodes)
        for m, xm in enumerate(x):
            psi_val = psi0(xm)
            phi_val = phi0(xm)
            V0[4*m]   = psi_val.real
            V0[4*m+1] = psi_val.imag
            V0[4*m+2] = phi_val.real
            V0[4*m+3] = phi_val.imag
        return V0

    def integrate(self, V0):
        V_current = V0.copy()
        t = 0.0

        self.times_saved  = [0.0]
        self.states_saved = [V0.copy()]

        n_steps    = self.cfg.n_steps
        save_every = self.cfg.save_every

        bar = tqdm(
            total=n_steps,
            desc="Integrating",
            unit="step",
            disable=not self.cfg.verbose,
        )

        for step in range(n_steps):
            V_new, n_iter = self.newton.solve(V_current, t)
            V_current = V_new
            t += self.cfg.k

            if (step + 1) % save_every == 0:
                self.times_saved.append(t)
                self.states_saved.append(V_current.copy())

            bar.update(1)
            bar.set_postfix(t=f"{t:.3f}", iters=n_iter)

        bar.close()

        return {
            'times':  np.array(self.times_saved),
            'states': np.array(self.states_saved),
            'x_grid': np.linspace(self.cfg.xL, self.cfg.xR, self.cfg.n_nodes),
        }
