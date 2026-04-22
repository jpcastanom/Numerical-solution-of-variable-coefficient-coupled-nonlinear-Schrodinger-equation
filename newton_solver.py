"""
VCNLS Newton Solver Module

Implements Newton's method for solving the nonlinear system Q(W) = 0
as described in Section 0.3 of the document.

Optimisations vs. naive implementation:
  - D(V) and the nonlinear Jacobian are computed ONCE per Newton iteration
    and reused for both Q and J (previously computed twice each).
  - The Jacobian is assembled directly as a CPU scipy sparse matrix,
    avoiding the expensive GPU dense toarray() + asnumpy() transfer.
  - build_D uses vectorised numpy on CPU (fast for N~300 problem sizes).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class NewtonSolver:
    """
    Newton's method solver for the nonlinear system arising from
    Crank-Nicolson discretization.

    The nonlinear system (Equation 18) is:
    Q(W) = M3(W - V^n) - k[a^{n+1/2}M1 + g^{n+1/2}M2 - d^{n+1/2}M3
           - h^{n+1/2}M3 D((W+V^n)/2)]((W+V^n)/2) = 0

    Newton iteration (Equation 19):
    W^{(l+1)} = W^{(l)} - J^{-1}(W^{(l)}) Q(W^{(l)})

    Convergence criterion:
    ||W^{(l+1)} - W^{(l)}||_inf < epsilon
    """

    def __init__(self, matrices, config):
        self.mat = matrices
        self.cfg = config
        self.k   = config.k

        # Pre-extract CPU scipy versions of the constant matrices so we
        # never pay the GPU->CPU conversion cost inside the hot loop.
        self._M3_cpu = self._to_cpu_sparse(matrices.M3)
        self._M1_cpu = self._to_cpu_sparse(matrices.M1)
        self._M2_cpu = self._to_cpu_sparse(matrices.M2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_cpu_sparse(mat):
        """Convert a matrix (GPU cupyx or CPU scipy) to CPU scipy csr_matrix."""
        try:
            # cupyx sparse matrix
            return sp.csr_matrix(mat.get())
        except AttributeError:
            return sp.csr_matrix(mat)

    def _build_D_cpu(self, V: np.ndarray) -> sp.csr_matrix:
        """
        Build block-diagonal nonlinear matrix D(V) entirely on CPU with
        vectorised numpy — fastest path for N~300.
        """
        n   = self.cfg.n_nodes
        V_r = V.reshape(n, 4)
        z   = np.einsum('ij,ij->i', V_r, V_r)   # sum of squares per node

        size = n * 4
        base = np.arange(n) * 4
        rows = np.concatenate([base,     base + 1, base + 2, base + 3])
        cols = np.concatenate([base + 1, base,     base + 3, base + 2])
        data = np.concatenate([-z, z, -z, z])
        return sp.csr_matrix((data, (rows, cols)), shape=(size, size))

    def _build_J_nl_cpu(self, V_avg: np.ndarray) -> sp.csr_matrix:
        """Vectorised nonlinear Jacobian on CPU numpy."""
        n  = self.cfg.n_nodes
        U  = V_avg.reshape(n, 4)
        u1, u2, u3, u4 = U[:,0], U[:,1], U[:,2], U[:,3]
        z  = u1**2 + u2**2 + u3**2 + u4**2

        r_off = np.array([0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3])
        c_off = np.array([0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3])
        vals  = np.stack([
            -2*u1*u2,     -(2*u2**2+z), -2*u3*u2,     -2*u4*u2,
             2*u1**2+z,    2*u1*u2,      2*u1*u3,      2*u1*u4,
            -2*u1*u4,     -2*u2*u4,     -2*u3*u4,     -(2*u4**2+z),
             2*u1*u3,      2*u2*u3,      2*u3**2+z,    2*u3*u4,
        ], axis=1)   # (n, 16)

        base = np.arange(n) * 4
        rows = (base[:, None] + r_off[None, :]).ravel()
        cols = (base[:, None] + c_off[None, :]).ravel()
        data = vals.ravel()
        size = n * 4
        return sp.csr_matrix((data, (rows, cols)), shape=(size, size))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, V_prev: np.ndarray, t: float):
        """
        Solve for V^{n+1} using Newton's method.

        Key optimisation: D(V_avg) and J_nl are computed ONCE per
        iteration and shared between Q and J assembly.

        Args:
            V_prev: Previous state vector V^n (CPU numpy array)
            t:      Current time t_n

        Returns:
            (V_new, n_iter)
        """
        t_half = t + self.k / 2.0
        a_half = self.cfg.a(t_half)
        d_half = self.cfg.d(t_half)
        g_half = self.cfg.g(t_half)
        h_half = self.cfg.h_coeff(t_half)

        # Pre-compute the linear part of the system matrix (constant
        # within this time step — coefficients don't change mid-step).
        # S_lin = a*M1 + 0.5*g*M2 - d*M3
        S_lin = (a_half * self._M1_cpu
                 + 0.5 * g_half * self._M2_cpu
                 - d_half * self._M3_cpu)

        W = V_prev.copy()

        for iteration in range(self.cfg.max_iter):
            V_avg = 0.5 * (W + V_prev)

            # --- Build D and J_nl ONCE, reuse for both Q and J ---
            D_avg = self._build_D_cpu(V_avg)
            J_nl  = self._build_J_nl_cpu(V_avg)

            # S = S_lin - h*M3*D
            M3_D  = self._M3_cpu @ D_avg
            S     = S_lin - h_half * M3_D

            # --- Residual Q ---
            Q_val = (self._M3_cpu @ (W - V_prev)
                     - self.k * (S @ V_avg))

            if np.linalg.norm(Q_val, np.inf) < self.cfg.epsilon:
                return W, iteration + 1

            # --- Jacobian J = M3 - (k/2)*S - (k/2)*h*M3*J_nl ---
            J = (self._M3_cpu
                 - (self.k / 2.0) * S
                 - (self.k / 2.0) * h_half * (self._M3_cpu @ J_nl))

            try:
                delta = spsolve(J, Q_val)
            except Exception as e:
                print(f"Sparse solver failed: {e}, using dense solver")
                delta = np.linalg.solve(J.toarray(), Q_val)

            W_new = W - delta

            if np.linalg.norm(W_new - W, np.inf) < self.cfg.epsilon:
                return W_new, iteration + 1

            W = W_new

        if self.cfg.verbose:
            print(f"Warning: Newton did not converge in {self.cfg.max_iter} iterations")

        return W, self.cfg.max_iter
