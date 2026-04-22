"""
VCNLS Matrix Assembly Module

Constructs the block-tridiagonal matrices M3, M1, M2 and the nonlinear 
block-diagonal matrix D(V) as defined in Section 0.1.1 and Equation (17)
of the document.
"""

import numpy as np
import scipy.sparse as sp
from typing import List

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class VCNLSMatrices:
    """
    Handles the construction of all matrices required for the VCNLS solver.

    M3: Mass matrix (symmetric, block-tridiagonal)
    M1: Stiffness matrix (related to Laplacian, block-tridiagonal)
    M2: Advection matrix (skew-symmetric, block-tridiagonal)
    D(V): Nonlinear diagonal blocks matrix
    """

    def __init__(self, config, use_gpu: bool = True):
        self.cfg = config
        self.h = config.h
        self.N = config.N
        self.n_nodes = config.n_nodes
        self.use_gpu = use_gpu and GPU_AVAILABLE

        self.A_mat = np.array([
            [0,  1,  0,  0],
            [-1, 0,  0,  0],
            [0,  0,  0,  1],
            [0,  0, -1,  0]
        ], dtype=np.float64)

        self.I4 = np.eye(4)

        # Build on CPU, then upload to GPU once
        M3_cpu = self._build_M3()
        M1_cpu = self._build_M1()
        M2_cpu = self._build_M2()

        if self.use_gpu:
            self.M3 = csp.csr_matrix(M3_cpu)
            self.M1 = csp.csr_matrix(M1_cpu)
            self.M2 = csp.csr_matrix(M2_cpu)
        else:
            self.M3 = M3_cpu
            self.M1 = M1_cpu
            self.M2 = M2_cpu

    def _build_M3(self) -> sp.csr_matrix:
        main_diag = [2 * self.I4] + [4 * self.I4] * (self.N - 1) + [2 * self.I4]
        off_diag = [self.I4] * self.N
        return (1.0 / 6.0) * self._block_tridiagonal(off_diag, main_diag, off_diag)

    def _build_M1(self) -> sp.csr_matrix:
        main_diag = [-self.A_mat] + [-2 * self.A_mat] * (self.N - 1) + [-self.A_mat]
        off_diag = [self.A_mat] * self.N
        return -(1.0 / self.h ** 2) * self._block_tridiagonal(off_diag, main_diag, off_diag)

    def _build_M2(self) -> sp.csr_matrix:
        main_diag = [np.zeros((4, 4))] * self.n_nodes
        lower_diag = [-self.I4] * self.N
        upper_diag = [self.I4] * self.N
        return (1.0 / (2.0 * self.h)) * self._block_tridiagonal(lower_diag, main_diag, upper_diag)

    def _block_tridiagonal(self, lower: List[np.ndarray],
                           main: List[np.ndarray],
                           upper: List[np.ndarray]) -> sp.csr_matrix:
        n_blocks = len(main)
        n = n_blocks * 4
        rows, cols, data = [], [], []

        for i in range(n_blocks):
            for r in range(4):
                for c in range(4):
                    rows.append(4 * i + r)
                    cols.append(4 * i + c)
                    data.append(main[i][r, c])
            if i > 0:
                for r in range(4):
                    for c in range(4):
                        rows.append(4 * i + r)
                        cols.append(4 * (i - 1) + c)
                        data.append(lower[i - 1][r, c])
            if i < n_blocks - 1:
                for r in range(4):
                    for c in range(4):
                        rows.append(4 * i + r)
                        cols.append(4 * (i + 1) + c)
                        data.append(upper[i][r, c])

        return sp.csr_matrix(sp.coo_matrix((data, (rows, cols)), shape=(n, n)))

    def build_D(self, V: np.ndarray):
        """
        Build block-diagonal nonlinear matrix D(V).
        V can be a numpy or cupy array; returns same type of sparse matrix.
        """
        if self.use_gpu:
            V_xp = cp.asarray(V).reshape(self.n_nodes, 4)
            z = cp.sum(V_xp ** 2, axis=1)
            n = self.n_nodes * 4
            base = cp.arange(self.n_nodes) * 4
            rows = cp.concatenate([base, base + 1, base + 2, base + 3])
            cols = cp.concatenate([base + 1, base, base + 3, base + 2])
            data = cp.concatenate([-z, z, -z, z])
            return csp.csr_matrix(csp.coo_matrix((data, (rows, cols)), shape=(n, n)))
        else:
            V_xp = np.asarray(V).reshape(self.n_nodes, 4)
            z = np.sum(V_xp ** 2, axis=1)
            n = self.n_nodes * 4
            base = np.arange(self.n_nodes) * 4
            rows = np.concatenate([base, base + 1, base + 2, base + 3])
            cols = np.concatenate([base + 1, base, base + 3, base + 2])
            data = np.concatenate([-z, z, -z, z])
            return sp.csr_matrix(sp.coo_matrix((data, (rows, cols)), shape=(n, n)))
