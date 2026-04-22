"""
VCNLS Convergence Analysis Module

Computes spatial and temporal convergence tables (L-infinity error vs step size)
for the coupled VCNLS system.

The exact solution is supplied by the caller as a callable:
    psi_exact(x, t) -> (psi, phi)  complex arrays of shape (n_nodes,)

Each function accepts a base VCNLSConfig whose parameters are reused across
all runs; only the swept variable (k or N) is overridden per run.

Each function returns a pandas DataFrame with columns:
    - Temporal : Delta_t, L_inf_Error, Order
    - Spatial  : Delta_x, L_inf_Error, Order

Usage:
    from convergence import temporal_convergence, spatial_convergence
    from config import VCNLSConfig

    def my_exact(x, t):
        ...
        return psi, phi

    # temporal: cfg.N must be large enough to make spatial error negligible
    base_cfg = VCNLSConfig(xL=-20, xR=60, N=80000, T=4.0, k=0.1, ...)
    df_time  = temporal_convergence(base_cfg, my_exact, k_list=[0.2, 0.1, 0.05, 0.025])

    # spatial: cfg.k must be small enough to make temporal error negligible
    base_cfg = VCNLSConfig(xL=-20, xR=60, N=400, T=4.0, k=0.001, ...)
    df_space = spatial_convergence(base_cfg, my_exact, N_list=[200, 400, 800, 1600])
"""

import os
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

if os.environ.get('TQDM_NOTEBOOK'):
    from tqdm.notebook import tqdm
else:
    from tqdm.auto import tqdm

from config import VCNLSConfig
from time_stepping import VCNLSIntegrator
from io_utils import VCNLSVisualizer


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def _run_single(
    cfg:       VCNLSConfig,
    psi_exact: Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]],
    use_gpu:   bool,
) -> float:
    """
    Integrate one configuration and return the L-infinity error at cfg.T.

    Args:
        cfg:       Fully configured VCNLSConfig instance.
        psi_exact: Callable (x, t) -> (psi, phi) returning the exact solution.
        use_gpu:   Whether to use GPU acceleration (passed to VCNLSIntegrator).

    Returns:
        L-infinity error (max over both components at final time cfg.T).
    """
    x = np.linspace(cfg.xL, cfg.xR, cfg.n_nodes)

    # Evaluate exact solution at t=0 on the current grid.
    # psi_exact must return either a tuple (psi, phi) or a single array
    # (interpreted as psi=phi, symmetric soliton case).
    _raw0 = psi_exact(x, 0.0)
    if isinstance(_raw0, tuple):
        psi0_c = np.asarray(_raw0[0], dtype=complex)
        phi0_c = np.asarray(_raw0[1], dtype=complex)
    else:
        psi0_c = np.asarray(_raw0, dtype=complex)
        phi0_c = psi0_c.copy()

    _rawT = psi_exact(x, cfg.T)
    if isinstance(_rawT, tuple):
        psi_ex = np.asarray(_rawT[0], dtype=complex)
        phi_ex = np.asarray(_rawT[1], dtype=complex)
    else:
        psi_ex = np.asarray(_rawT, dtype=complex)
        phi_ex = psi_ex.copy()

    # set_initial_condition calls psi0(xm) pointwise; we return the
    # precomputed value at the matching index using a lookup dict.
    psi0_map = {float(x[i]): psi0_c[i] for i in range(len(x))}
    phi0_map = {float(x[i]): phi0_c[i] for i in range(len(x))}

    def psi0(xp: float) -> complex:
        return psi0_map[float(xp)]

    def phi0(xp: float) -> complex:
        return phi0_map[float(xp)]

    integrator = VCNLSIntegrator(cfg, use_gpu=use_gpu)
    V0     = integrator.set_initial_condition(psi0, phi0)
    result = integrator.integrate(V0)

    final_state      = result['states'][-1:]   # shape (1, 4*n_nodes)
    psi_num, phi_num = VCNLSVisualizer.reconstruct_wavefunctions(
        final_state, cfg.n_nodes
    )

    return max(np.max(np.abs(psi_ex - psi_num[0])),
               np.max(np.abs(phi_ex - phi_num[0])))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def temporal_convergence(
    cfg:       VCNLSConfig,
    psi_exact: Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]],
    k_list:    Optional[List[float]] = None,
    use_gpu:   bool        = True,
) -> pd.DataFrame:
    """
    Compute temporal convergence table (L-infinity error vs Delta_t).

    Fixes the spatial grid from cfg (cfg.N should be large enough so that
    spatial error is negligible) and sweeps over k_list, overriding cfg.k
    for each run.

    Args:
        cfg:       Base VCNLSConfig. cfg.N and all physical parameters are
                   kept fixed across runs; only k is overridden.
        psi_exact: Callable (x, t) -> (psi, phi) with the exact solution.
        k_list:    Time steps to sweep. Default: [0.2, 0.1, 0.05, 0.025]
        use_gpu:   Use GPU acceleration (same flag as VCNLSIntegrator).

    Returns:
        DataFrame with columns [Delta_t, L_inf_Error, Order].
    """
    if k_list is None:
        k_list = [0.2, 0.1, 0.05, 0.025]

    print("=" * 45)
    print("TEMPORAL CONVERGENCE ANALYSIS")
    print("=" * 45)
    print(f"  Fixed spatial intervals : N = {cfg.N}  (h = {cfg.h:.5f})")
    print(f"  Time steps tested       : {k_list}")
    print(f"  Final time              : T = {cfg.T}")
    print(f"  Newton tolerance        : {cfg.epsilon:.0e}")
    print(f"  GPU acceleration        : {use_gpu}")
    print()

    errs = []
    bar  = tqdm(k_list, desc="Temporal convergence", unit="run")

    for k in bar:
        bar.set_postfix(k=f"{k:.4f}")

        run_cfg = replace(
            cfg,
            k          = k,
            save_every = max(1, round(cfg.T / k)),  # keep only final state
            verbose    = False,
        )

        err = _run_single(run_cfg, psi_exact, use_gpu)
        errs.append(err)
        bar.set_postfix(k=f"{k:.4f}", L_inf=f"{err:.3e}")

    ks    = np.array(k_list, dtype=float)
    errs  = np.array(errs)
    order = np.full(len(ks), np.nan)
    order[1:] = np.diff(np.log(errs)) / np.diff(np.log(ks))

    df = pd.DataFrame({
        'Delta_t':     ks,
        'L_inf_Error': errs,
        'Order':       order,
    })

    print()
    print("=" * 45)
    print("TEMPORAL CONVERGENCE TABLE")
    print("=" * 45)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()
    return df


def spatial_convergence(
    cfg:       VCNLSConfig,
    psi_exact: Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]],
    N_list:    Optional[List[int]] = None,
    use_gpu:   bool      = True,
) -> pd.DataFrame:
    """
    Compute spatial convergence table (L-infinity error vs Delta_x).

    Fixes the time step from cfg (cfg.k should be small enough so that
    temporal error is negligible) and sweeps over N_list, overriding cfg.N
    for each run.

    Args:
        cfg:       Base VCNLSConfig. cfg.k and all physical parameters are
                   kept fixed across runs; only N is overridden.
        psi_exact: Callable (x, t) -> (psi, phi) with the exact solution.
        N_list:    Spatial interval counts to sweep.
                   Default: [200, 400, 800, 1600]
        use_gpu:   Use GPU acceleration (same flag as VCNLSIntegrator).

    Returns:
        DataFrame with columns [Delta_x, L_inf_Error, Order].
    """
    if N_list is None:
        N_list = [200, 400, 800, 1600]

    domain_w = cfg.xR - cfg.xL
    h_list   = [domain_w / N for N in N_list]

    print("=" * 45)
    print("SPATIAL CONVERGENCE ANALYSIS")
    print("=" * 45)
    print(f"  Fixed time step         : k = {cfg.k}")
    print(f"  Spatial intervals tested: {N_list}")
    print(f"  Corresponding h values  : {[f'{h:.5f}' for h in h_list]}")
    print(f"  Final time              : T = {cfg.T}")
    print(f"  Newton tolerance        : {cfg.epsilon:.0e}")
    print(f"  GPU acceleration        : {use_gpu}")
    print()

    errs = []
    bar  = tqdm(
        zip(N_list, h_list),
        total=len(N_list),
        desc="Spatial convergence",
        unit="run",
    )

    for N, h_val in bar:
        bar.set_postfix(N=N, h=f"{h_val:.5f}")

        run_cfg = replace(
            cfg,
            N          = N,
            save_every = max(1, round(cfg.T / cfg.k)),
            verbose    = False,
        )

        err = _run_single(run_cfg, psi_exact, use_gpu)
        errs.append(err)
        bar.set_postfix(N=N, h=f"{h_val:.5f}", L_inf=f"{err:.3e}")

    hs    = np.array(h_list)
    errs  = np.array(errs)
    order = np.full(len(hs), np.nan)
    order[1:] = np.diff(np.log(errs)) / np.diff(np.log(hs))

    df = pd.DataFrame({
        'Delta_x':     hs,
        'L_inf_Error': errs,
        'Order':       order,
    })

    print()
    print("=" * 45)
    print("SPATIAL CONVERGENCE TABLE")
    print("=" * 45)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()
    return df
