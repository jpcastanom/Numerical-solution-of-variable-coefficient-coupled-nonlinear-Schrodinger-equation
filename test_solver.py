"""
VCNLS Solver Unit Tests

Tests for matrix properties, Newton convergence, and conservation laws.
"""

import numpy as np
from config import VCNLSConfig
from matrices import VCNLSMatrices
from newton_solver import NewtonSolver
from time_stepping import VCNLSIntegrator
from io_utils import VCNLSVisualizer


class VCNLSUnitTests:
    """Unit tests for the VCNLS solver components."""
    
    @staticmethod
    def test_matrix_symmetry():
        """Verify M3 is symmetric positive definite and M2 is skew-symmetric."""
        print("\nTest: Matrix Symmetry Properties")
        
        cfg = VCNLSConfig(N=20)
        mat = VCNLSMatrices(cfg)
        
        # Test M3 symmetry: M3 = M3^T
        M3_dense = mat.M3.toarray()
        assert np.allclose(M3_dense, M3_dense.T), "M3 is not symmetric"
        
        # Test M3 positive definiteness (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(M3_dense)
        assert bool(np.all(eigenvalues > 0)), "M3 is not positive definite"
        
        # Test M2 skew-symmetry: M2 = -M2^T
        M2_dense = mat.M2.toarray()
        assert np.allclose(M2_dense, -M2_dense.T), "M2 is not skew-symmetric"
        
        print("  ✓ M3 is symmetric positive definite")
        print("  ✓ M2 is skew-symmetric")
        return True
    
    @staticmethod
    def test_newton_convergence():
        """Verify Newton's method converges in few iterations for small time step."""
        print("\nTest: Newton Convergence")
        
        cfg = VCNLSConfig(
            N=50,
            k=0.001,  # Small time step for fast convergence
            epsilon=1e-10,
            max_iter=50,
            verbose=False
        )
        
        # Simple Gaussian initial condition
        psi0 = lambda x: np.exp(-x**2)
        phi0 = lambda x: np.exp(-x**2)
        
        integrator = VCNLSIntegrator(cfg)
        V0 = integrator.set_initial_condition(psi0, phi0)
        
        # Single Newton step
        newton = NewtonSolver(integrator.matrices, cfg)
        V1, n_iter = newton.solve(V0, t=0.0)
        
        # Should converge in less than 10 iterations
        assert n_iter < 10, f"Newton took too many iterations: {n_iter}"
        print(f"  ✓ Newton converged in {n_iter} iterations")
        
        # Verify small residual: run one more solve from the converged point;
        # if already converged, Newton exits at iteration 1 with norm < epsilon.
        _, n_iter2 = newton.solve(V1, t=0.0)
        assert n_iter2 == 1, f"Residual too large: solution not converged (took {n_iter2} iters)"
        print(f"  ✓ Residual verified (re-solve converged in {n_iter2} iteration)")
        
        return True
    
    @staticmethod
    def test_mass_conservation():
        """Verify mass conservation when d(t) = 0 (no gain/loss)."""
        print("\nTest: Mass Conservation")
        
        cfg = VCNLSConfig(
            N=100,
            T=1.0,
            k=0.01,
            d=lambda t: 0.0,  # No gain/loss
            save_every=10,
            verbose=False
        )
        
        psi0 = lambda x: np.exp(-x**2 / 2)
        phi0 = lambda x: np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        
        integrator = VCNLSIntegrator(cfg)
        V0 = integrator.set_initial_condition(psi0, phi0)
        result = integrator.integrate(V0)
        
        states = result['states']
        x = result['x_grid']
        
        # Reconstruct and compute norms
        psi, phi = VCNLSVisualizer.reconstruct_wavefunctions(states, cfg.n_nodes)
        norms = np.trapezoid(np.abs(psi)**2 + np.abs(phi)**2, x, axis=1)
        norm = np.trapezoid(np.abs(psi)**2, x, axis=1)
        
        # Check conservation (should be very small variation)
        variation = np.max(np.abs(norms - norms[0]))
        assert variation < 1e-4, f"Mass not conserved: variation {variation}"
        print(f"  ✓ Mass variation: {variation:.2e}")
        
        return True
    
    @staticmethod
    def test_linearity_preservation():
        """Verify solver handles linear case (h=0) without errors."""
        print("\nTest: Linear Case (h=0)")
        
        cfg = VCNLSConfig(
            N=50,
            T=0.5,
            k=0.01,
            h_coeff=lambda t: 0.0,  # Linear case
            verbose=False
        )
        
        psi0 = lambda x: np.exp(-x**2)
        phi0 = lambda x: np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        
        integrator = VCNLSIntegrator(cfg)
        V0 = integrator.set_initial_condition(psi0, phi0)
        result = integrator.integrate(V0)
        
        print("  ✓ Linear case integrated without errors")
        return True
    
    @staticmethod
    def run_all_tests():
        """Execute all unit tests."""
        print("=" * 60)
        print("VCNLS SOLVER - UNIT TESTS")
        print("=" * 60)
        
        tests = [
            VCNLSUnitTests.test_matrix_symmetry,
            VCNLSUnitTests.test_newton_convergence,
            VCNLSUnitTests.test_mass_conservation,
            VCNLSUnitTests.test_linearity_preservation
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)
        
        return failed == 0


# Run tests if executed directly
if __name__ == "__main__":
    VCNLSUnitTests.run_all_tests()