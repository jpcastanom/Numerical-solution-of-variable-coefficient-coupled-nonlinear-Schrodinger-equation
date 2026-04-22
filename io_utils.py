"""
VCNLS Visualization and I/O Utilities

Provides functions for reconstructing wavefunctions from the state vector
and creating visualizations of the solution.
"""

import numpy as np
import matplotlib.pyplot as plt


class VCNLSVisualizer:
    """
    Visualization tools for VCNLS simulation results.
    
    Handles reconstruction of complex wavefunctions ψ and φ from the
    real state vector V = [u1, u2, u3, u4]^T.
    """
    
    @staticmethod
    def reconstruct_wavefunctions(states, n_nodes):
        """
        Reconstruct complex wavefunctions ψ and φ from state vector V.
        
        The state vector stores:
            V[4m] = u1_m = Re(ψ(x_m))
            V[4m+1] = u2_m = Im(ψ(x_m))
            V[4m+2] = u3_m = Re(φ(x_m))
            V[4m+3] = u4_m = Im(φ(x_m))
        
        Reconstruction:
            ψ(x_m) = u1_m + i*u2_m
            φ(x_m) = u3_m + i*u4_m
            
        Args:
            states: State array of shape (n_times, 4*n_nodes) or (4*n_nodes,)
            n_nodes: Number of spatial nodes (N+1)
            
        Returns:
            tuple: (psi, phi) where each is a complex array of shape (n_times, n_nodes)
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        n_times = states.shape[0]
        psi = np.zeros((n_times, n_nodes), dtype=np.complex128)
        phi = np.zeros((n_times, n_nodes), dtype=np.complex128)
        
        for i in range(n_times):
            V = states[i]
            for m in range(n_nodes):
                # Reconstruct complex values from real components
                psi[i, m] = V[4 * m] + 1j * V[4 * m + 1]
                phi[i, m] = V[4 * m + 2] + 1j * V[4 * m + 3]
        
        return psi, phi
    
    @staticmethod
    def plot_snapshot(x, psi, phi, time, ax=None, title=None):
        """
        Plot a single time snapshot showing |ψ|² and |φ|².
        
        Args:
            x: Spatial grid array
            psi: Complex wavefunction ψ at single time
            phi: Complex wavefunction φ at single time
            time: Time value for title
            ax: Matplotlib axes (creates new if None)
            title: Custom title (optional)
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot probability densities
        ax.plot(x, np.abs(psi)**2, 'b-', label=r'$|\psi|^2$', linewidth=2)
        ax.plot(x, np.abs(phi)**2, 'r--', label=r'$|\phi|^2$', linewidth=2)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(title or f't = {time:.3f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_evolution(x, times, psi, phi):
        """
        Plot multiple time snapshots showing the evolution.
        
        Creates a 2×3 subplot grid with 6 time instances.
        
        Args:
            x: Spatial grid array
            times: Time points array
            psi: Complex wavefunction ψ array (n_times × n_nodes)
            phi: Complex wavefunction φ array (n_times × n_nodes)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select 6 evenly spaced time points
        n_plots = min(6, len(times))
        indices = np.linspace(0, len(times) - 1, n_plots, dtype=int)
        
        for idx, ax in zip(indices, axes):
            VCNLSVisualizer.plot_snapshot(
                x, psi[idx], phi[idx], times[idx], ax=ax
            )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_amplitude_map(x, times, psi, phi):
        """
        Create contour plots showing the full spatiotemporal evolution.
        
        Args:
            x: Spatial grid array
            times: Time points array
            psi: Complex wavefunction ψ array (n_times × n_nodes)
            phi: Complex wavefunction φ array (n_times × n_nodes)
            
        Returns:
            Matplotlib figure object with two subplots
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Contour plot for |ψ|²
        im1 = ax1.contourf(x, times, np.abs(psi)**2, levels=50, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title(r'$|\psi(x,t)|^2$')
        plt.colorbar(im1, ax=ax1)
        
        # Contour plot for |φ|²
        im2 = ax2.contourf(x, times, np.abs(phi)**2, levels=50, cmap='plasma')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_title(r'$|\phi(x,t)|^2$')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        return fig