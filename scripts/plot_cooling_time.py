#!/usr/bin/env python3
"""
Density Threshold Test for Cooling in Gadget

This script creates a visualization of the cooling efficiency threshold
by comparing cooling times to dynamical times across the temperature-density phase space.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm
import os
from scipy import interpolate
import argparse

def calculate_cooling_function(T, ne_fraction=1.0):
    """
    Approximate cooling function for primordial gas.
    Based on Sutherland & Dopita (1993) for primordial composition.
    
    Args:
        T: Temperature in Kelvin (can be a NumPy array)
        ne_fraction: electron fraction relative to hydrogen number density
    
    Returns:
        Cooling rate in erg cm³ s⁻¹
    """
    # Convert to NumPy array if it's not already
    T = np.asarray(T)
    log_T = np.log10(T)
    
    # Define conditions and corresponding functions
    condlist = [
        log_T < 4.0,
        (log_T >= 4.0) & (log_T < 5.5),
        (log_T >= 5.5) & (log_T < 7.5),
        log_T >= 7.5
    ]
    
    funclist = [
        lambda t: 1e-24 * np.sqrt(t),
        lambda t: 1e-22 * np.power(t/1e4, -1.5),
        lambda t: 2e-23 * np.sqrt(t),
        lambda t: 5e-23 * np.sqrt(t)
    ]
    
    return np.piecewise(T, condlist, funclist)

def cooling_time(u, rho, mu=0.6, gamma=5/3, cooling_func=None):
    """
    Calculate cooling time based on internal energy and density.
    
    Args:
        u: Internal energy in code units
        rho: Density in code units
        mu: Mean molecular weight
        gamma: Adiabatic index
        cooling_func: Function to calculate cooling rate
    
    Returns:
        Cooling time in code units
    """
    # Constants in cgs
    k_B = 1.380649e-16  # Boltzmann constant [erg/K]
    m_p = 1.672622e-24  # Proton mass [g]
    
    # Convert to physical quantities (approximate)
    # These conversion factors should match your simulation
    rho_cgs = rho * 6.77e-22  # Density conversion to g/cm³
    u_cgs = u * 1e10  # Internal energy conversion to (cm/s)²
    
    # Temperature from internal energy
    T = mu * m_p * (gamma-1) * u_cgs / k_B
    
    # Particle number density
    n_H = rho_cgs / (mu * m_p)
    
    # If custom cooling function provided, use it
    if cooling_func is None:
        cooling_rate = calculate_cooling_function(T)
    else:
        cooling_rate = cooling_func(T)
    
    # Cooling time = thermal energy / cooling rate
    u_thermal = u_cgs  # thermal energy per unit mass
    cooling_rate_per_mass = n_H * cooling_rate / rho_cgs  # erg/s/g
    
    t_cool = u_thermal / cooling_rate_per_mass  # seconds
    
    # Convert back to code units
    # Assuming time unit of approximately 3.08568e16 seconds (≈ 1 Gyr)
    t_cool_code = t_cool / 3.08568e16
    
    return t_cool_code

def dynamical_time(rho, G=43007.1):
    """
    Calculate dynamical time based on density.
    
    Args:
        rho: Density in code units
        G: Gravitational constant in code units
    
    Returns:
        Dynamical time in code units
    """
    return 1.0 / np.sqrt(G * rho)

def analyze_snapshot(filename, output_prefix="cooling_test"):
    """
    Analyze a Gadget snapshot to test cooling thresholds.
    
    Args:
        filename: Path to the Gadget HDF5 snapshot
        output_prefix: Prefix for output files
    """
    print(f"Analyzing {filename}...")
    
    # Load snapshot data
    with h5py.File(filename, 'r') as f:
        # Get gas properties
        try:
            gas_density = f['PartType0/Density'][:]
            gas_internal_energy = f['PartType0/InternalEnergy'][:]
            
            # Try to get temperature directly if available
            if 'PartType0/Temperature' in f:
                gas_temperature = f['PartType0/Temperature'][:]
            else:
                # Estimate temperature from internal energy
                # T = (gamma-1) * u * mu * m_p / k_B
                gamma = 5.0/3.0
                mu = 0.6  # assumption for fully ionized primordial gas
                gas_temperature = estimate_temperature(gas_internal_energy, mu, gamma)
            
            # Get some header info
            redshift = f['Header'].attrs['Redshift']
            time = f['Header'].attrs['Time']
            box_size = f['Header'].attrs['BoxSize']
            
            # Get masses if available, otherwise assume uniform
            if 'PartType0/Masses' in f:
                gas_masses = f['PartType0/Masses'][:]
            else:
                gas_masses = np.ones_like(gas_density)
                
        except KeyError as e:
            print(f"Error: Could not find required dataset: {e}")
            return
    
    # Calculate cooling and dynamical times
    t_cool = cooling_time(gas_internal_energy, gas_density)
    t_dyn = dynamical_time(gas_density)
    
    # Calculate the ratio of cooling time to dynamical time
    cooling_ratio = t_cool / t_dyn
    
    # Create temperature-density phase space
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Temperature-density diagram colored by gas mass
    ax = axes[0, 0]
    h, xedges, yedges, im = ax.hist2d(
        np.log10(gas_density), 
        np.log10(gas_temperature),
        bins=100, 
        weights=gas_masses,
        norm=LogNorm(),
        cmap='viridis'
    )
    ax.set_xlabel(r'log$_{10}(\rho)$ [code units]')
    ax.set_ylabel(r'log$_{10}(T)$ [K]')
    ax.set_title(f'Gas Mass Distribution (z={redshift:.2f})')
    plt.colorbar(im, ax=ax, label='Gas Mass')
    
    # Temperature-density diagram colored by cooling/dynamical time ratio
    ax = axes[0, 1]
    scatter = ax.scatter(
        np.log10(gas_density), 
        np.log10(gas_temperature),
        c=np.log10(cooling_ratio),
        cmap='coolwarm',
        vmin=-2, vmax=2,
        alpha=0.5,
        s=1
    )
    ax.set_xlabel(r'log$_{10}(\rho)$ [code units]')
    ax.set_ylabel(r'log$_{10}(T)$ [K]')
    ax.set_title(r'Cooling Efficiency: log$_{10}(t_{cool}/t_{dyn})$')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'log$_{10}(t_{cool}/t_{dyn})$')
    
    # Add a line where cooling time equals dynamical time
    # Create a grid of density and temperature values
    rho_grid = np.logspace(np.log10(gas_density.min()), np.log10(gas_density.max()), 100)
    T_grid = np.logspace(1, 8, 100)
    
    rho_mesh, T_mesh = np.meshgrid(rho_grid, T_grid)
    
    # Calculate cooling time on the grid
    # Convert T to internal energy for the cooling_time function
    k_B = 1.380649e-16  # Boltzmann constant [erg/K]
    m_p = 1.672622e-24  # Proton mass [g]
    mu = 0.6
    gamma = 5.0/3.0
    u_grid = k_B * T_mesh / (mu * m_p * (gamma-1)) / 1e10  # Convert to code units
    
    t_cool_grid = np.zeros_like(rho_mesh)
    t_dyn_grid = np.zeros_like(rho_mesh)
    
    for i in range(len(rho_grid)):
        for j in range(len(T_grid)):
            t_cool_grid[j, i] = cooling_time(u_grid[j, i], rho_mesh[j, i])
            t_dyn_grid[j, i] = dynamical_time(rho_mesh[j, i])
    
    ratio_grid = t_cool_grid / t_dyn_grid
    
    # Plot the contour where ratio = 1
    ax = axes[1, 0]
    contour = ax.contour(
        np.log10(rho_mesh), 
        np.log10(T_mesh), 
        ratio_grid,
        levels=[0.01, 0.1, 1.0, 10.0, 100.0],
        colors=['blue', 'cyan', 'black', 'orange', 'red'],
        linewidths=2
    )
    ax.clabel(contour, inline=True, fontsize=10)
    
    # Plot the particle distribution again for reference
    h, xedges, yedges, im = ax.hist2d(
        np.log10(gas_density), 
        np.log10(gas_temperature),
        bins=100, 
        cmap='Greys',
        norm=LogNorm(),
        alpha=0.3
    )
    ax.set_xlabel(r'log$_{10}(\rho)$ [code units]')
    ax.set_ylabel(r'log$_{10}(T)$ [K]')
    ax.set_title('Cooling Threshold (t_cool = t_dyn contour in black)')
    
    # Histogram of cooling/dynamical time ratio
    ax = axes[1, 1]
    ax.hist(np.log10(cooling_ratio), bins=50, alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label='t_cool = t_dyn')
    ax.set_xlabel(r'log$_{10}(t_{cool}/t_{dyn})$')
    ax.set_ylabel('Number of Particles')
    ax.set_title('Distribution of Cooling Efficiency')
    ax.legend()
    
    # Add some statistics
    efficiently_cooling = np.sum(cooling_ratio < 1.0) / len(cooling_ratio) * 100
    text = f"Fraction of gas with t_cool < t_dyn: {efficiently_cooling:.1f}%"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_phase_diagram.png", dpi=300)
    print(f"Plot saved as {output_prefix}_phase_diagram.png")
    
    # Create a second figure focused on the cooling threshold
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a more detailed contour plot
    contour = ax.contourf(
        np.log10(rho_mesh), 
        np.log10(T_mesh), 
        np.log10(ratio_grid),
        levels=np.linspace(-2, 2, 20),
        cmap='coolwarm',
        extend='both'
    )
    
    # Add contour lines for specific ratios
    line_contour = ax.contour(
        np.log10(rho_mesh), 
        np.log10(T_mesh), 
        ratio_grid,
        levels=[0.01, 0.1, 1.0, 10.0, 100.0],
        colors='black',
        linewidths=1.5
    )
    ax.clabel(line_contour, inline=True, fontsize=10, fmt='%1.2f')
    
    # Plot the particle distribution as scatter
    ax.scatter(
        np.log10(gas_density), 
        np.log10(gas_temperature),
        s=0.5,
        color='white',
        alpha=0.3
    )
    
    # Add colorbar and labels
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(r'log$_{10}(t_{cool}/t_{dyn})$')
    ax.set_xlabel(r'log$_{10}(\rho)$ [code units]')
    ax.set_ylabel(r'log$_{10}(T)$ [K]')
    ax.set_title(f'Cooling Threshold Map (z={redshift:.2f})')
    
    # Add a text box with summary info
    cold_fraction = np.sum(gas_temperature < 100) / len(gas_temperature) * 100
    hot_fraction = np.sum(gas_temperature > 10000) / len(gas_temperature) * 100
    
    info_text = (
        f"Total gas particles: {len(gas_temperature)}\n"
        f"Cold gas (T < 100K): {cold_fraction:.1f}%\n"
        f"Hot gas (T > 10000K): {hot_fraction:.1f}%\n"
        f"Efficiently cooling: {efficiently_cooling:.1f}%"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cooling_threshold.png", dpi=300)
    print(f"Threshold map saved as {output_prefix}_cooling_threshold.png")
    
    return

def estimate_temperature(internal_energy, mu=0.6, gamma=5/3):
    """
    Estimate temperature from internal energy.
    
    Args:
        internal_energy: Internal energy per unit mass in code units
        mu: Mean molecular weight
        gamma: Adiabatic index
    
    Returns:
        Temperature in Kelvin
    """
    # Constants in cgs
    k_B = 1.380649e-16  # Boltzmann constant [erg/K]
    m_p = 1.672622e-24  # Proton mass [g]
    
    # Convert internal energy to cgs (assuming Gadget internal energy units)
    # This is an approximation - adjust conversion factor based on your exact code units
    u_cgs = internal_energy * 1e10  # to (cm/s)²
    
    # T = (gamma-1) * u * mu * m_p / k_B
    temperature = (gamma - 1) * u_cgs * mu * m_p / k_B
    
    return temperature

def main():
    parser = argparse.ArgumentParser(description='Analyze cooling thresholds in Gadget snapshots')
    parser.add_argument('snapshot', help='Path to Gadget snapshot file')
    parser.add_argument('--output', '-o', default='cooling_test', help='Output prefix for generated files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.snapshot):
        print(f"Error: Snapshot file {args.snapshot} not found")
        return
    
    analyze_snapshot(args.snapshot, args.output)

if __name__ == "__main__":
    main()