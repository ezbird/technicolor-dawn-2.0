#!/usr/bin/env python3
"""
Density Threshold Test for Cooling in Gadget

This script creates a visualization of the cooling efficiency threshold
by comparing cooling times to dynamical times across the temperature-density phase space.
Uses a cooling function that closely matches the Gadget-4 implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm
import os
import argparse

def calculate_cooling_function(T, ne_fraction=1.0):
    """
    Accurate cooling function based on Gadget implementation.
    Incorporates collisional ionization, recombination, and free-free processes.
    
    Args:
        T: Temperature in Kelvin (can be a scalar or NumPy array)
        ne_fraction: electron fraction relative to hydrogen number density
                    (scalar or array matching T's shape)
    
    Returns:
        Cooling rate in erg cm³ s⁻¹
    """
    # Constants
    eV_to_K = 11604.505    # Conversion factor from eV to Kelvin
    eV_to_erg = 1.602e-12  # Conversion factor from eV to erg
    
    # Ensure T is a NumPy array
    T = np.asarray(T)
    
    # If ne_fraction is a scalar, convert to array matching T's shape
    if np.isscalar(ne_fraction):
        ne_fraction = np.full_like(T, ne_fraction)
    elif len(ne_fraction) != len(T):
        # If array lengths don't match, use a scalar default
        print(f"Warning: ne_fraction shape {np.shape(ne_fraction)} doesn't match T shape {np.shape(T)}. Using default value 1.0.")
        ne_fraction = np.full_like(T, 1.0)
    
    # Hydrogen and Helium fractions (from Gadget)
    XH = 0.76
    yhelium = (1.0 - XH) / (4.0 * XH)
    
    # Initialize the cooling rate array
    cooling_rate = np.zeros_like(T)
    
    # Compute temperature factor used in several cooling terms
    tfact = 1.0 / (1.0 + np.sqrt(T / 1e5))
    
    # Free-free emission (Betaff in Gadget)
    lambda_ff = 1.43e-27 * np.sqrt(T) * (1.1 + 0.34 * np.exp(-np.power(5.5 - np.log10(T), 2)/3.0))
    
    # Create electron fraction array for each point
    ne_array = np.ones_like(T)  # Default to fully ionized
    
    # For high temperatures (T > 1e5), assume full ionization
    high_T_mask = T > 1e5
    ne_array[high_T_mask] = 1.0
    
    # For low temperatures (T < 1e4), estimate partial ionization
    low_T_mask = T < 1e4
    ne_array[low_T_mask] = 0.01 + 0.99 * (T[low_T_mask] / 1e4)  # Linear interpolation
    
    # For intermediate temperatures, use input ne_fraction or estimate
    mid_T_mask = (T >= 1e4) & (T <= 1e5)
    ne_array[mid_T_mask] = ne_fraction[mid_T_mask]
    
    # Contribution from free-free emission - apply to all temperatures
    # Factor accounts for total ionized H and He
    cooling_rate += ne_array * lambda_ff * (1.0 + 0.6 * yhelium)
    
    # Apply temperature-dependent cooling mechanisms using masks
    
    # H⁰ collisional excitation (similar to BetaH0 in Gadget)
    mask_H0 = 118348.0/T < 70.0
    if np.any(mask_H0):
        lambda_exc_H0 = 7.5e-19 * np.exp(-118348.0/T[mask_H0]) * tfact[mask_H0]
        
        # Estimate neutral H fraction (decreases with temperature)
        neutral_H_frac = np.clip(1.0 - ne_array[mask_H0], 0.01, 0.99)
        
        cooling_rate[mask_H0] += ne_array[mask_H0] * lambda_exc_H0 * neutral_H_frac
    
    # He⁺ collisional excitation (BetaHep in Gadget)
    mask_Hep = 473638.0/T < 70.0
    if np.any(mask_Hep):
        lambda_exc_Hep = 5.54e-17 * np.power(T[mask_Hep], -0.397) * np.exp(-473638.0/T[mask_Hep]) * tfact[mask_Hep]
        
        # Simplified He+ abundance model
        Hep_fraction = np.minimum(ne_array[mask_Hep], 0.1) * yhelium
        
        cooling_rate[mask_Hep] += ne_array[mask_Hep] * lambda_exc_Hep * Hep_fraction
    
    # Collisional ionization losses
    T_eV = T / eV_to_K
    
    # H⁰ ionization
    mask_H0_ion = T > 5000.0  # Only significant at higher temperatures
    if np.any(mask_H0_ion):
        U = 13.6/T_eV[mask_H0_ion]
        gamma_eH0 = 0.291e-7 * np.power(U, 0.39) * np.exp(-U) / (0.232 + U)
        lambda_ion_H0 = 2.18e-11 * gamma_eH0
        
        # Estimate neutral H fraction (decreases with temperature)
        neutral_H_frac = np.clip(1.0 - ne_array[mask_H0_ion], 0.01, 0.99)
        
        cooling_rate[mask_H0_ion] += ne_array[mask_H0_ion] * lambda_ion_H0 * neutral_H_frac
    
    # Very high temperatures - fully ionized regime
    mask_very_high = T >= 1e7
    if np.any(mask_very_high):
        cooling_rate[mask_very_high] = 2.0e-23 * np.sqrt(T[mask_very_high])  # Simplified for very high T
    
    return cooling_rate

def cooling_time(u, rho, mu=0.6, gamma=5/3, ne_fraction=1.0):
    """
    Calculate cooling time based on internal energy and density.
    
    Args:
        u: Internal energy in code units
        rho: Density in code units
        mu: Mean molecular weight
        gamma: Adiabatic index
        ne_fraction: electron fraction relative to hydrogen number density
    
    Returns:
        Cooling time in code units
    """
    # Constants in cgs
    k_B = 1.380649e-16  # Boltzmann constant [erg/K]
    m_p = 1.672622e-24  # Proton mass [g]
    
    # Convert to physical quantities
    rho_cgs = rho * 6.77e-22  # Density conversion to g/cm³
    u_cgs = u * 1e10  # Internal energy conversion to (cm/s)²
    
    # Temperature from internal energy
    T = mu * m_p * (gamma-1) * u_cgs / k_B
    
    # Particle number density
    n_H = rho_cgs / (mu * m_p)
    
    # If ne_fraction is an array, make sure it has the same shape as T
    if not np.isscalar(ne_fraction) and len(ne_fraction) != len(T):
        print(f"Warning: ne_fraction shape {np.shape(ne_fraction)} doesn't match T shape {np.shape(T)}. Using default value 1.0.")
        ne_fraction = 1.0  # Fall back to a scalar
    
    # Calculate cooling rate
    cooling_rate = calculate_cooling_function(T, ne_fraction)
    
    # Cooling time = thermal energy / cooling rate
    u_thermal = u_cgs  # thermal energy per unit mass
    cooling_rate_per_mass = n_H * cooling_rate / rho_cgs  # erg/s/g
    
    # Handle zero or negative cooling rates (heating or no cooling)
    # Set cooling time to a very large value
    mask_zero_cooling = cooling_rate <= 0
    if np.any(mask_zero_cooling):
        t_cool = np.zeros_like(u_thermal)
        non_zero_mask = ~mask_zero_cooling
        
        if np.any(non_zero_mask):
            t_cool[non_zero_mask] = u_thermal[non_zero_mask] / cooling_rate_per_mass[non_zero_mask]
        
        # Set large cooling time for regions with zero or negative cooling
        t_cool[mask_zero_cooling] = 1.0e20  # A very large number in seconds
    else:
        t_cool = u_thermal / cooling_rate_per_mass
    
    # Convert back to code units (assuming 1 Gyr time unit)
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
    u_cgs = internal_energy * 1e10  # to (cm/s)²
    
    # T = (gamma-1) * u * mu * m_p / k_B
    temperature = (gamma - 1) * u_cgs * mu * m_p / k_B
    
    return temperature

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
                gamma = 5.0/3.0
                mu = 0.6  # assumption for fully ionized primordial gas
                gas_temperature = estimate_temperature(gas_internal_energy, mu, gamma)
            
            # Try to get electron abundance if available
            if 'PartType0/ElectronAbundance' in f:
                ne_fraction = f['PartType0/ElectronAbundance'][:]
                print(f"Found electron abundance in snapshot. Shape: {ne_fraction.shape}")
            else:
                # Assume fully ionized gas for temperatures > 10^4 K, partially ionized below
                ne_fraction = np.ones_like(gas_temperature)
                ne_fraction[gas_temperature < 1e4] = 0.5  # Rough approximation for cooler gas
                print(f"Using default electron abundance. Shape: {ne_fraction.shape}")
            
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
    
    # Print some info about arrays to help debug
    print(f"Gas density shape: {gas_density.shape}")
    print(f"Gas internal energy shape: {gas_internal_energy.shape}")
    print(f"Gas temperature shape: {gas_temperature.shape}")
    print(f"Temperature range: {gas_temperature.min():.2e} - {gas_temperature.max():.2e} K")
    
    # Calculate cooling and dynamical times
    print("Calculating cooling times...")
    t_cool = cooling_time(gas_internal_energy, gas_density, ne_fraction=1.0)  # Use scalar for safety
    
    print("Calculating dynamical times...")
    t_dyn = dynamical_time(gas_density)
    
    # Calculate the ratio of cooling time to dynamical time
    cooling_ratio = t_cool / t_dyn
    
    print(f"Cooling ratio range: {cooling_ratio.min():.2e} - {cooling_ratio.max():.2e}")
    
    # Create temperature-density phase space
    print("Creating phase diagrams...")
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
    print("Creating cooling threshold grid...")
    
    # Use a reasonable range based on the data
    rho_min = max(1e-6, gas_density.min())
    rho_max = min(1e3, gas_density.max())
    rho_grid = np.logspace(np.log10(rho_min), np.log10(rho_max), 50)
    T_grid = np.logspace(2, 8, 50)  # From 100K to 10^8K
    
    rho_mesh, T_mesh = np.meshgrid(rho_grid, T_grid)
    rho_mesh_flat = rho_mesh.flatten()
    T_mesh_flat = T_mesh.flatten()
    
    # Calculate cooling time on the grid
    # Convert T to internal energy for the cooling_time function
    k_B = 1.380649e-16  # Boltzmann constant [erg/K]
    m_p = 1.672622e-24  # Proton mass [g]
    mu = 0.6
    gamma = 5.0/3.0
    u_grid_flat = k_B * T_mesh_flat / (mu * m_p * (gamma-1)) / 1e10  # Convert to code units
    
    # Calculate cooling and dynamical times for the grid (use flattened arrays)
    print("Calculating grid cooling times...")
    t_cool_grid_flat = cooling_time(u_grid_flat, rho_mesh_flat, ne_fraction=1.0)
    
    print("Calculating grid dynamical times...")
    t_dyn_grid_flat = dynamical_time(rho_mesh_flat)
    
    # Reshape back to grid
    t_cool_grid = t_cool_grid_flat.reshape(T_mesh.shape)
    t_dyn_grid = t_dyn_grid_flat.reshape(T_mesh.shape)
    
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
    cbar.set_label(r'log$_{10}(t_{cool}/t_{dyn}'))
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