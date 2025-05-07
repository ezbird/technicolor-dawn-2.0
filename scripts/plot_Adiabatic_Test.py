#!/usr/bin/env python3
"""
Adiabatic Relation Test for Gadget-4 Snapshots

This script analyzes Gadget-4 snapshots to determine if the temperature-density
relationship follows the adiabatic relation T ∝ ρ^(γ-1).

For an ideal gas with adiabatic index γ, we expect:
    T ∝ ρ^(γ-1)  or  log(T) = (γ-1)log(ρ) + constant

For pure adiabatic evolution, the gas should follow this relation with γ = 5/3 for monatomic gas.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import os
import argparse
from scipy import stats
from matplotlib.colors import LogNorm

def read_snapshot(filename, verbose=False):
    """Read density and temperature data from a Gadget-4 snapshot."""
    try:
        with h5py.File(filename, 'r') as f:
            # Print file structure if verbose
            if verbose:
                print(f"File structure for {filename}:")
                for key in f.keys():
                    print(f" - {key}")
                    if isinstance(f[key], h5py.Group):
                        for subkey in f[key].keys():
                            print(f"   - {subkey}")
                
                # Print header attributes
                if 'Header' in f:
                    print("Header attributes:")
                    for key, value in f['Header'].attrs.items():
                        print(f"  {key}: {value}")
            
            # Get header information
            header = f['Header'].attrs
            redshift = header['Redshift']
            time = header['Time']
            h = header.get('HubbleParam', 0.7)  # Default to 0.7 if not found
            boxsize = header.get('BoxSize', 1.0)
            
            # Gas properties
            if 'PartType0' not in f:
                print(f"No gas particles found in {filename}")
                return None, None, None, None, None
            
            # Get gas density
            if 'Density' in f['PartType0']:
                density = f['PartType0/Density'][:]
            else:
                print(f"No density information found in {filename}")
                return None, None, None, None, None
            
            # Get or calculate temperature based on what's available
            temperature = None
            internal_energy = None
            
            if 'Temperature' in f['PartType0']:
                temperature = f['PartType0/Temperature'][:]
                if verbose:
                    print(f"Found direct temperature data. Range: {np.min(temperature)} - {np.max(temperature)} K")
            elif 'InternalEnergy' in f['PartType0']:
                internal_energy = f['PartType0/InternalEnergy'][:]
                
                # We'll calculate temperature later based on mean molecular weight
                if verbose:
                    print(f"Found internal energy data. Range: {np.min(internal_energy)} - {np.max(internal_energy)}")
            else:
                print(f"No temperature or internal energy information found in {filename}")
                return None, None, None, None, None
            
            # Also get electron abundance if available (for mean molecular weight)
            ne = None
            if 'ElectronAbundance' in f['PartType0']:
                ne = f['PartType0/ElectronAbundance'][:]
                if verbose:
                    print(f"Found electron abundance data. Range: {np.min(ne)} - {np.max(ne)}")
            
            return density, internal_energy, temperature, ne, redshift, time
    except Exception as e:
        print(f"Error reading snapshot {filename}: {e}")
        return None, None, None, None, None, None

def calculate_temperature(internal_energy, ne=None, verbose=False):
    """Calculate temperature from internal energy."""
    # For primordial gas
    X_H = 0.76  # Hydrogen mass fraction
    Y_He = 1.0 - X_H  # Helium mass fraction
    
    # Calculate mean molecular weight
    if ne is not None:
        # With electron abundance
        mu = (1.0 + 4.0 * Y_He) / (1.0 + Y_He + ne)
    else:
        # Approximate for neutral gas
        mu = 1.22
    
    gamma = 5/3  # adiabatic index
    m_p = 1.67e-24  # proton mass in g
    k_B = 1.38e-16  # Boltzmann constant in erg/K
    
    # Calculate temperature (K)
    temperature = (gamma - 1) * internal_energy * mu * m_p / k_B
    
    if verbose:
        print(f"Calculated temperature. Range: {np.min(temperature)} - {np.max(temperature)} K")
        print(f"Using mu = {mu}")
    
    return temperature

def analyze_snapshot(filename, output_prefix=None, verbose=False):
    """Analyze a snapshot for adiabatic behavior and create plots."""
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(filename))[0]
    
    # Read data
    result = read_snapshot(filename, verbose)
    if len(result) == 6:
        density, internal_energy, temperature, ne, redshift, time = result
    else:
        print(f"Error reading snapshot {filename}")
        return False
    
    if density is None:
        return False
    
    # Calculate temperature if not directly available
    if temperature is None and internal_energy is not None:
        temperature = calculate_temperature(internal_energy, ne, verbose)
    
    if temperature is None:
        print("Unable to determine temperature")
        return False
    
    # Filter out any invalid values
    mask = (density > 0) & (temperature > 0) & np.isfinite(density) & np.isfinite(temperature)
    if np.sum(mask) == 0:
        print("No valid (density, temperature) pairs found")
        return False
    
    density = density[mask]
    temperature = temperature[mask]
    
    # Calculate logarithms
    log_density = np.log10(density)
    log_temperature = np.log10(temperature)
    
    # Print statistics
    if verbose:
        print(f"Valid data points: {len(log_density)}")
        print(f"Density range: {10**np.min(log_density):.2e} - {10**np.max(log_density):.2e} g/cm³")
        print(f"Temperature range: {10**np.min(log_temperature):.2f} - {10**np.max(log_temperature):.2f} K")
    
    # Perform linear regression to find the slope (which should be γ-1 for adiabatic)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_density, log_temperature)
    
    # Expected slope for adiabatic evolution
    gamma = 5/3  # For monatomic gas
    expected_slope = gamma - 1  # = 2/3 ≈ 0.667
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Determine density and temperature ranges for the plot
    density_min = np.min(log_density)
    density_max = np.max(log_density)
    temp_min = np.min(log_temperature)
    temp_max = np.max(log_temperature)
    
    # Add padding
    density_range = [density_min - 0.5, density_max + 0.5]
    temp_range = [temp_min - 0.5, temp_max + 0.5]
    
    # Create the 2D histogram plot with explicit density and temperature ranges
    try:
        h, xedges, yedges = np.histogram2d(
            log_density, 
            log_temperature,
            bins=100,
            range=[density_range, temp_range]
        )
        
        # Create a proper 2D histogram with pcolormesh to avoid colorbar issues
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        plt.pcolormesh(X, Y, h.T, norm=LogNorm(vmin=1), cmap='viridis')
        plt.colorbar(label='Number of Gas Particles')
    except Exception as e:
        print(f"Warning: Could not create 2D histogram: {e}")
        # Fall back to scatter plot
        plt.scatter(log_density, log_temperature, s=1, alpha=0.1, c='blue')
    
    # Plot the best-fit line
    x_vals = np.linspace(density_min, density_max, 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label=f'Best fit: slope = {slope:.3f}')
    
    # Plot the expected adiabatic relation
    y_adiabatic = expected_slope * x_vals + intercept
    plt.plot(x_vals, y_adiabatic, 'k--', linewidth=2, label=f'Adiabatic: slope = {expected_slope:.3f}')
    
    # Configure plot
    plt.xlabel(r'log$_{10}$(Density [g/cm$^3$])')
    plt.ylabel(r'log$_{10}$(Temperature [K])')
    plt.title(f'Temperature-Density Phase Diagram (z={redshift:.2f}, a={time:.4f})')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add annotation with statistics
    info_text = (
        f"Fitting results:\n"
        f"Slope: {slope:.4f} (expected: {expected_slope:.4f})\n"
        f"R²: {r_value**2:.4f}\n"
        f"The relation T ∝ ρ^{slope:.4f}\n"
        f"Adiabatic expects T ∝ ρ^{expected_slope:.4f}\n"
        f"Deviation: {100*abs(slope-expected_slope)/expected_slope:.2f}%"
    )
    
    plt.annotate(info_text, xy=(0.02, 0.96), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                va='top', fontsize=9)
    
    # Save plot
    output_file = f"{output_prefix}_adiabatic_test.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved to {output_file}")
    
    # Create a file with raw data for future reference
    data_file = f"{output_prefix}_adiabatic_data.txt"
    with open(data_file, 'w') as f:
        f.write(f"# Snapshot: {filename}\n")
        f.write(f"# Redshift: {redshift}\n")
        f.write(f"# Scale factor: {time}\n")
        f.write(f"# Adiabatic test results:\n")
        f.write(f"# Slope: {slope:.6f} (expected for adiabatic: {expected_slope:.6f})\n")
        f.write(f"# Intercept: {intercept:.6f}\n")
        f.write(f"# R-squared: {r_value**2:.6f}\n")
        f.write(f"# Deviation from adiabatic: {100*abs(slope-expected_slope)/expected_slope:.2f}%\n")
    
    print(f"Data saved to {data_file}")
    return True

def analyze_multiple_snapshots(directory, output_dir=None, verbose=False):
    """Analyze all snapshots in a directory to track evolution of the adiabatic relation."