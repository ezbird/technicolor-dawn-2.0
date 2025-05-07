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

def read_snapshot(filename):
    """Read density and temperature data from a Gadget-4 snapshot."""
    with h5py.File(filename, 'r') as f:
        # Get header information
        header = f['Header'].attrs
        redshift = header['Redshift']
        time = header['Time']
        h = header.get('HubbleParam', 0.7)  # Default to 0.7 if not found
        
        # Gas properties
        if 'PartType0' not in f or 'Density' not in f['PartType0']:
            print(f"No gas particles or density found in {filename}")
            return None, None, None, None
        
        # Get gas density
        density = f['PartType0/Density'][:]
        
        # Get or calculate temperature based on what's available
        temperature = None
        if 'Temperature' in f['PartType0']:
            temperature = f['PartType0/Temperature'][:]
        elif 'InternalEnergy' in f['PartType0']:
            # Calculate temperature from internal energy
            u = f['PartType0/InternalEnergy'][:]
            
            # Approximate temperature conversion (assuming primordial gas)
            # T = 2/3 * u * μ * m_p / k_B
            # where μ is mean molecular weight (≈ 0.6 for ionized gas)
            mu = 0.6  # mean molecular weight
            gamma = 5/3  # adiabatic index
            m_p = 1.67e-24  # proton mass in g
            k_B = 1.38e-16  # Boltzmann constant in erg/K
            
            # Calculate temperature (K)
            temperature = (gamma - 1) * u * mu * m_p / k_B
        else:
            print(f"No temperature or internal energy information found in {filename}")
            return None, None, None, None
    
    return density, temperature, redshift, time

def analyze_snapshot(filename, output_prefix=None):
    """Analyze a snapshot for adiabatic behavior and create plots."""
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(filename))[0]
    
    # Read data
    density, temperature, redshift, time = read_snapshot(filename)
    if density is None or temperature is None:
        return False
    
    # Calculate logarithms
    log_density = np.log10(density)
    log_temperature = np.log10(temperature)
    
    # Perform linear regression to find the slope (which should be γ-1 for adiabatic)
    mask = np.isfinite(log_density) & np.isfinite(log_temperature)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_density[mask], log_temperature[mask])
    
    # Expected slope for adiabatic evolution
    gamma = 5/3  # For monatomic gas
    expected_slope = gamma - 1  # = 2/3 ≈ 0.667
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create the 2D histogram plot
    h, xedges, yedges, img = plt.hist2d(
        log_density, 
        log_temperature,
        bins=100,
        range=[[-32, -24], [1, 8]],  # Adjust these ranges as needed
        norm=LogNorm(),
        cmap='viridis'
    )
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Number of Gas Particles')
    
    # Plot the best-fit line
    x_vals = np.linspace(min(log_density[mask]), max(log_density[mask]), 100)
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

def analyze_multiple_snapshots(directory, output_dir=None):
    """Analyze all snapshots in a directory to track evolution of the adiabatic relation."""
    if output_dir is None:
        output_dir = "adiabatic_analysis"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all snapshot files
    snapshot_files = sorted(glob.glob(os.path.join(directory, "snapshot_*.hdf5")))
    if not snapshot_files:
        print(f"No snapshot files found in {directory}")
        return False
    
    print(f"Found {len(snapshot_files)} snapshot files")
    
    # Lists to store results
    redshifts = []
    times = []
    slopes = []
    r_squared = []
    deviation_pct = []
    
    # Process each snapshot
    for filename in snapshot_files:
        print(f"Processing {filename}...")
        density, temperature, redshift, time = read_snapshot(filename)
        if density is None or temperature is None:
            continue
        
        # Calculate logarithms
        log_density = np.log10(density)
        log_temperature = np.log10(temperature)
        
        # Perform linear regression
        mask = np.isfinite(log_density) & np.isfinite(log_temperature)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_density[mask], log_temperature[mask])
        
        # Expected slope for adiabatic evolution
        gamma = 5/3  # For monatomic gas
        expected_slope = gamma - 1  # = 2/3 ≈ 0.667
        
        # Store results
        redshifts.append(redshift)
        times.append(time)
        slopes.append(slope)
        r_squared.append(r_value**2)
        deviation_pct.append(100*abs(slope-expected_slope)/expected_slope)
        
        # Create individual snapshot plot
        snapshot_prefix = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0])
        analyze_snapshot(filename, snapshot_prefix)
    
    if not redshifts:
        print("No valid snapshots processed")
        return False
    
    # Sort all results by time (or redshift)
    sorted_indices = np.argsort(times)
    redshifts = np.array(redshifts)[sorted_indices]
    times = np.array(times)[sorted_indices]
    slopes = np.array(slopes)[sorted_indices]
    r_squared = np.array(r_squared)[sorted_indices]
    deviation_pct = np.array(deviation_pct)[sorted_indices]
    
    # Create evolution plot
    plt.figure(figsize=(12, 8))
    
    # Plot slope evolution
    plt.subplot(2, 1, 1)
    plt.plot(redshifts, slopes, 'bo-', linewidth=2)
    plt.axhline(y=2/3, color='r', linestyle='--', label='Adiabatic (γ-1 = 2/3)')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Slope (γ-1)')
    plt.title('Evolution of Temperature-Density Relation Slope')
    plt.grid(True)
    plt.legend()
    
    # Plot deviation percentage
    plt.subplot(2, 1, 2)
    plt.plot(redshifts, deviation_pct, 'go-', linewidth=2)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Deviation from Adiabatic (%)')
    plt.title('Deviation from Adiabatic Relation')
    plt.grid(True)
    
    # Save the evolution plot
    evolution_file = os.path.join(output_dir, "adiabatic_evolution.png")
    plt.tight_layout()
    plt.savefig(evolution_file, dpi=300)
    plt.close()
    print(f"Evolution plot saved to {evolution_file}")
    
    # Save the evolution data
    data_file = os.path.join(output_dir, "adiabatic_evolution_data.txt")
    with open(data_file, 'w') as f:
        f.write("# Evolution of adiabatic relation in snapshots\n")
        f.write("# Redshift  Time  Slope  R-squared  Deviation(%)\n")
        for i in range(len(redshifts)):
            f.write(f"{redshifts[i]:.4f}  {times[i]:.6f}  {slopes[i]:.6f}  {r_squared[i]:.6f}  {deviation_pct[i]:.2f}\n")
    
    print(f"Evolution data saved to {data_file}")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Gadget-4 snapshots for adiabatic behavior')
    parser.add_argument('--file', help='Analyze a single snapshot file')
    parser.add_argument('--dir', help='Analyze all snapshots in a directory')
    parser.add_argument('--output', help='Output directory for analysis results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.file:
        if os.path.exists(args.file):
            analyze_snapshot(args.file, args.output)
        else:
            print(f"File not found: {args.file}")
    elif args.dir:
        if os.path.isdir(args.dir):
            analyze_multiple_snapshots(args.dir, args.output)
        else:
            print(f"Directory not found: {args.dir}")
    else:
        print("Please specify either a file (--file) or directory (--dir) to analyze")

if __name__ == "__main__":
    main()