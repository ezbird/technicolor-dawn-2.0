#!/usr/bin/env python3
"""
Temperature Histogram for Gadget Snapshots

This script creates detailed histograms of gas temperature distribution
from Gadget simulation snapshots, with options for weighting by mass
and examining specific density ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from matplotlib.ticker import LogFormatter, LogLocator

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

def analyze_temperature(filename, output_prefix="temp_hist", density_min=None, density_max=None):
    """
    Analyze temperature distribution in a Gadget snapshot.
    
    Args:
        filename: Path to the Gadget HDF5 snapshot
        output_prefix: Prefix for output files
        density_min: Minimum density to include (code units, optional)
        density_max: Maximum density to include (code units, optional)
    """
    print(f"Analyzing temperature distribution in {filename}...")
    
    # Load snapshot data
    with h5py.File(filename, 'r') as f:
        # Get gas properties
        try:
            gas_density = f['PartType0/Density'][:]
            gas_internal_energy = f['PartType0/InternalEnergy'][:]
            
            # Try to get temperature directly if available
            if 'PartType0/Temperature' in f:
                gas_temperature = f['PartType0/Temperature'][:]
                print("Using temperature directly from snapshot")
            else:
                # Estimate temperature from internal energy
                gamma = 5.0/3.0
                mu = 0.6  # assumption for fully ionized primordial gas
                gas_temperature = estimate_temperature(gas_internal_energy, mu, gamma)
                print("Estimated temperature from internal energy")
            
            # Get masses if available, otherwise assume uniform
            if 'PartType0/Masses' in f:
                gas_masses = f['PartType0/Masses'][:]
                print("Using particle masses from snapshot")
            else:
                gas_masses = np.ones_like(gas_density)
                print("Assuming uniform particle masses")
            
            # Get redshift from header
            redshift = f['Header'].attrs['Redshift']
            
        except KeyError as e:
            print(f"Error: Could not find required dataset: {e}")
            return
    
    # Apply density filter if specified
    mask = np.ones_like(gas_density, dtype=bool)
    if density_min is not None:
        mask = mask & (gas_density >= density_min)
    if density_max is not None:
        mask = mask & (gas_density <= density_max)
    
    # Apply mask to all arrays
    gas_temperature = gas_temperature[mask]
    gas_density = gas_density[mask]
    gas_masses = gas_masses[mask]
    
    print(f"Temperature range: {gas_temperature.min():.2e} - {gas_temperature.max():.2e} K")
    print(f"Number of gas particles analyzed: {len(gas_temperature)}")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create linear histogram first (with both count and mass weighting)
    plt.figure(figsize=(12, 6))
    
    # Set up temperature bins on a linear scale
    # Focus on the interesting range between floor and ~1000K
    temp_min = max(1, gas_temperature.min() * 0.9)
    temp_max = min(1000, gas_temperature.max() * 1.1)
    
    # Use a reasonable bin count to see structure
    bins = np.linspace(temp_min, temp_max, 100)
    
    # Plot count-weighted histogram
    ax1 = plt.subplot(1, 2, 1)
    counts, bins, _ = plt.hist(gas_temperature, bins=bins, alpha=0.7, 
                               label='Gas Particle Count', color='blue')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Number of Particles')
    plt.title(f'Temperature Distribution (z={redshift:.2f})')
    
    # Add vertical lines at 50K and 100K for reference
    plt.axvline(x=50, color='black', linestyle='--', alpha=0.7, label='50K')
    plt.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100K')
    plt.axvline(x=200, color='green', linestyle='--', alpha=0.7, label='200K')
    plt.legend()
    
    # Annotate peaks
    peak_indices = np.where((counts[1:-1] > counts[:-2]) & (counts[1:-1] > counts[2:]))[0] + 1
    for idx in peak_indices:
        if counts[idx] > np.max(counts) * 0.1:  # Only label significant peaks
            peak_temp = (bins[idx] + bins[idx+1]) / 2
            plt.annotate(f'{peak_temp:.1f}K', 
                        xy=(peak_temp, counts[idx]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot mass-weighted histogram
    ax2 = plt.subplot(1, 2, 2)
    plt.hist(gas_temperature, bins=bins, weights=gas_masses, alpha=0.7,
             label='Mass-Weighted', color='green')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Mass [code units]')
    plt.title('Mass-Weighted Temperature Distribution')
    
    # Add vertical lines at 50K and 100K for reference
    plt.axvline(x=50, color='black', linestyle='--', alpha=0.7, label='50K')
    plt.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100K')
    plt.axvline(x=200, color='green', linestyle='--', alpha=0.7, label='200K')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{output_prefix}_linear_hist.png', dpi=300)
    print(f"Linear histogram saved as plots/{output_prefix}_linear_hist.png")
    
    # Now create logarithmic histogram for a broader view
    plt.figure(figsize=(12, 6))
    
    # Set up temperature bins on a log scale
    log_bins = np.logspace(np.log10(max(1, gas_temperature.min() * 0.9)), 
                          np.log10(gas_temperature.max() * 1.1), 100)
    
    # Plot count-weighted log histogram
    ax1 = plt.subplot(1, 2, 1)
    counts, bins, _ = plt.hist(gas_temperature, bins=log_bins, alpha=0.7, 
                              label='Gas Particle Count', color='blue')
    plt.xscale('log')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Number of Particles')
    plt.title(f'Log Temperature Distribution (z={redshift:.2f})')
    
    # Add vertical lines at key temperatures
    plt.axvline(x=50, color='black', linestyle='--', alpha=0.7, label='50K')
    plt.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100K')
    plt.axvline(x=1000, color='green', linestyle='--', alpha=0.7, label='1000K')
    plt.axvline(x=10000, color='purple', linestyle='--', alpha=0.7, label='10000K')
    plt.legend()
    
    # Plot mass-weighted log histogram
    ax2 = plt.subplot(1, 2, 2)
    plt.hist(gas_temperature, bins=log_bins, weights=gas_masses, alpha=0.7,
             label='Mass-Weighted', color='green')
    plt.xscale('log')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Mass [code units]')
    plt.title('Mass-Weighted Log Temperature Distribution')
    
    # Add vertical lines at key temperatures
    plt.axvline(x=50, color='black', linestyle='--', alpha=0.7, label='50K')
    plt.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100K')
    plt.axvline(x=1000, color='green', linestyle='--', alpha=0.7, label='1000K')
    plt.axvline(x=10000, color='purple', linestyle='--', alpha=0.7, label='10000K')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{output_prefix}_log_hist.png', dpi=300)
    print(f"Log histogram saved as plots/{output_prefix}_log_hist.png")
    
    # Create a more detailed view of the interesting range
    plt.figure(figsize=(10, 8))
    
    # Focus on the 10K-500K range with finer bins
    detail_bins = np.linspace(10, 500, 200)
    
    plt.hist(gas_temperature, bins=detail_bins, alpha=0.7, 
             label='Gas Particle Count', color='blue')
    
    # Add a zoomed inset around the 50-150K range
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    
    # Create the zoomed inset
    axins = zoomed_inset_axes(plt.gca(), zoom=3, loc='upper right')
    axins.hist(gas_temperature, bins=np.linspace(40, 150, 100), alpha=0.7, color='blue')
    axins.axvline(x=50, color='black', linestyle='--', alpha=0.7)
    axins.axvline(x=100, color='red', linestyle='--', alpha=0.7)
    
    # Set the limits for the inset
    axins.set_xlim(40, 150)
    y_max = plt.hist(gas_temperature, bins=np.linspace(40, 150, 100))[0].max() * 1.1
    axins.set_ylim(0, y_max)
    
    # Hide tick labels in inset
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    
    # Connect inset to the main plot
    mark_inset(plt.gca(), axins, loc1=1, loc2=3, fc="none", ec="0.5")
    
    # Main plot formatting
    plt.xlabel('Temperature [K]')
    plt.ylabel('Number of Particles')
    plt.title(f'Detailed Temperature Distribution (z={redshift:.2f})')
    
    # Add vertical lines at key temperatures
    plt.axvline(x=50, color='black', linestyle='--', alpha=0.7, label='50K')
    plt.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100K')
    plt.axvline(x=200, color='green', linestyle='--', alpha=0.7, label='200K')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{output_prefix}_detailed_hist.png', dpi=300)
    print(f"Detailed histogram saved as plots/{output_prefix}_detailed_hist.png")
    
    # Create temperature vs. density colored by count density
    plt.figure(figsize=(10, 8))
    
    h, xedges, yedges = np.histogram2d(
        np.log10(gas_density),
        np.log10(gas_temperature),
        bins=(100, 100)
    )
    
    # Transpose h because imshow plots it differently
    h = h.T
    
    # Create a 2D histogram using pcolormesh for better control
    plt.pcolormesh(
        xedges, yedges, 
        h,
        cmap='viridis',
        shading='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Number of Particles')
    
    # Add horizontal lines at key temperatures
    plt.axhline(y=np.log10(50), color='black', linestyle='--', alpha=0.7, label='50K')
    plt.axhline(y=np.log10(100), color='red', linestyle='--', alpha=0.7, label='100K')
    plt.axhline(y=np.log10(200), color='green', linestyle='--', alpha=0.7, label='200K')
    
    # Set labels and title
    plt.xlabel(r'log$_{10}(\rho)$ [code units]')
    plt.ylabel(r'log$_{10}(T)$ [K]')
    plt.title(f'Temperature vs. Density (z={redshift:.2f})')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'plots/{output_prefix}_temp_vs_density.png', dpi=300)
    print(f"Temperature vs. density plot saved as plots/{output_prefix}_temp_vs_density.png")
    
    # Calculate and print statistics
    print("\nTemperature Statistics:")
    print(f"Mean temperature: {np.mean(gas_temperature):.2f} K")
    print(f"Median temperature: {np.median(gas_temperature):.2f} K")
    
    # Calculate percentages in different temperature ranges
    t_ranges = [
        (0, 50),
        (50, 75),
        (75, 125),
        (125, 200),
        (200, 1000),
        (1000, 10000),
        (10000, np.inf)
    ]
    
    print("\nTemperature Distribution:")
    for t_min, t_max in t_ranges:
        if t_max < np.inf:
            mask = (gas_temperature >= t_min) & (gas_temperature < t_max)
            range_label = f"{t_min}-{t_max}K"
        else:
            mask = (gas_temperature >= t_min)
            range_label = f">{t_min}K"
            
        count_percent = np.sum(mask) / len(gas_temperature) * 100
        mass_percent = np.sum(gas_masses[mask]) / np.sum(gas_masses) * 100
        
        print(f"{range_label}: {count_percent:.1f}% by count, {mass_percent:.1f}% by mass")
    
    return

def main():
    parser = argparse.ArgumentParser(description='Analyze temperature distribution in Gadget snapshots')
    parser.add_argument('snapshot', help='Path to Gadget snapshot file')
    parser.add_argument('--output', '-o', default='temp_hist', help='Output prefix for generated files')
    parser.add_argument('--density-min', type=float, help='Minimum density to include [code units]')
    parser.add_argument('--density-max', type=float, help='Maximum density to include [code units]')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.snapshot):
        print(f"Error: Snapshot file {args.snapshot} not found")
        return
    
    analyze_temperature(args.snapshot, args.output, args.density_min, args.density_max)

if __name__ == "__main__":
    main()