#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import os
import argparse
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from scipy.stats import binned_statistic_2d

def read_data_from_snapshot(filename):
    """Read relevant data from a Gadget HDF5 snapshot."""
    with h5py.File(filename, 'r') as f:
        # Get header information
        header = f['Header'].attrs
        redshift = header['Redshift']
        boxsize = header['BoxSize']
        time = header['Time']
        h = header['HubbleParam']
        
        # Get particle data (gas and stars)
        # Gas particles - type 0
        gas_masses = f['PartType0/Masses'][:]
        gas_sfr = None
        if 'StarFormationRate' in f['PartType0']:
            gas_sfr = f['PartType0/StarFormationRate'][:]
        
        # Star particles - type 4
        star_masses = None
        star_ages = None
        if 'PartType4' in f:
            if len(f['PartType4/Masses']) > 0:
                star_masses = f['PartType4/Masses'][:]
                # In Gadget, star ages are often stored as formation time
                if 'StellarFormationTime' in f['PartType4']:
                    star_formation_time = f['PartType4/StellarFormationTime'][:]
                    # Convert formation time to age
                    star_ages = time - star_formation_time
    
    return {
        'redshift': redshift,
        'boxsize': boxsize,
        'time': time,
        'h': h,
        'gas_masses': gas_masses,
        'gas_sfr': gas_sfr,
        'star_masses': star_masses,
        'star_ages': star_ages
    }

def calculate_sfr_density(snapshot_data):
    """Calculate star formation rate density in Msun/yr/Mpc^3 (comoving)."""
    gas_sfr = snapshot_data['gas_sfr']
    
    if gas_sfr is None or len(gas_sfr) == 0:
        return 0.0
    
    # Convert to Msun/yr
    # Assuming gas_sfr is in internal code units (Msun/h / (unit_time/h))
    h = snapshot_data['h']
    box_volume = snapshot_data['boxsize']**3  # Mpc^3/h^3
    
    # Sum all SFRs and divide by box volume
    total_sfr = np.sum(gas_sfr)  # Msun/yr/h
    
    # Convert to comoving units
    sfr_density = total_sfr / box_volume  # Msun/yr/h / (Mpc/h)^3 = Msun/yr/Mpc^3
    
    return sfr_density

def calculate_stellar_mass_density(snapshot_data):
    """Calculate stellar mass density in Msun/Mpc^3 (comoving)."""
    star_masses = snapshot_data['star_masses']
    
    if star_masses is None or len(star_masses) == 0:
        return 0.0
    
    h = snapshot_data['h']
    box_volume = snapshot_data['boxsize']**3  # Mpc^3/h^3
    
    # Sum all stellar masses and divide by box volume
    total_stellar_mass = np.sum(star_masses)  # Msun/h
    
    # Convert to comoving units
    stellar_mass_density = total_stellar_mass / box_volume  # Msun/h / (Mpc/h)^3 = Msun/Mpc^3
    
    return stellar_mass_density

def process_snapshots(snapshot_dir):
    """Process all snapshots in the given directory."""
    # Find all snapshot files
    snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.hdf5")))
    
    if not snapshot_files:
        print(f"No snapshot files found in {snapshot_dir}")
        return None
    
    results = []
    
    for filename in snapshot_files:
        try:
            print(f"Processing {filename}...")
            snapshot_data = read_data_from_snapshot(filename)
            
            # Calculate statistics
            sfr_density = calculate_sfr_density(snapshot_data)
            stellar_mass_density = calculate_stellar_mass_density(snapshot_data)
            
            results.append({
                'filename': os.path.basename(filename),
                'redshift': snapshot_data['redshift'],
                'time': snapshot_data['time'],
                'sfr_density': sfr_density,
                'stellar_mass_density': stellar_mass_density
            })
            
            print(f"  z={snapshot_data['redshift']:.2f}, SFR={sfr_density:.2e} Msun/yr/Mpc^3, M*={stellar_mass_density:.2e} Msun/Mpc^3")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Sort by redshift
    results.sort(key=lambda x: x['redshift'], reverse=True)
    
    return results

def plot_lilly_madau(results1, results2=None, output_file="lilly_madau_plot.png"):
    """Create a Lilly-Madau plot."""
    plt.figure(figsize=(12, 10))
    
    # Set up subplots
    ax1 = plt.subplot(211)  # SFR density
    ax2 = plt.subplot(212)  # Stellar mass density
    
    # Observational data (simplified example points)
    # In a real application, you would load actual observational data
    obs_z = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    obs_sfr = np.array([0.01, 0.05, 0.1, 0.15, 0.1, 0.06, 0.03, 0.01, 0.005])
    obs_sfr_err = obs_sfr * 0.3  # 30% error bars
    
    obs_stellar = np.array([1e8, 8e7, 5e7, 3e7, 1e7, 5e6, 2e6, 8e5, 3e5])
    obs_stellar_err = obs_stellar * 0.3
    
    # Plot first simulation data
    redshift1 = [item['redshift'] for item in results1]
    sfr_density1 = [item['sfr_density'] for item in results1]
    stellar_density1 = [item['stellar_mass_density'] for item in results1]
    
    ax1.plot(redshift1, sfr_density1, 'o-', label='Simulation 1', color='blue')
    ax2.plot(redshift1, stellar_density1, 'o-', label='Simulation 1', color='blue')
    
    # Plot second simulation data if provided
    if results2 is not None:
        redshift2 = [item['redshift'] for item in results2]
        sfr_density2 = [item['sfr_density'] for item in results2]
        stellar_density2 = [item['stellar_mass_density'] for item in results2]
        
        ax1.plot(redshift2, sfr_density2, 'o-', label='Simulation 2', color='red')
        ax2.plot(redshift2, stellar_density2, 'o-', label='Simulation 2', color='red')
    
    # Plot observational data (example)
    ax1.errorbar(obs_z, obs_sfr, yerr=obs_sfr_err, fmt='s', color='green', label='Observations (example)')
    ax2.errorbar(obs_z, obs_stellar, yerr=obs_stellar_err, fmt='s', color='green', label='Observations (example)')
    
    # Configure plot 1 (SFR density)
    ax1.set_xlabel('Redshift (z)')
    ax1.set_ylabel('SFR Density [M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$]')
    ax1.set_yscale('log')
    ax1.set_title('Cosmic Star Formation Rate Density vs. Redshift')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    # Configure plot 2 (Stellar mass density)
    ax2.set_xlabel('Redshift (z)')
    ax2.set_ylabel('Stellar Mass Density [M$_\\odot$ Mpc$^{-3}$]')
    ax2.set_yscale('log')
    ax2.set_title('Cosmic Stellar Mass Density vs. Redshift')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create Lilly-Madau plot from Gadget snapshots')
    parser.add_argument('--dir1', required=True, help='Directory containing first set of snapshots')
    parser.add_argument('--dir2', help='Directory containing second set of snapshots (optional)')
    parser.add_argument('--output', default='lilly_madau_plot.png', help='Output filename')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Process first set of snapshots
    print(f"Processing snapshots in {args.dir1}...")
    results1 = process_snapshots(args.dir1)
    
    # Process second set of snapshots if provided
    results2 = None
    if args.dir2:
        print(f"Processing snapshots in {args.dir2}...")
        results2 = process_snapshots(args.dir2)
    
    # Create the plot
    if results1:
        plot_lilly_madau(results1, results2, args.output)
    else:
        print("No valid results to plot.")

if __name__ == "__main__":
    main()