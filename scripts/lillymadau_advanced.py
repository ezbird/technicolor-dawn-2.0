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
from datetime import datetime

# Observational data for comparison (Madau & Dickinson 2014)
MADAU_DICKINSON_2014 = {
    'redshift': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.7, 2.0, 2.5, 3.0, 3.8, 4.9, 6.0, 7.0, 8.0],
    'sfrd': [0.015, 0.024, 0.035, 0.043, 0.052, 0.065, 0.07, 0.073, 0.098, 0.108, 0.097, 0.059, 0.034, 0.02, 0.01, 0.005],
    'sfrd_err_up': [0.003, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.01, 0.02, 0.02, 0.02, 0.015, 0.009, 0.005, 0.003, 0.002],
    'sfrd_err_down': [0.003, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.01, 0.02, 0.02, 0.02, 0.015, 0.009, 0.005, 0.003, 0.002],
    'smdens': [8.6e7, 7.9e7, 7.2e7, 6.5e7, 5.8e7, 5.1e7, 4.5e7, 3.5e7, 2.8e7, 2.2e7, 1.7e7, 1.2e7, 6.8e6, 3.6e6, 1.9e6, 1.0e6],
    'smdens_err_up': [1.5e7, 1.4e7, 1.3e7, 1.1e7, 1.0e7, 0.9e7, 0.8e7, 0.6e7, 0.5e7, 0.4e7, 0.3e7, 0.25e7, 1.5e6, 1.0e6, 0.7e6, 0.4e6],
    'smdens_err_down': [1.5e7, 1.4e7, 1.3e7, 1.1e7, 1.0e7, 0.9e7, 0.8e7, 0.6e7, 0.5e7, 0.4e7, 0.3e7, 0.25e7, 1.5e6, 1.0e6, 0.7e6, 0.4e6]
}

def print_hdf5_structure(h5file, prefix=''):
    """Helper function to print the structure of an HDF5 file."""
    def _print_hdf5_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{prefix}{name}: {obj.shape} {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{prefix}{name}/")
    
    h5file.visititems(_print_hdf5_structure)

def read_data_from_snapshot(filename, verbose=False):
    """Read relevant data from a Gadget HDF5 snapshot."""
    try:
        with h5py.File(filename, 'r') as f:
            if verbose:
                print(f"\nHDF5 file structure for {filename}:")
                print_hdf5_structure(f)
                print("\nHeader attributes:")
                for key, value in f['Header'].attrs.items():
                    print(f"  {key}: {value}")
            
            # Get header information
            header = f['Header'].attrs
            redshift = header['Redshift']
            boxsize = header['BoxSize']
            time = header['Time']
            
            # Try different keys for HubbleParam
            h = 0.7  # default value
            if 'HubbleParam' in header:
                h = header['HubbleParam']
            elif 'h_val' in header:
                h = header['h_val']
            elif 'H0' in header:
                h = header['H0'] / 100.0  # Convert H0 to little h
            
            if verbose:
                print(f"Using HubbleParam = {h}")
            
            # Gas particles - type 0
            gas_masses = None
            gas_sfr = None
            if 'PartType0' in f and len(f['PartType0/Masses']) > 0:
                gas_masses = f['PartType0/Masses'][:]
                if 'StarFormationRate' in f['PartType0']:
                    gas_sfr = f['PartType0/StarFormationRate'][:]
                elif 'Rate' in f['PartType0']:
                    gas_sfr = f['PartType0/Rate'][:]
            
            # Star particles - type 4
            star_masses = None
            star_formation_time = None
            if 'PartType4' in f and 'Masses' in f['PartType4'] and len(f['PartType4/Masses']) > 0:
                star_masses = f['PartType4/Masses'][:]
                # In Gadget, star ages are often stored as formation time
                if 'StellarFormationTime' in f['PartType4']:
                    star_formation_time = f['PartType4/StellarFormationTime'][:]
                elif 'FormationTime' in f['PartType4']:
                    star_formation_time = f['PartType4/FormationTime'][:]
                    
            # Read unit information if available
            unit_mass = 1.0e10  # Default: 10^10 Msun/h
            unit_length = 1.0  # Default: 1 kpc/h
            unit_time = 1.0  # Default: depends on other units
            
            if 'Units' in f:
                if 'UnitMass_in_g' in f['Units'].attrs:
                    unit_mass = f['Units'].attrs['UnitMass_in_g'] / 1.989e33  # Convert g to Msun
                if 'UnitLength_in_cm' in f['Units'].attrs:
                    unit_length = f['Units'].attrs['UnitLength_in_cm'] / 3.086e21  # Convert cm to kpc
            
            if verbose:
                print(f"Units: Mass={unit_mass} Msun/h, Length={unit_length} kpc/h")
            
            # Get temperature data if available
            temperatures = None
            if 'PartType0' in f:
                if 'Temperature' in f['PartType0']:
                    temperatures = f['PartType0/Temperature'][:]
                elif 'InternalEnergy' in f['PartType0']:
                    # Will calculate temperature later
                    internal_energy = f['PartType0/InternalEnergy'][:]
                    temperatures = internal_energy  # Placeholder, will convert later
        
        return {
            'redshift': redshift,
            'boxsize': boxsize, 
            'time': time,
            'h': h,
            'gas_masses': gas_masses,
            'gas_sfr': gas_sfr,
            'star_masses': star_masses,
            'star_formation_time': star_formation_time,
            'temperatures': temperatures,
            'unit_mass': unit_mass,
            'unit_length': unit_length
        }
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def calculate_sfr_density(snapshot_data, verbose=False):
    """Calculate star formation rate density in Msun/yr/Mpc^3 (comoving)."""
    if snapshot_data is None or snapshot_data['gas_sfr'] is None:
        return 0.0
    
    gas_sfr = snapshot_data['gas_sfr']
    h = snapshot_data['h']
    
    # Box volume in comoving (Mpc/h)^3
    box_volume = snapshot_data['boxsize']**3
    
    # Sum all SFRs and divide by box volume
    total_sfr = np.sum(gas_sfr)
    
    # Convert to physical units: Msun/yr/Mpc^3
    # SFRs in Gadget are typically in Msun/h per year
    sfr_density = total_sfr / box_volume
    
    if verbose:
        print(f"Total SFR: {total_sfr} Msun/yr, Box Volume: {box_volume} (Mpc/h)^3")
        print(f"SFR Density: {sfr_density} (Msun/yr)/(Mpc/h)^3")
    
    # Convert h-factors correctly - SFR density should be in h=1 units
    # In most simulations: SFR ~ Msun/h / yr and volume ~ (Mpc/h)^3
    # So SFR density ~ (Msun/h)/(Mpc/h)^3 / yr = Msun/yr/Mpc^3 * h^2
    sfr_density = sfr_density * h**2
    
    if verbose:
        print(f"SFR Density (h-corrected): {sfr_density} Msun/yr/Mpc^3")
    
    return sfr_density

def calculate_stellar_mass_density(snapshot_data, verbose=False):
    """Calculate stellar mass density in Msun/Mpc^3 (comoving)."""
    if snapshot_data is None or snapshot_data['star_masses'] is None:
        return 0.0
    
    star_masses = snapshot_data['star_masses']
    h = snapshot_data['h']
    
    # Box volume in comoving (Mpc/h)^3
    box_volume = snapshot_data['boxsize']**3
    
    # Sum all stellar masses and divide by box volume
    total_stellar_mass = np.sum(star_masses)
    
    # Convert to physical units: Msun/Mpc^3
    # Most Gadget simulations store masses as Msun/h
    stellar_mass_density = total_stellar_mass / box_volume
    
    if verbose:
        print(f"Total Stellar Mass: {total_stellar_mass} Msun/h, Box Volume: {box_volume} (Mpc/h)^3")
        print(f"Stellar Mass Density: {stellar_mass_density} (Msun/h)/(Mpc/h)^3")
    
    # Convert h-factors correctly - should be in h=1 units
    # In most simulations: mass ~ Msun/h and volume ~ (Mpc/h)^3
    # So mass density ~ (Msun/h)/(Mpc/h)^3 = Msun/Mpc^3 * h^2
    stellar_mass_density = stellar_mass_density * h**2
    
    if verbose:
        print(f"Stellar Mass Density (h-corrected): {stellar_mass_density} Msun/Mpc^3")
    
    return stellar_mass_density

def process_snapshots(snapshot_dir, hubble_time=13.8e9, verbose=False):
    """Process all snapshots in the given directory."""
    # Find all snapshot files
    snapshot_pattern = os.path.join(snapshot_dir, "snap*.hdf5")
    snapshot_files = sorted(glob.glob(snapshot_pattern))
    
    if not snapshot_files:
        print(f"No snapshot files found matching {snapshot_pattern}")
        return None
    
    print(f"Found {len(snapshot_files)} snapshot files")
    
    results = []
    previous_data = None
    
    for filename in snapshot_files:
        try:
            print(f"Processing {filename}...")
            snapshot_data = read_data_from_snapshot(filename, verbose)
            
            if snapshot_data is None:
                print(f"Skipping {filename} due to read error")
                continue
            
            # Calculate statistics
            sfr_density = calculate_sfr_density(snapshot_data, verbose)
            stellar_mass_density = calculate_stellar_mass_density(snapshot_data, verbose)
            
            # Calculate temperature statistics
            temp_stats = {}
            if snapshot_data['temperatures'] is not None:
                temps = snapshot_data['temperatures']
                temp_stats = {
                    'min': np.min(temps),
                    'max': np.max(temps),
                    'mean': np.mean(temps),
                    'median': np.median(temps)
                }
            
            results.append({
                'filename': os.path.basename(filename),
                'redshift': snapshot_data['redshift'],
                'time': snapshot_data['time'],
                'h': snapshot_data['h'],
                'sfr_density': sfr_density,
                'stellar_mass_density': stellar_mass_density,
                'temp_stats': temp_stats
            })
            
            print(f"  z={snapshot_data['redshift']:.2f}, SFR={sfr_density:.2e} Msun/yr/Mpc^3, " + 
                  f"M*={stellar_mass_density:.2e} Msun/Mpc^3")
            
            previous_data = snapshot_data
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Sort by redshift
    results.sort(key=lambda x: x['redshift'], reverse=True)
    
    return results

def madau_fit_function(z, a=0.01, b=2.6, c=3.2, d=6.2):
    """
    The Madau & Dickinson (2014) fitting function for SFRD.
    Returns SFRD in Msun/yr/Mpc^3.
    """
    return a * (1 + z)**b / (1 + ((1 + z)/c)**d)

def plot_lilly_madau(results1, results2=None, output_file="lilly_madau_plot.png", 
                     sim1_label="Simulation 1", sim2_label="Simulation 2"):
    """Create a Lilly-Madau plot."""
    plt.figure(figsize=(12, 10))
    
    # Set up subplots
    ax1 = plt.subplot(211)  # SFR density
    ax2 = plt.subplot(212)  # Stellar mass density
    
    # Plot observational data
    z_obs = np.array(MADAU_DICKINSON_2014['redshift'])
    sfrd_obs = np.array(MADAU_DICKINSON_2014['sfrd'])
    sfrd_errp = np.array(MADAU_DICKINSON_2014['sfrd_err_up'])
    sfrd_errm = np.array(MADAU_DICKINSON_2014['sfrd_err_down'])
    
    smdens_obs = np.array(MADAU_DICKINSON_2014['smdens'])
    smdens_errp = np.array(MADAU_DICKINSON_2014['smdens_err_up'])
    smdens_errm = np.array(MADAU_DICKINSON_2014['smdens_err_down'])
    
    # Plot Madau & Dickinson best-fit curve
    z_fine = np.linspace(0, 10, 100)
    sfrd_fit = madau_fit_function(z_fine)
    
    # Plot first simulation data
    if results1:
        redshift1 = [item['redshift'] for item in results1]
        sfr_density1 = [item['sfr_density'] for item in results1]
        stellar_density1 = [item['stellar_mass_density'] for item in results1]
        
        ax1.plot(redshift1, sfr_density1, 'o-', label=sim1_label, color='blue', markersize=4)
        ax2.plot(redshift1, stellar_density1, 'o-', label=sim1_label, color='blue', markersize=4)
    
    # Plot second simulation data if provided
    if results2:
        redshift2 = [item['redshift'] for item in results2]
        sfr_density2 = [item['sfr_density'] for item in results2]
        stellar_density2 = [item['stellar_mass_density'] for item in results2]
        
        ax1.plot(redshift2, sfr_density2, 'o-', label=sim2_label, color='red', markersize=4)
        ax2.plot(redshift2, stellar_density2, 'o-', label=sim2_label, color='red', markersize=4)
    
    # Plot observational data
    ax1.errorbar(z_obs, sfrd_obs, yerr=[sfrd_errm, sfrd_errp], fmt='s', color='green', 
                 label='Madau & Dickinson (2014)', capsize=3, markersize=4)
    ax1.plot(z_fine, sfrd_fit, '--', color='darkgreen', label='Madau & Dickinson Fit')
    
    ax2.errorbar(z_obs, smdens_obs, yerr=[smdens_errm, smdens_errp], fmt='s', color='green', 
                 label='Madau & Dickinson (2014)', capsize=3, markersize=4)
    
    # Configure plot 1 (SFR density)
    ax1.set_xlabel('Redshift (z)')
    ax1.set_ylabel('SFR Density [M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$]')
    ax1.set_yscale('log')
    ax1.set_ylim(0.001, 0.5)
    ax1.set_xlim(-0.1, 10)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize=9)
    ax1.set_title('Cosmic Star Formation Rate Density vs. Redshift')
    
    # Configure plot 2 (Stellar mass density)
    ax2.set_xlabel('Redshift (z)')
    ax2.set_ylabel('Stellar Mass Density [M$_\\odot$ Mpc$^{-3}$]')
    ax2.set_yscale('log')
    ax2.set_xlim(-0.1, 10)
    ax2.set_ylim(1e5, 2e8)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(fontsize=9)
    ax2.set_title('Cosmic Stellar Mass Density vs. Redshift')
    
    # Add timestamp
    plt.figtext(0.01, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=8, ha='left')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Save data to CSV for further analysis
    save_results_to_csv(results1, results2, output_file.replace('.png', '.csv'))
    
    return plt

def save_results_to_csv(results1, results2=None, csv_file="lilly_madau_data.csv"):
    """Save the processed results to a CSV file for further analysis."""
    import csv
    
    # Prepare the header and data
    header = ['Simulation', 'Filename', 'Redshift', 'Time', 'h', 'SFR_Density', 'Stellar_Mass_Density']
    rows = []
    
    # Add results from simulation 1
    if results1:
        for item in results1:
            rows.append([
                'Sim1',
                item['filename'],
                item['redshift'],
                item['time'],
                item['h'],
                item['sfr_density'],
                item['stellar_mass_density']
            ])
    
    # Add results from simulation 2 if available
    if results2:
        for item in results2:
            rows.append([
                'Sim2',
                item['filename'],
                item['redshift'],
                item['time'],
                item['h'],
                item['sfr_density'],
                item['stellar_mass_density']
            ])
    
    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Data saved to {csv_file}")

def create_temperature_density_plot(filename, output_prefix=None, verbose=False):
    """Create a temperature vs. density phase plot for a single snapshot."""
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(filename))[0]
    
    try:
        # Read the snapshot data
        snapshot_data = read_data_from_snapshot(filename, verbose)
        if snapshot_data is None:
            print(f"Could not read data from {filename}")
            return False
        
        with h5py.File(filename, 'r') as f:
            # Get gas properties
            if 'PartType0' not in f:
                print(f"No gas particles found in {filename}")
                return False
            
            # Get gas density
            if 'Density' in f['PartType0']:
                density = f['PartType0/Density'][:]
            else:
                print(f"No density information found in {filename}")
                return False
            
            # Get or calculate temperature
            temperatures = None
            if 'Temperature' in f['PartType0']:
                temperatures = f['PartType0/Temperature'][:]
            elif 'InternalEnergy' in f['PartType0']:
                # Calculate temperature from internal energy
                u = f['PartType0/InternalEnergy'][:]
                
                # Approximate temperature conversion
                mu = 0.6  # mean molecular weight (approx. for ionized gas)
                gamma = 5/3  # adiabatic index
                m_p = 1.67e-24  # proton mass in g
                k_B = 1.38e-16  # Boltzmann constant in erg/K
                
                # T = 2/3 * u * μ * m_p / k_B  (for γ = 5/3)
                temperatures = (gamma - 1) * u * mu * m_p / k_B
            else:
                print(f"No temperature or internal energy information found in {filename}")
                return False
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(
            np.log10(density), 
            np.log10(temperatures),
            bins=100,
            range=[[-32, -24], [1, 8]]  # Adjust these ranges as needed
        )
        
        # Create a log-normalized pcolormesh
        X, Y = np.meshgrid(xedges, yedges)
        plt.pcolormesh(X, Y, h.T, norm=LogNorm(), cmap='viridis')
        plt.colorbar(label='Number of Gas Particles')
        
        plt.xlabel(r'log$_{10}$(Density [g/cm$^3$])')
        plt.ylabel(r'log$_{10}$(Temperature [K])')
        plt.title(f'Temperature-Density Phase Diagram (z={snapshot_data["redshift"]:.2f})')
        
        plt.grid(alpha=0.3)
        
        # Add annotation with temperature statistics
        median_temp = np.median(temperatures)
        mean_temp = np.mean(temperatures)
        min_temp = np.min(temperatures)
        max_temp = np.max(temperatures)
        
        info_text = (
            f"Statistics:\n"
            f"Min Temp: {min_temp:.1f} K\n"
            f"Max Temp: {max_temp:.1f} K\n"
            f"Median: {median_temp:.1f} K\n"
            f"Mean: {mean_temp:.1f} K"
        )
        
        plt.annotate(info_text, xy=(0.02, 0.96), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                     va='top', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        output_file = f"{output_prefix}_phase_diagram.png"
        plt.savefig(output_file, dpi=300)
        print(f"Phase diagram saved to {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Error creating phase diagram for {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create Lilly-Madau plot from Gadget snapshots')
    parser.add_argument('--dir1', help='Directory containing first set of snapshots')
    parser.add_argument('--dir2', help='Directory containing second set of snapshots (optional)')
    parser.add_argument('--output', default='lilly_madau_plot.png', help='Output filename')
    parser.add_argument('--sim1-label', default='Simulation 1', help='Label for first simulation')
    parser.add_argument('--sim2-label', default='Simulation 2', help='Label for second simulation')
    parser.add_argument('--phase-diagram', help='Create phase diagram for specific snapshot')
    parser.add_argument('--hubble-time', type=float, default=13.8e9, help='Hubble time in years (default: 13.8 Gyr)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--inspect', help='Inspect the structure of a specific snapshot file')
    return parser.parse_args()

def inspect_snapshot(filename):
    """Detailed inspection of a snapshot file structure"""
    try:
        with h5py.File(filename, 'r') as f:
            print(f"\n=== Inspecting HDF5 file: {filename} ===\n")
            
            print("File structure:")
            print_hdf5_structure(f)
            
            print("\nHeader attributes:")
            for key, value in f['Header'].attrs.items():
                print(f"  {key}: {value}")
            
            # Check for units information
            if 'Units' in f:
                print("\nUnits information:")
                for key, value in f['Units'].attrs.items():
                    print(f"  {key}: {value}")
            
            # Check particle types and their datasets
            for ptype in ['PartType0', 'PartType1', 'PartType2', 'PartType3', 'PartType4', 'PartType5']:
                if ptype in f:
                    print(f"\n{ptype} datasets:")
                    for dataset_name in f[ptype]:
                        shape = f[ptype][dataset_name].shape
                        dtype = f[ptype][dataset_name].dtype
                        print(f"  {dataset_name}: shape={shape}, dtype={dtype}")
            
            return True
    except Exception as e:
        print(f"Error inspecting {filename}: {e}")
        return False

def main():
    args = parse_arguments()
    
    # Inspect mode
    if args.inspect:
        if os.path.exists(args.inspect):
            inspect_snapshot(args.inspect)
        else:
            print(f"Snapshot file {args.inspect} not found")
        return
    
    # Phase diagram mode
    if args.phase_diagram:
        if os.path.exists(args.phase_diagram):
            create_temperature_density_plot(args.phase_diagram, verbose=args.verbose)
        else:
            print(f"Snapshot file {args.phase_diagram} not found")
        return
    
    # Process first set of snapshots
    if not args.dir1:
        print("Please specify at least one snapshot directory with --dir1")
        return
    
    print(f"Processing snapshots in {args.dir1}...")
    results1 = process_snapshots(args.dir1, args.hubble_time, args.verbose)
    
    # Process second set of snapshots if provided
    results2 = None
    if args.dir2:
        print(f"Processing snapshots in {args.dir2}...")
        results2 = process_snapshots(args.dir2, args.hubble_time, args.verbose)
    
    # Create the plot
    if results1:
        plot_lilly_madau(results1, results2, args.output, args.sim1_label, args.sim2_label)
    else:
        print("No valid results to plot.")

if __name__ == "__main__":
    main()