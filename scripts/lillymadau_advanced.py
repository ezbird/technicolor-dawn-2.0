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

# Observational data for comparison
# These are approximate values from literature
MADAU_DICKINSON_2014 = {
    'redshift': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.7, 2.0, 2.5, 3.0, 3.8, 4.9, 6.0, 7.0, 8.0],
    'sfrd': [0.015, 0.024, 0.035, 0.043, 0.052, 0.065, 0.07, 0.073, 0.098, 0.108, 0.097, 0.059, 0.034, 0.02, 0.01, 0.005],
    'sfrd_err_up': [0.003, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.01, 0.02, 0.02, 0.02, 0.015, 0.009, 0.005, 0.003, 0.002],
    'sfrd_err_down': [0.003, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.01, 0.02, 0.02, 0.02, 0.015, 0.009, 0.005, 0.003, 0.002],
    'smdens': [8.6e7, 7.9e7, 7.2e7, 6.5e7, 5.8e7, 5.1e7, 4.5e7, 3.5e7, 2.8e7, 2.2e7, 1.7e7, 1.2e7, 6.8e6, 3.6e6, 1.9e6, 1.0e6],
    'smdens_err_up': [1.5e7, 1.4e7, 1.3e7, 1.1e7, 1.0e7, 0.9e7, 0.8e7, 0.6e7, 0.5e7, 0.4e7, 0.3e7, 0.25e7, 1.5e6, 1.0e6, 0.7e6, 0.4e6],
    'smdens_err_down': [1.5e7, 1.4e7, 1.3e7, 1.1e7, 1.0e7, 0.9e7, 0.8e7, 0.6e7, 0.5e7, 0.4e7, 0.3e7, 0.25e7, 1.5e6, 1.0e6, 0.7e6, 0.4e6]
}

def read_data_from_snapshot(filename):
    """Read relevant data from a Gadget HDF5 snapshot."""
    try:
        with h5py.File(filename, 'r') as f:
            # Get header information
            header = f['Header'].attrs
            redshift = header['Redshift']
            boxsize = header['BoxSize']
            time = header['Time']
            h = header['HubbleParam']
            
            # Gas particles - type 0
            gas_masses = None
            gas_sfr = None
            if 'PartType0' in f and len(f['PartType0/Masses']) > 0:
                gas_masses = f['PartType0/Masses'][:]
                if 'StarFormationRate' in f['PartType0']:
                    gas_sfr = f['PartType0/StarFormationRate'][:]
            
            # Star particles - type 4
            star_masses = None
            star_formation_time = None
            if 'PartType4' in f and 'Masses' in f['PartType4'] and len(f['PartType4/Masses']) > 0:
                star_masses = f['PartType4/Masses'][:]
                # In Gadget, star ages are often stored as formation time
                if 'StellarFormationTime' in f['PartType4']:
                    star_formation_time = f['PartType4/StellarFormationTime'][:]
        
        return {
            'redshift': redshift,
            'boxsize': boxsize, 
            'time': time,
            'h': h,
            'gas_masses': gas_masses,
            'gas_sfr': gas_sfr,
            'star_masses': star_masses,
            'star_formation_time': star_formation_time
        }
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def calculate_sfr_density(snapshot_data):
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
    # Assuming SFR is in solar masses/yr
    sfr_density = total_sfr / box_volume
    
    return sfr_density

def calculate_sfr_density_from_new_stars(snapshot_data, previous_data, hubble_time):
    """
    Alternative method: Calculate SFR density from newly formed stars between snapshots.
    This can be more accurate than instantaneous SFR if star formation is bursty.
    """
    if snapshot_data is None or previous_data is None:
        return 0.0
    
    # Get star masses for this snapshot
    if snapshot_data['star_masses'] is None:
        return 0.0
    
    # Get star masses for previous snapshot
    prev_star_masses_total = 0
    if previous_data['star_masses'] is not None:
        prev_star_masses_total = np.sum(previous_data['star_masses'])
    
    # Calculate total stellar mass in this snapshot
    curr_star_masses_total = np.sum(snapshot_data['star_masses'])
    
    # Calculate new stellar mass formed between snapshots
    new_stellar_mass = curr_star_masses_total - prev_star_masses_total
    
    if new_stellar_mass <= 0:
        return 0.0
    
    # Time difference between snapshots (in Hubble time units)
    dt = snapshot_data['time'] - previous_data['time']
    
    # Convert to physical time in years
    dt_years = dt * hubble_time
    
    # Box volume in comoving (Mpc/h)^3
    box_volume = snapshot_data['boxsize']**3
    
    # Calculate SFR density: Msun/yr/Mpc^3
    sfr_density = new_stellar_mass / dt_years / box_volume
    
    return sfr_density

def calculate_stellar_mass_density(snapshot_data):
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
    stellar_mass_density = total_stellar_mass / box_volume
    
    return stellar_mass_density

def process_snapshots(snapshot_dir, hubble_time=13.8e9):
    """Process all snapshots in the given directory."""
    # Find all snapshot files
    snapshot_pattern = os.path.join(snapshot_dir, "snapshot_*.hdf5")
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
            snapshot_data = read_data_from_snapshot(filename)
            
            if snapshot_data is None:
                print(f"Skipping {filename} due to read error")
                continue
            
            # Calculate statistics
            sfr_density = calculate_sfr_density(snapshot_data)
            
            # Calculate SFR from new stars (if possible)
            sfr_density_new_stars = 0
            if previous_data is not None:
                sfr_density_new_stars = calculate_sfr_density_from_new_stars(
                    snapshot_data, previous_data, hubble_time)
            
            stellar_mass_density = calculate_stellar_mass_density(snapshot_data)
            
            results.append({
                'filename': os.path.basename(filename),
                'redshift': snapshot_data['redshift'],
                'time': snapshot_data['time'],
                'sfr_density': sfr_density,
                'sfr_density_new_stars': sfr_density_new_stars,
                'stellar_mass_density': stellar_mass_density
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

def plot_lilly_madau(results1, results2=None, output_file="lilly_madau_plot.png", sim1_label="Simulation 1", sim2_label="Simulation 2"):
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
        sfr_density_new1 = [item['sfr_density_new_stars'] for item in results1]
        stellar_density1 = [item['stellar_mass_density'] for item in results1]
        
        ax1.plot(redshift1, sfr_density1, 'o-', label=f'{sim1_label} (Instant. SFR)', color='blue', markersize=4)
        if any(sfr_density_new1):
            ax1.plot(redshift1, sfr_density_new1, 's--', label=f'{sim1_label} (New Stars)', color='lightblue', markersize=4)
        ax2.plot(redshift1, stellar_density1, 'o-', label=sim1_label, color='blue', markersize=4)
    
    # Plot second simulation data if provided
    if results2:
        redshift2 = [item['redshift'] for item in results2]
        sfr_density2 = [item['sfr_density'] for item in results2]
        sfr_density_new2 = [item['sfr_density_new_stars'] for item in results2]
        stellar_density2 = [item['stellar_mass_density'] for item in results2]
        
        ax1.plot(redshift2, sfr_density2, 'o-', label=f'{sim2_label} (Instant. SFR)', color='red', markersize=4)
        if any(sfr_density_new2):
            ax1.plot(redshift2, sfr_density_new2, 's--', label=f'{sim2_label} (New Stars)', color='lightcoral', markersize=4)
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
    header = ['Simulation', 'Filename', 'Redshift', 'Time', 'SFR_Density', 'SFR_Density_NewStars', 'Stellar_Mass_Density']
    rows = []
    
    # Add results from simulation 1
    if results1:
        for item in results1:
            rows.append([
                'Sim1',
                item['filename'],
                item['redshift'],
                item['time'],
                item['sfr_density'],
                item['sfr_density_new_stars'],
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
                item['sfr_density'],
                item['sfr_density_new_stars'],
                item['stellar_mass_density']
            ])
    
    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Data saved to {csv_file}")

def create_temperature_density_plot(filename, output_prefix=None):
    """Create a temperature vs. density phase plot for a single snapshot."""
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(filename))[0]
    
    try:
        with h5py.File(filename, 'r') as f:
            # Get header information
            header = f['Header'].attrs
            redshift = header['Redshift']
            time = header['Time']
            
            # Gas properties
            if 'PartType0' not in f or 'InternalEnergy' not in f['PartType0']:
                print(f"No gas particles or internal energy found in {filename}")
                return False
            
            # Get gas density and temperature
            density = f['PartType0/Density'][:]
            
            # Calculate temperature from internal energy
            # Note: This conversion depends on your simulation's specific parameters
            u = f['PartType0/InternalEnergy'][:]
            
            # Approximate temperature conversion (assuming primordial gas)
            # T = 2/3 * u * μ * m_p / k_B
            # where μ is mean molecular weight (≈ 0.6 for ionized gas)
            mu = 0.6  # mean molecular weight
            gamma = 5/3  # adiabatic index
            m_p = 1.67e-24  # proton mass in g
            k_B = 1.38e-16  # Boltzmann constant in erg/K
            
            # Calculate temperature (K)
            temp = (gamma - 1) * u * mu * m_p / k_B
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Create 2D histogram
            bins = 100
            h, xedges, yedges, img = plt.hist2d(
                np.log10(density), 
                np.log10(temp),
                bins=bins,
                range=[[-32, -24], [1, 8]],  # Adjust these ranges as needed
                norm=LogNorm(),
                cmap='viridis'
            )
            
            plt.colorbar(label='Number of Gas Particles')
            
            plt.xlabel(r'log$_{10}$(Density [g/cm$^3$])')
            plt.ylabel(r'log$_{10}$(Temperature [K])')
            plt.title(f'Temperature-Density Phase Diagram (z={redshift:.2f}, a={time:.4f})')
            
            plt.grid(alpha=0.3)
            
            # Add annotation with temperature statistics
            median_temp = np.median(temp)
            mean_temp = np.mean(temp)
            min_temp = np.min(temp)
            max_temp = np.max(temp)
            
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
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create Lilly-Madau plot from Gadget snapshots')
    parser.add_argument('--dir1', required=True, help='Directory containing first set of snapshots')
    parser.add_argument('--dir2', help='Directory containing second set of snapshots (optional)')
    parser.add_argument('--output', default='lilly_madau_plot.png', help='Output filename')
    parser.add_argument('--sim1-label', default='Simulation 1', help='Label for first simulation')
    parser.add_argument('--sim2-label', default='Simulation 2', help='Label for second simulation')
    parser.add_argument('--phase-diagram', help='Create phase diagram for specific snapshot')
    parser.add_argument('--hubble-time', type=float, default=13.8e9, help='Hubble time in years (default: 13.8 Gyr)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # If phase diagram mode is selected
    if args.phase_diagram:
        if os.path.exists(args.phase_diagram):
            create_temperature_density_plot(args.phase_diagram)
        else:
            print(f"Snapshot file {args.phase_diagram} not found")
        return
    
    # Process first set of snapshots
    print(f"Processing snapshots in {args.dir1}...")
    results1 = process_snapshots(args.dir1, args.hubble_time)
    
    # Process second set of snapshots if provided
    results2 = None
    if args.dir2:
        print(f"Processing snapshots in {args.dir2}...")
        results2 = process_snapshots(args.dir2, args.hubble_time)
    
    # Create the plot
    if results1:
        plot_lilly_madau(results1, results2, args.output, args.sim1_label, args.sim2_label)
    else:
        print("No valid results to plot.")

if __name__ == "__main__":
    main()