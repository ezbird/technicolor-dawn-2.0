#!/usr/bin/env python3
"""
Lilly-Madau Plot Generator for Gadget Snapshots

This script analyzes Gadget snapshots to create Lilly-Madau plots for comparing
cosmic star formation rate density (SFRD) and stellar mass density across different simulations.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import os
import re
import argparse
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from datetime import datetime

# Observational data (Madau & Dickinson 2014)
MADAU_DICKINSON_2014 = {
    'redshift': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.7, 2.0, 2.5, 3.0, 3.8, 4.9, 6.0, 7.0, 8.0],
    'sfrd': [0.015, 0.024, 0.035, 0.043, 0.052, 0.065, 0.07, 0.073, 0.098, 0.108, 0.097, 0.059, 0.034, 0.02, 0.01, 0.005],
    'sfrd_err_up': [0.003, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.01, 0.02, 0.02, 0.02, 0.015, 0.009, 0.005, 0.003, 0.002],
    'sfrd_err_down': [0.003, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.01, 0.02, 0.02, 0.02, 0.015, 0.009, 0.005, 0.003, 0.002],
    'smdens': [8.6e7, 7.9e7, 7.2e7, 6.5e7, 5.8e7, 5.1e7, 4.5e7, 3.5e7, 2.8e7, 2.2e7, 1.7e7, 1.2e7, 6.8e6, 3.6e6, 1.9e6, 1.0e6],
    'smdens_err_up': [1.5e7, 1.4e7, 1.3e7, 1.1e7, 1.0e7, 0.9e7, 0.8e7, 0.6e7, 0.5e7, 0.4e7, 0.3e7, 0.25e7, 1.5e6, 1.0e6, 0.7e6, 0.4e6],
    'smdens_err_down': [1.5e7, 1.4e7, 1.3e7, 1.1e7, 1.0e7, 0.9e7, 0.8e7, 0.6e7, 0.5e7, 0.4e7, 0.3e7, 0.25e7, 1.5e6, 1.0e6, 0.7e6, 0.4e6]
}

def print_hdf5_structure(f, indent=0):
    """Print the structure of an HDF5 file for diagnostics."""
    for key in f.keys():
        print(' ' * indent + key)
        if isinstance(f[key], h5py.Group):
            print_hdf5_structure(f[key], indent + 2)
        else:
            print(' ' * (indent + 2) + f"Shape: {f[key].shape}, Type: {f[key].dtype}")

def find_parameters(snapshot_file):
    """
    Extract key parameters from a Gadget snapshot file.
    Returns a dictionary with parameters like H0, Omega0, BoxSize, etc.
    """
    params = {}
    
    try:
        with h5py.File(snapshot_file, 'r') as f:
            # Extract header attributes
            if 'Header' in f:
                header = f['Header'].attrs
                for key in header.keys():
                    params[key] = header[key]
            
            # Look for parameter blocks in different locations
            param_locations = ['Parameters', 'RunPars', 'Config', 'Constants']
            for loc in param_locations:
                if loc in f:
                    for key in f[loc].attrs.keys():
                        params[key] = f[loc].attrs[key]
            
            # Also check for unit information
            if 'Units' in f:
                for key in f['Units'].attrs.keys():
                    params[f"Units_{key}"] = f['Units'].attrs[key]
                    
    except Exception as e:
        print(f"Error reading parameters from {snapshot_file}: {e}")
    
    return params

def read_snapshot_data(snapshot_file, verbose=False):
    """
    Read star formation and stellar data from a Gadget snapshot.
    """
    data = {
        'redshift': None,
        'time': None,
        'boxsize': None,
        'h': None,
        'star_masses': None,
        'star_formation_times': None,
        'sfr': None,
        'gas_masses': None,
        'omega_b': None,
        'omega_m': None,
        'particle_count': {}
    }
    
    try:
        with h5py.File(snapshot_file, 'r') as f:
            if verbose:
                print(f"\nReading {snapshot_file}:")
                print_hdf5_structure(f)
            
            # Get header information
            if 'Header' in f:
                header = f['Header'].attrs
                data['redshift'] = header.get('Redshift', 0.0)
                data['time'] = header.get('Time', 0.0)
                data['boxsize'] = header.get('BoxSize', 1.0)
                
                # Get particle counts
                if 'NumPart_Total' in header:
                    for i, count in enumerate(header['NumPart_Total']):
                        data['particle_count'][f'Type{i}'] = count
            
            # Try to find Hubble parameter in different possible locations
            data['h'] = get_hubble_param(f, verbose)
            
            # Get cosmological parameters
            data['omega_b'] = get_cosmological_param(f, 'OmegaBaryon', 0.0455)
            data['omega_m'] = get_cosmological_param(f, 'Omega0', 0.3089)
            
            # Get star formation rates (gas particles, Type 0)
            if 'PartType0' in f:
                gas_count = len(f['PartType0/Masses']) if 'Masses' in f['PartType0'] else 0
                if gas_count > 0:
                    # Try different possible names for SFR
                    sfr_fields = ['StarFormationRate', 'SFR', 'Rate', 'SfrRate']
                    data['gas_masses'] = f['PartType0/Masses'][:]
                    
                    sfr_found = False
                    for field in sfr_fields:
                        if field in f['PartType0']:
                            data['sfr'] = f['PartType0'][field][:]
                            sfr_found = True
                            if verbose:
                                print(f"Found SFR field: {field} with shape: {data['sfr'].shape}")
                                print(f"SFR range: {np.min(data['sfr'])} - {np.max(data['sfr'])}")
                            break
                    
                    if not sfr_found and verbose:
                        print(f"Warning: Could not find SFR field in {snapshot_file}")
                        print(f"Available fields in PartType0: {list(f['PartType0'].keys())}")
            
            # Get stellar masses and formation times (star particles, Type 4)
            if 'PartType4' in f and 'Masses' in f['PartType4'] and len(f['PartType4/Masses']) > 0:
                star_count = len(f['PartType4/Masses'])
                data['star_masses'] = f['PartType4/Masses'][:]
                
                if verbose:
                    print(f"Found {star_count} star particles")
                    print(f"Star mass range: {np.min(data['star_masses'])} - {np.max(data['star_masses'])}")
                
                # Try different possible names for formation times
                time_fields = ['StellarFormationTime', 'FormationTime', 'BirthTime']
                birth_time_found = False
                
                for field in time_fields:
                    if field in f['PartType4'] and len(f['PartType4'][field]) > 0:
                        data['star_formation_times'] = f['PartType4'][field][:]
                        birth_time_found = True
                        if verbose:
                            print(f"Found formation time field: {field}")
                        break
                
                if not birth_time_found and verbose:
                    print(f"Warning: Could not find formation time field in {snapshot_file}")
                    print(f"Available fields in PartType4: {list(f['PartType4'].keys())}")
            
    except Exception as e:
        print(f"Error reading snapshot {snapshot_file}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return data

def get_hubble_param(f, verbose=False):
    """Try to find the Hubble parameter in different possible locations."""
    # Common names for Hubble parameter
    h_names = ['HubbleParam', 'h_val', 'h', 'H0']
    
    # Look in header first
    if 'Header' in f and 'HubbleParam' in f['Header'].attrs:
        h_val = f['Header'].attrs['HubbleParam']
        if verbose:
            print(f"Found HubbleParam in Header: {h_val}")
        return h_val
    
    # Look in various parameter groups
    param_groups = ['Parameters', 'RunPars', 'Config', 'Constants']
    for group in param_groups:
        if group in f:
            for name in h_names:
                if name in f[group].attrs:
                    h_val = f[group].attrs[name]
                    # If it's H0 in km/s/Mpc, convert to little h
                    if name == 'H0' and h_val > 10:  # Likely H0 in km/s/Mpc
                        h_val = h_val / 100.0
                    if verbose:
                        print(f"Found Hubble parameter as {name} in {group}: {h_val}")
                    return h_val
    
    # Default value
    if verbose:
        print("Could not find Hubble parameter, using default h=0.7")
    return 0.7  # Most simulations use h=0.7 if not specified

def get_cosmological_param(f, param_name, default_value):
    """Try to find a cosmological parameter in different possible locations."""
    # Look in header first
    if 'Header' in f and param_name in f['Header'].attrs:
        return f['Header'].attrs[param_name]
    
    # Look in various parameter groups
    param_groups = ['Parameters', 'RunPars', 'Config', 'Cosmology', 'Constants']
    for group in param_groups:
        if group in f and param_name in f[group].attrs:
            return f[group].attrs[param_name]
    
    # Default value
    return default_value

def calculate_sfr_density(snapshot_data, verbose=False):
    """
    Calculate star formation rate density in Msun/yr/Mpc^3 (comoving)
    from a processed snapshot.
    """
    if snapshot_data is None:
        return 0.0
    
    # First method: Use direct SFR from gas particles
    if snapshot_data['sfr'] is not None and len(snapshot_data['sfr']) > 0:
        # Sum all SFRs
        sfr_array = snapshot_data['sfr']
        
        # Some simulations may have NaN or inf values
        sfr_array = sfr_array[np.isfinite(sfr_array)]
        
        if len(sfr_array) == 0:
            if verbose:
                print("  Warning: All SFR values are non-finite")
            return 0.0
        
        total_sfr = np.sum(sfr_array)  # Usually in Msun/yr
        
        # Get the box volume in comoving (Mpc/h)^3
        box_volume = snapshot_data['boxsize']**3
        
        # Convert to physical units: Msun/yr/Mpc^3
        # Remember to account for h factors correctly
        h = snapshot_data['h']
        
        # Check if SFR already includes h-factors
        # This depends on your simulation, but most Gadget sims store SFR in Msun/yr/h
        sfr_density = total_sfr / box_volume * h**2  # Convert from (Msun/h)/(Mpc/h)^3/yr to Msun/yr/Mpc^3
        
        if verbose:
            print(f"  Total SFR: {total_sfr} Msun/yr")
            print(f"  Box volume: {box_volume} (Mpc/h)^3")
            print(f"  h value: {h}")
            print(f"  SFR density: {sfr_density} Msun/yr/Mpc^3")
        
        return sfr_density
    
    # Second method (fallback): Try to estimate from star formation history
    if snapshot_data['star_masses'] is not None and snapshot_data['star_formation_times'] is not None:
        # Without specific knowledge of how these are stored, we can make a simple approximation
        # Assuming formation_times are scale factors (common in Gadget)
        # Count stars formed in the last ~100 Myr
        
        if verbose:
            print("  Using star formation history to estimate SFR")
        
        # Extract data
        star_masses = snapshot_data['star_masses']
        formation_times = snapshot_data['star_formation_times']
        current_time = snapshot_data['time']
        
        # Filter to only include valid values
        mask = np.isfinite(star_masses) & np.isfinite(formation_times)
        star_masses = star_masses[mask]
        formation_times = formation_times[mask]
        
        if len(star_masses) == 0:
            if verbose:
                print("  Warning: No valid star particles found")
            return 0.0
        
        # In Gadget, formation times are often scale factors (a)
        # We want stars formed in the last ~100 Myr which is approximately da=0.01 at low z
        # This is a crude approximation but can give some estimate
        
        # For scale factor, larger values are more recent
        time_window = 0.01  # Adjust based on your simulation
        recently_formed = formation_times > (current_time - time_window)
        
        if np.sum(recently_formed) == 0:
            if verbose:
                print("  No recently formed stars found")
            return 0.0
        
        recent_mass = np.sum(star_masses[recently_formed])
        
        # Convert to SFR (Msun/yr)
        # Approximate 100 Myr in years
        time_window_years = 1.0e8
        approx_sfr = recent_mass / time_window_years
        
        # Convert to SFR density
        box_volume = snapshot_data['boxsize']**3
        h = snapshot_data['h']
        sfr_density = approx_sfr / box_volume * h**2
        
        if verbose:
            print(f"  Recent stellar mass: {recent_mass} Msun/h")
            print(f"  Approximate SFR: {approx_sfr} Msun/yr")
            print(f"  SFR density (approx): {sfr_density} Msun/yr/Mpc^3")
        
        return sfr_density
    
    return 0.0

def calculate_stellar_mass_density(snapshot_data, verbose=False):
    """
    Calculate stellar mass density in Msun/Mpc^3 (comoving)
    from a processed snapshot.
    """
    if snapshot_data is None or snapshot_data['star_masses'] is None:
        return 0.0
    
    # Filter out any invalid values
    star_masses = snapshot_data['star_masses']
    star_masses = star_masses[np.isfinite(star_masses)]
    
    if len(star_masses) == 0:
        if verbose:
            print("  No valid star particle masses found")
        return 0.0
    
    # Sum all stellar masses
    total_stellar_mass = np.sum(star_masses)
    
    # Get the box volume in comoving (Mpc/h)^3
    box_volume = snapshot_data['boxsize']**3
    
    # Convert to physical units: Msun/Mpc^3
    # Account for h factors correctly
    h = snapshot_data['h']
    stellar_mass_density = total_stellar_mass / box_volume * h**2  # Convert from (Msun/h)/(Mpc/h)^3 to Msun/Mpc^3
    
    if verbose:
        print(f"  Total stellar mass: {total_stellar_mass} Msun/h")
        print(f"  Box volume: {box_volume} (Mpc/h)^3")
        print(f"  h value: {h}")
        print(f"  Stellar mass density: {stellar_mass_density} Msun/Mpc^3")
    
    return stellar_mass_density

def process_snapshots(snapshot_dir, verbose=False):
    """Process all snapshots in a directory and calculate SFRD and stellar mass density."""
    # Find all snapshot files
    patterns = ['snapshot_*.hdf5', 'snap_*.hdf5', 'snap_*.h5', '*.hdf5', '*.h5']
    snapshot_files = []
    
    for pattern in patterns:
        snapshot_files.extend(glob.glob(os.path.join(snapshot_dir, pattern)))
    
    if not snapshot_files:
        print(f"No snapshot files found in {snapshot_dir}")
        return None
    
    # Sort files naturally (so snapshot_9.hdf5 comes before snapshot_10.hdf5)
    snapshot_files = sorted(snapshot_files, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
    if verbose:
        print(f"Found {len(snapshot_files)} snapshot files")
    
    results = []
    
    # Process each snapshot
    for filename in snapshot_files:
        try:
            if verbose:
                print(f"\nProcessing {filename}...")
            
            # Read the snapshot data
            snapshot_data = read_snapshot_data(filename, verbose)
            
            if snapshot_data is None:
                print(f"Skipping {filename} due to read error")
                continue
            
            # Calculate statistics
            sfr_density = calculate_sfr_density(snapshot_data, verbose)
            stellar_mass_density = calculate_stellar_mass_density(snapshot_data, verbose)
            
            # Store results
            results.append({
                'filename': os.path.basename(filename),
                'redshift': snapshot_data['redshift'],
                'time': snapshot_data['time'],
                'h': snapshot_data['h'],
                'sfr_density': sfr_density,
                'stellar_mass_density': stellar_mass_density,
                'omega_b': snapshot_data['omega_b'],
                'omega_m': snapshot_data['omega_m'],
                'boxsize': snapshot_data['boxsize'],
                'particle_counts': snapshot_data['particle_count']
            })
            
            if verbose:
                print(f"  z={snapshot_data['redshift']:.2f}, SFR={sfr_density:.2e} Msun/yr/Mpc^3, " +
                      f"M*={stellar_mass_density:.2e} Msun/Mpc^3")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort by redshift
    results.sort(key=lambda x: x['redshift'], reverse=True)
    
    return results

def madau_fit_function(z, a=0.015, b=2.7, c=2.9, d=5.6):
    """
    The Madau & Dickinson (2014) fitting function for SFRD.
    Returns SFRD in Msun/yr/Mpc^3.
    """
    return a * (1 + z)**b / (1 + ((1 + z)/c)**d)

def plot_lilly_madau(results1, results2=None, output_file="lilly_madau_plot.png", 
                     sim1_label="Simulation 1", sim2_label="Simulation 2", show_obs=True):
    """Create a Lilly-Madau plot comparing two simulations."""
    # Set up the figure with a nice layout
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Set up subplots
    ax1 = plt.subplot(gs[0])  # SFR density
    ax2 = plt.subplot(gs[1])  # Stellar mass density
    
    # Collect data points for setting axes limits
    all_redshifts = []
    all_sfrd = []
    all_smdens = []
    
    # Plot observational data
    if show_obs:
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
        
        # Plot on first panel (SFRD)
        ax1.errorbar(z_obs, sfrd_obs, yerr=[sfrd_errm, sfrd_errp], fmt='o', color='green', 
                    label='Madau & Dickinson (2014)', capsize=3, markersize=4, alpha=0.6)
        ax1.plot(z_fine, sfrd_fit, '--', color='darkgreen', label='Madau & Dickinson Fit', alpha=0.6)
        
        # Plot on second panel (Stellar mass density)
        ax2.errorbar(z_obs, smdens_obs, yerr=[smdens_errm, smdens_errp], fmt='o', color='green', 
                    label='Madau & Dickinson (2014)', capsize=3, markersize=4, alpha=0.6)
        
        # Add to data for setting axis limits
        all_redshifts.extend(z_obs)
        all_sfrd.extend(sfrd_obs)
        all_smdens.extend(smdens_obs)
    
    # Plot first simulation data
    if results1:
        redshift1 = [item['redshift'] for item in results1]
        sfr_density1 = [item['sfr_density'] for item in results1]
        stellar_density1 = [item['stellar_mass_density'] for item in results1]
        
        # Filter out zeros for better plotting
        valid_indices = [i for i, val in enumerate(sfr_density1) if val > 0]
        valid_redshift1 = [redshift1[i] for i in valid_indices]
        valid_sfr1 = [sfr_density1[i] for i in valid_indices]
        
        valid_indices = [i for i, val in enumerate(stellar_density1) if val > 0]
        valid_z_stellar1 = [redshift1[i] for i in valid_indices]
        valid_stellar1 = [stellar_density1[i] for i in valid_indices]
        
        # Plot non-zero values
        if valid_sfr1:
            ax1.plot(valid_redshift1, valid_sfr1, 'o-', label=sim1_label, color='blue', markersize=6, linewidth=2)
            all_redshifts.extend(valid_redshift1)
            all_sfrd.extend(valid_sfr1)
        
        if valid_stellar1:
            ax2.plot(valid_z_stellar1, valid_stellar1, 'o-', label=sim1_label, color='blue', markersize=6, linewidth=2)
            all_redshifts.extend(valid_z_stellar1)
            all_smdens.extend(valid_stellar1)
    
    # Plot second simulation data if provided
    if results2:
        redshift2 = [item['redshift'] for item in results2]
        sfr_density2 = [item['sfr_density'] for item in results2]
        stellar_density2 = [item['stellar_mass_density'] for item in results2]
        
        # Filter out zeros for better plotting
        valid_indices = [i for i, val in enumerate(sfr_density2) if val > 0]
        valid_redshift2 = [redshift2[i] for i in valid_indices]
        valid_sfr2 = [sfr_density2[i] for i in valid_indices]
        
        valid_indices = [i for i, val in enumerate(stellar_density2) if val > 0]
        valid_z_stellar2 = [redshift2[i] for i in valid_indices]
        valid_stellar2 = [stellar_density2[i] for i in valid_indices]
        
        # Plot non-zero values
        if valid_sfr2:
            ax1.plot(valid_redshift2, valid_sfr2, 'o-', label=sim2_label, color='red', markersize=6, linewidth=2)
            all_redshifts.extend(valid_redshift2)
            all_sfrd.extend(valid_sfr2)
        
        if valid_stellar2:
            ax2.plot(valid_z_stellar2, valid_stellar2, 'o-', label=sim2_label, color='red', markersize=6, linewidth=2)
            all_redshifts.extend(valid_z_stellar2)
            all_smdens.extend(valid_stellar2)
    
    # Configure plot 1 (SFR density)
    ax1.set_xlabel('Redshift (z)', fontsize=12)
    ax1.set_ylabel('SFR Density [M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$]', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('Cosmic Star Formation Rate Density vs. Redshift', fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize=10, loc='upper left')
    
    # Configure plot 2 (Stellar mass density)
    ax2.set_xlabel('Redshift (z)', fontsize=12)
    ax2.set_ylabel('Stellar Mass Density [M$_\\odot$ Mpc$^{-3}$]', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_title('Cosmic Stellar Mass Density vs. Redshift', fontsize=14)
    
    # Set sensible axis limits based on actual data
    if all_redshifts:
        # X-axis (redshift)
        min_z = max(0, min(all_redshifts) - 0.5)
        max_z = min(10, max(all_redshifts) + 0.5)
        
        ax1.set_xlim(min_z, max_z)
        ax2.set_xlim(min_z, max_z)
        
        # Y-axis (SFR density) - adjust for log scale and include observational range
        if all_sfrd:
            min_sfrd = min([s for s in all_sfrd if s > 0]) / 10.0
            max_sfrd = max(all_sfrd) * 10.0
            ax1.set_ylim(min_sfrd, max_sfrd)
        else:
            # Default from observations if no simulation data
            ax1.set_ylim(0.001, 0.5)
        
        # Y-axis (Stellar mass density) - adjust for log scale
        if all_smdens:
            min_smdens = min([s for s in all_smdens if s > 0]) / 10.0
            max_smdens = max(all_smdens) * 10.0
            ax2.set_ylim(min_smdens, max_smdens)
        else:
            # Default from observations if no simulation data
            ax2.set_ylim(1e5, 2e8)
    else:
        # Default limits if no data
        ax1.set_xlim(0, 10)
        ax2.set_xlim(0, 10)
        ax1.set_ylim(0.001, 0.5)
        ax2.set_ylim(1e5, 2e8)
    
    # Add timestamp
    plt.figtext(0.01, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=8, ha='left')
    
    # Add simulation metadata if available
    metadata = []
    if results1 and len(results1) > 0:
        r1 = results1[0]  # Get metadata from first snapshot
        metadata.append(f"{sim1_label}: Box={r1['boxsize']:.1f} Mpc/h, Ωm={r1['omega_m']:.4f}, Ωb={r1['omega_b']:.4f}")
    
    if results2 and len(results2) > 0:
        r2 = results2[0]  # Get metadata from first snapshot
        metadata.append(f"{sim2_label}: Box={r2['boxsize']:.1f} Mpc/h, Ωm={r2['omega_m']:.4f}, Ωb={r2['omega_b']:.4f}")
    
    if metadata:
        plt.figtext(0.5, 0.01, " | ".join(metadata), fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    return fig

def save_results_to_csv(results1, results2=None, csv_file="lilly_madau_data.csv"):
    """Save the processed results to a CSV file for further analysis."""
    import csv
    
    # Prepare the header and data
    header = ['Simulation', 'Filename', 'Redshift', 'Time', 'h', 'SFR_Density', 'Stellar_Mass_Density', 'BoxSize', 'Omega_m', 'Omega_b']
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
                item['stellar_mass_density'],
                item['boxsize'],
                item['omega_m'],
                item['omega_b']
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
                item['stellar_mass_density'],
                item['boxsize'],
                item['omega_m'],
                item['omega_b']
            ])
    
    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Data saved to {csv_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Create Lilly-Madau plot comparing Gadget simulations')
    parser.add_argument('--sim1', required=True, help='Directory containing first set of snapshots')
    parser.add_argument('--sim2', help='Directory containing second set of snapshots (optional)')
    parser.add_argument('--label1', default='Simulation 1', help='Label for first simulation')
    parser.add_argument('--label2', default='Simulation 2', help='Label for second simulation')
    parser.add_argument('--output', default='lilly_madau_comparison.png', help='Output filename')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--no-obs', action='store_true', help='Do not show observational data')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Process first set of snapshots
    print(f"Processing snapshots in {args.sim1}...")
    results1 = process_snapshots(args.sim1, args.verbose)
    
    # Process second set of snapshots if provided
    results2 = None
    if args.sim2:
        print(f"Processing snapshots in {args.sim2}...")
        results2 = process_snapshots(args.sim2, args.verbose)
    
    # Create the plot
    if results1 or results2:
        plot_lilly_madau(results1, results2, args.output, args.label1, args.label2, not args.no_obs)
        
        # Save the data to CSV
        csv_file = args.output.replace('.png', '.csv')
        save_results_to_csv(results1, results2, csv_file)
    else:
        print("No valid results to plot.")

if __name__ == "__main__":
    main()