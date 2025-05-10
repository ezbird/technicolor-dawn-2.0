#!/usr/bin/env python3
"""
Compare Gadget snapshots to identify what went wrong.
This script analyzes two consecutive snapshots to identify
what variables may have caused a simulation to go unstable.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import LogNorm

def analyze_snapshot(filename):
    """Analyze a Gadget snapshot and return key statistics."""
    print(f"Analyzing {filename}...")
    
    try:
        with h5py.File(filename, 'r') as f:
            # Get header info
            redshift = f['Header'].attrs['Redshift']
            time = f['Header'].attrs['Time']
            hubble = f['Parameters'].attrs['HubbleParam']
            box_size = f['Parameters'].attrs['BoxSize']
            
            # Get gas properties
            gas_density = f['PartType0/Density'][:]
            
            # Check if Entropy or InternalEnergy exists
            if 'PartType0/InternalEnergy' in f:
                gas_internal_energy = f['PartType0/InternalEnergy'][:]
                have_energy = True
            else:
                gas_internal_energy = np.zeros_like(gas_density)
                have_energy = False
                
            if 'PartType0/Entropy' in f:
                gas_entropy = f['PartType0/Entropy'][:]
                have_entropy = True
            else:
                gas_entropy = np.zeros_like(gas_density)
                have_entropy = False
            
            # Get velocities
            gas_vel_x = f['PartType0/Velocities'][:, 0]
            gas_vel_y = f['PartType0/Velocities'][:, 1]
            gas_vel_z = f['PartType0/Velocities'][:, 2]
            
            # Calculate velocity magnitude
            gas_vel_mag = np.sqrt(gas_vel_x**2 + gas_vel_y**2 + gas_vel_z**2)
            
            # Get masses if available
            if 'PartType0/Masses' in f:
                gas_masses = f['PartType0/Masses'][:]
            else:
                gas_masses = np.ones_like(gas_density)
                
            # Get coordinates
            gas_pos_x = f['PartType0/Coordinates'][:, 0]
            gas_pos_y = f['PartType0/Coordinates'][:, 1]
            gas_pos_z = f['PartType0/Coordinates'][:, 2]
                
            # Try to get other useful variables if available
            if 'PartType0/ElectronAbundance' in f:
                gas_ne = f['PartType0/ElectronAbundance'][:]
                have_ne = True
            else:
                gas_ne = np.ones_like(gas_density)
                have_ne = False
                
            if 'PartType0/Pressure' in f:
                gas_pressure = f['PartType0/Pressure'][:]
                have_pressure = True
            else:
                gas_pressure = np.zeros_like(gas_density)
                have_pressure = False
                
            # Get IDs to track individual particles
            gas_ids = f['PartType0/ParticleIDs'][:]
    except Exception as e:
        print(f"Error reading snapshot: {e}")
        return None
    
    # Check for NaN values in key variables
    nan_count_density = np.sum(~np.isfinite(gas_density))
    nan_count_energy = np.sum(~np.isfinite(gas_internal_energy)) if have_energy else 0
    nan_count_entropy = np.sum(~np.isfinite(gas_entropy)) if have_entropy else 0
    nan_count_vel = np.sum(~np.isfinite(gas_vel_mag))
    
    print(f"NaN counts - Density: {nan_count_density}, Energy: {nan_count_energy}, "
          f"Entropy: {nan_count_entropy}, Velocity: {nan_count_vel}")
    
    # Compute basic statistics excluding NaN values
    # For density
    valid_density = gas_density[np.isfinite(gas_density)]
    if len(valid_density) > 0:
        min_density = np.min(valid_density)
        max_density = np.max(valid_density)
        mean_density = np.mean(valid_density)
        median_density = np.median(valid_density)
    else:
        min_density = max_density = mean_density = median_density = np.nan
        
    # For internal energy
    if have_energy:
        valid_energy = gas_internal_energy[np.isfinite(gas_internal_energy)]
        if len(valid_energy) > 0:
            min_energy = np.min(valid_energy)
            max_energy = np.max(valid_energy)
            mean_energy = np.mean(valid_energy)
            median_energy = np.median(valid_energy)
        else:
            min_energy = max_energy = mean_energy = median_energy = np.nan
    else:
        min_energy = max_energy = mean_energy = median_energy = np.nan
        
    # For entropy
    if have_entropy:
        valid_entropy = gas_entropy[np.isfinite(gas_entropy)]
        if len(valid_entropy) > 0:
            min_entropy = np.min(valid_entropy)
            max_entropy = np.max(valid_entropy)
            mean_entropy = np.mean(valid_entropy)
            median_entropy = np.median(valid_entropy)
        else:
            min_entropy = max_entropy = mean_entropy = median_entropy = np.nan
    else:
        min_entropy = max_entropy = mean_entropy = median_entropy = np.nan
        
    # For velocity
    valid_vel = gas_vel_mag[np.isfinite(gas_vel_mag)]
    if len(valid_vel) > 0:
        min_vel = np.min(valid_vel)
        max_vel = np.max(valid_vel)
        mean_vel = np.mean(valid_vel)
        median_vel = np.median(valid_vel)
    else:
        min_vel = max_vel = mean_vel = median_vel = np.nan
    
    print(f"Density - Min: {min_density:.2e}, Max: {max_density:.2e}, Mean: {mean_density:.2e}, Median: {median_density:.2e}")
    if have_energy:
        print(f"Energy - Min: {min_energy:.2e}, Max: {max_energy:.2e}, Mean: {mean_energy:.2e}, Median: {median_energy:.2e}")
    if have_entropy:
        print(f"Entropy - Min: {min_entropy:.2e}, Max: {max_entropy:.2e}, Mean: {mean_entropy:.2e}, Median: {median_entropy:.2e}")
    print(f"Velocity - Min: {min_vel:.2e}, Max: {max_vel:.2e}, Mean: {mean_vel:.2e}, Median: {median_vel:.2e}")
    
    # Check for extreme values
    extreme_density_count = np.sum(valid_density > 1e4 * median_density) if len(valid_density) > 0 else 0
    extreme_vel_count = np.sum(valid_vel > 1e2 * median_vel) if len(valid_vel) > 0 else 0
    
    if have_energy:
        extreme_energy_count = np.sum(valid_energy < 1e-4 * median_energy) if len(valid_energy) > 0 else 0
        print(f"Extreme values - Density: {extreme_density_count}, Velocity: {extreme_vel_count}, Energy (low): {extreme_energy_count}")
    else:
        print(f"Extreme values - Density: {extreme_density_count}, Velocity: {extreme_vel_count}")
    
    # Return a dictionary of key data for further analysis
    result = {
        'filename': filename,
        'redshift': redshift,
        'time': time,
        'density': gas_density,
        'internal_energy': gas_internal_energy if have_energy else None,
        'entropy': gas_entropy if have_entropy else None,
        'velocity': gas_vel_mag,
        'positions': np.column_stack((gas_pos_x, gas_pos_y, gas_pos_z)),
        'particle_ids': gas_ids,
        'nan_counts': {
            'density': nan_count_density,
            'energy': nan_count_energy,
            'entropy': nan_count_entropy,
            'velocity': nan_count_vel
        },
        'statistics': {
            'density': {'min': min_density, 'max': max_density, 'mean': mean_density, 'median': median_density},
            'energy': {'min': min_energy, 'max': max_energy, 'mean': mean_energy, 'median': median_energy} if have_energy else None,
            'entropy': {'min': min_entropy, 'max': max_entropy, 'mean': mean_entropy, 'median': median_entropy} if have_entropy else None,
            'velocity': {'min': min_vel, 'max': max_vel, 'mean': mean_vel, 'median': median_vel}
        }
    }
    
    return result

def compare_snapshots(data1, data2):
    """Compare two snapshots and identify key differences."""
    if data1 is None or data2 is None:
        print("Cannot compare snapshots: one or both datasets are invalid")
        return
    
    # Compare redshift/time
    print(f"\nSnapshot comparison:")
    print(f"Time: {data1['time']:.6f} -> {data2['time']:.6f}")
    print(f"Redshift: {data1['redshift']:.6f} -> {data2['redshift']:.6f}")
    
    # Compare NaN counts
    print("\nNaN count changes:")
    for key in data1['nan_counts']:
        if data1['nan_counts'][key] is not None and data2['nan_counts'][key] is not None:
            diff = data2['nan_counts'][key] - data1['nan_counts'][key]
            print(f"  {key}: {data1['nan_counts'][key]} -> {data2['nan_counts'][key]} (Change: {diff:+d})")
    
    # Compare statistics
    print("\nStatistics changes:")
    for key in ['density', 'energy', 'entropy', 'velocity']:
        if data1['statistics'][key] is not None and data2['statistics'][key] is not None:
            print(f"\n{key.capitalize()}:")
            for stat in ['min', 'max', 'mean', 'median']:
                val1 = data1['statistics'][key][stat]
                val2 = data2['statistics'][key][stat]
                if np.isfinite(val1) and np.isfinite(val2):
                    rel_change = (val2 - val1) / val1 * 100 if val1 != 0 else float('inf')
                    print(f"  {stat}: {val1:.4e} -> {val2:.4e} (Change: {rel_change:+.2f}%)")
                else:
                    print(f"  {stat}: {val1} -> {val2} (Cannot calculate change)")
    
    # Track specific particles with issues
    print("\nTracking particles between snapshots:")
    
    # Match particle IDs between snapshots
    common_ids = np.intersect1d(data1['particle_ids'], data2['particle_ids'])
    print(f"Common particles: {len(common_ids)}")
    
    # Find indices of these particles in each snapshot
    idx1 = np.array([np.where(data1['particle_ids'] == pid)[0][0] for pid in common_ids])
    idx2 = np.array([np.where(data2['particle_ids'] == pid)[0][0] for pid in common_ids])
    
    # Check for particles that were healthy in snap1 but NaN in snap2
    if data1['internal_energy'] is not None and data2['internal_energy'] is not None:
        healthy_in_1 = np.isfinite(data1['internal_energy'][idx1])
        nan_in_2 = ~np.isfinite(data2['internal_energy'][idx2])
        became_nan = np.logical_and(healthy_in_1, nan_in_2)
        
        print(f"Particles with good energy in snap1 but NaN in snap2: {np.sum(became_nan)}")
        
        if np.sum(became_nan) > 0:
            # Analyze a sample of these particles
            sample_size = min(10, np.sum(became_nan))
            sample_idx = np.where(became_nan)[0][:sample_size]
            
            print("\nSample particles that transitioned to NaN:")
            for i in sample_idx:
                pid = common_ids[i]
                idx_in_1 = idx1[i]
                idx_in_2 = idx2[i]
                
                print(f"\nParticle ID: {pid}")
                print(f"  Density: {data1['density'][idx_in_1]:.4e} -> {data2['density'][idx_in_2]}")
                print(f"  Internal Energy: {data1['internal_energy'][idx_in_1]:.4e} -> {data2['internal_energy'][idx_in_2]}")
                if data1['entropy'] is not None and data2['entropy'] is not None:
                    print(f"  Entropy: {data1['entropy'][idx_in_1]:.4e} -> {data2['entropy'][idx_in_2]}")
                print(f"  Velocity: {data1['velocity'][idx_in_1]:.4e} -> {data2['velocity'][idx_in_2]}")
                
                # Check for extreme changes in healthy variables between snapshots
                if np.isfinite(data1['density'][idx_in_1]) and np.isfinite(data2['density'][idx_in_2]):
                    density_change = data2['density'][idx_in_2] / data1['density'][idx_in_1]
                    print(f"  Density change ratio: {density_change:.4f}x")
                
                if np.isfinite(data1['velocity'][idx_in_1]) and np.isfinite(data2['velocity'][idx_in_2]):
                    vel_change = data2['velocity'][idx_in_2] / data1['velocity'][idx_in_1]
                    print(f"  Velocity change ratio: {vel_change:.4f}x")
    
    # Create plots for better visualization
    os.makedirs("snapshot_analysis", exist_ok=True)
    
    # Plot 1: Phase diagram evolution
    if data1['internal_energy'] is not None and data2['internal_energy'] is not None:
        plt.figure(figsize=(12, 6))
        
        # Plot for snapshot 1
        plt.subplot(1, 2, 1)
        valid_idx = np.logical_and(np.isfinite(data1['density']), np.isfinite(data1['internal_energy']))
        if np.sum(valid_idx) > 0:
            plt.hexbin(np.log10(data1['density'][valid_idx]), 
                      np.log10(data1['internal_energy'][valid_idx]),
                      bins='log', gridsize=50, cmap='viridis')
        plt.xlabel('log Density')
        plt.ylabel('log Internal Energy')
        plt.title(f'Snapshot {os.path.basename(data1["filename"])}')
        plt.colorbar(label='Count')
        
        # Plot for snapshot 2
        plt.subplot(1, 2, 2)
        valid_idx = np.logical_and(np.isfinite(data2['density']), np.isfinite(data2['internal_energy']))
        if np.sum(valid_idx) > 0:
            plt.hexbin(np.log10(data2['density'][valid_idx]), 
                      np.log10(data2['internal_energy'][valid_idx]),
                      bins='log', gridsize=50, cmap='viridis')
        plt.xlabel('log Density')
        plt.ylabel('log Internal Energy')
        plt.title(f'Snapshot {os.path.basename(data2["filename"])}')
        plt.colorbar(label='Count')
        
        plt.tight_layout()
        plt.savefig("snapshot_analysis/phase_diagram_comparison.png", dpi=300)
        print("Phase diagram saved to snapshot_analysis/phase_diagram_comparison.png")
        
    # Plot 2: Velocity distribution evolution
    plt.figure(figsize=(12, 6))
    
    # Plot for snapshot 1
    plt.subplot(1, 2, 1)
    valid_idx = np.isfinite(data1['velocity'])
    if np.sum(valid_idx) > 0:
        plt.hist(np.log10(data1['velocity'][valid_idx]), bins=50, alpha=0.7)
    plt.xlabel('log Velocity')
    plt.ylabel('Count')
    plt.title(f'Velocity Histogram - {os.path.basename(data1["filename"])}')
    
    # Plot for snapshot 2
    plt.subplot(1, 2, 2)
    valid_idx = np.isfinite(data2['velocity'])
    if np.sum(valid_idx) > 0:
        plt.hist(np.log10(data2['velocity'][valid_idx]), bins=50, alpha=0.7)
    plt.xlabel('log Velocity')
    plt.ylabel('Count')
    plt.title(f'Velocity Histogram - {os.path.basename(data2["filename"])}')
    
    plt.tight_layout()
    plt.savefig("snapshot_analysis/velocity_comparison.png", dpi=300)
    print("Velocity comparison saved to snapshot_analysis/velocity_comparison.png")
    
    # Plot 3: Spatial distribution of NaN particles
    if data1['internal_energy'] is not None and data2['internal_energy'] is not None:
        plt.figure(figsize=(10, 8))
        
        # Find particles that became NaN
        valid_in_1 = np.isfinite(data1['internal_energy'][idx1])
        nan_in_2 = ~np.isfinite(data2['internal_energy'][idx2])
        became_nan = np.logical_and(valid_in_1, nan_in_2)
        
        # Original positions of particles that became NaN
        if np.sum(became_nan) > 0:
            pos = data1['positions'][idx1[became_nan]]
            
            # Plot 3D scatter
            ax = plt.axes(projection='3d')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=1, alpha=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Positions of Particles that Transitioned to NaN')
            
            plt.savefig("snapshot_analysis/nan_particle_positions.png", dpi=300)
            print("NaN particle positions saved to snapshot_analysis/nan_particle_positions.png")
    
    # Plot 4: Changes in key variables for individual particles
    if len(common_ids) > 0 and data1['internal_energy'] is not None and data2['internal_energy'] is not None:
        plt.figure(figsize=(12, 10))
        
        # Only include particles that were valid in both snapshots for the variable being plotted
        valid_energy = np.logical_and(np.isfinite(data1['internal_energy'][idx1]), 
                                     np.isfinite(data2['internal_energy'][idx2]))
        valid_density = np.logical_and(np.isfinite(data1['density'][idx1]), 
                                      np.isfinite(data2['density'][idx2]))
        valid_velocity = np.logical_and(np.isfinite(data1['velocity'][idx1]), 
                                       np.isfinite(data2['velocity'][idx2]))
        
        # Energy change vs initial energy
        plt.subplot(2, 2, 1)
        if np.sum(valid_energy) > 0:
            energy_ratio = data2['internal_energy'][idx2[valid_energy]] / data1['internal_energy'][idx1[valid_energy]]
            plt.scatter(data1['internal_energy'][idx1[valid_energy]], energy_ratio, 
                       s=1, alpha=0.3, c='blue')
            plt.axhline(y=1, color='red', linestyle='--')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Initial Energy')
            plt.ylabel('Energy Ratio (snap2/snap1)')
            plt.title('Energy Change vs Initial Energy')
        
        # Density change vs initial density
        plt.subplot(2, 2, 2)
        if np.sum(valid_density) > 0:
            density_ratio = data2['density'][idx2[valid_density]] / data1['density'][idx1[valid_density]]
            plt.scatter(data1['density'][idx1[valid_density]], density_ratio, 
                       s=1, alpha=0.3, c='green')
            plt.axhline(y=1, color='red', linestyle='--')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Initial Density')
            plt.ylabel('Density Ratio (snap2/snap1)')
            plt.title('Density Change vs Initial Density')
        
        # Velocity change vs initial velocity
        plt.subplot(2, 2, 3)
        if np.sum(valid_velocity) > 0:
            vel_ratio = data2['velocity'][idx2[valid_velocity]] / data1['velocity'][idx1[valid_velocity]]
            plt.scatter(data1['velocity'][idx1[valid_velocity]], vel_ratio, 
                       s=1, alpha=0.3, c='purple')
            plt.axhline(y=1, color='red', linestyle='--')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Initial Velocity')
            plt.ylabel('Velocity Ratio (snap2/snap1)')
            plt.title('Velocity Change vs Initial Velocity')
        
        # Energy vs Density colored by velocity
        plt.subplot(2, 2, 4)
        if np.sum(valid_energy) > 0 and np.sum(valid_density) > 0 and np.sum(valid_velocity) > 0:
            # Find particles valid in all three
            all_valid = np.logical_and(valid_energy, np.logical_and(valid_density, valid_velocity))
            if np.sum(all_valid) > 0:
                sc = plt.scatter(data1['density'][idx1[all_valid]], 
                                data1['internal_energy'][idx1[all_valid]],
                                c=data1['velocity'][idx1[all_valid]], 
                                s=2, alpha=0.5, cmap='viridis')
                plt.colorbar(sc, label='Velocity')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Density')
                plt.ylabel('Internal Energy')
                plt.title('Energy vs Density (colored by velocity)')
        
        plt.tight_layout()
        plt.savefig("snapshot_analysis/particle_changes.png", dpi=300)
        print("Particle changes analysis saved to snapshot_analysis/particle_changes.png")

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_snapshots.py snapshot1.hdf5 snapshot2.hdf5")
        sys.exit(1)
        
    snapshot1 = sys.argv[1]
    snapshot2 = sys.argv[2]
    
    if not os.path.exists(snapshot1):
        print(f"Error: Snapshot file {snapshot1} not found")
        sys.exit(1)
        
    if not os.path.exists(snapshot2):
        print(f"Error: Snapshot file {snapshot2} not found")
        sys.exit(1)
    
    # Analyze each snapshot
    data1 = analyze_snapshot(snapshot1)
    data2 = analyze_snapshot(snapshot2)
    
    # Compare them
    compare_snapshots(data1, data2)

if __name__ == "__main__":
    main()