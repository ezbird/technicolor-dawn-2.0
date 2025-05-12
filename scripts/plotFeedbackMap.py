'''
Map to show the impact of the Feedback module by showing where energy 
is being deposited into the gas...
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
import glob
import os

# Create a temperature map from aan individual snapshot file
def create_temperature_map(snapshot_file, slice_thickness=0.5, resolution=200):

    print(f"\nProcessing {snapshot_file}...")
    
    try:
        with h5py.File("../"+snapshot_file, 'r') as f:
            # Extract header info
            time = f['Header'].attrs['Time']  # Scale factor
            
            # Calculate redshift from scale factor: z = 1/a - 1
            redshift = 1.0/time - 1.0
            
            # Count star particles (Type 4)
            num_stars = 0
            if 'PartType4' in f:
                num_stars = len(f['PartType4/Coordinates'])
            
            # Extract gas particle data
            pos = f['PartType0/Coordinates'][:]
            mass = f['PartType0/Masses'][:]
            
            # Get temperature data
            if 'PartType0/Temperature' in f:
                temp = f['PartType0/Temperature'][:]
            else:
                if 'PartType0/InternalEnergy' in f:
                    u = f['PartType0/InternalEnergy'][:]
                    temp = u * 10000  # Adjust conversion factor as needed
                else:
                    return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), time, redshift, num_stars
            
            # Create 2D histogram (projection) - use thicker slice
            mask = np.abs(pos[:, 2]) < slice_thickness
            print(f"Particles in slice: {np.sum(mask)} out of {len(mask)} ({np.sum(mask)/len(mask)*100:.2f}%)")
            
            if np.sum(mask) == 0:
                return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), time, redshift, num_stars
            
            # Determine ranges based on particle positions
            x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
            y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
            
            # Add buffer
            x_buffer = (x_max - x_min) * 0.05
            y_buffer = (y_max - y_min) * 0.05
            
            x_edges = np.linspace(x_min - x_buffer, x_max + x_buffer, resolution)
            y_edges = np.linspace(y_min - y_buffer, y_max + y_buffer, resolution)
            
            # Create temperature map
            H, _, _ = np.histogram2d(
                pos[mask, 0], pos[mask, 1], 
                bins=[x_edges, y_edges], 
                weights=temp[mask]*mass[mask]
            )
            
            # Mass map for normalization
            M, _, _ = np.histogram2d(
                pos[mask, 0], pos[mask, 1], 
                bins=[x_edges, y_edges], 
                weights=mass[mask]
            )
            
            # Mass-weighted temperature, with Gaussian smoothing
            temp_map = np.zeros_like(H)
            nonzero = M > 0
            temp_map[nonzero] = H[nonzero] / M[nonzero]
            
            # Apply Gaussian smoothing to make the map more continuous
            from scipy.ndimage import gaussian_filter
            temp_map = gaussian_filter(temp_map, sigma=1.0)
            
            return temp_map, x_edges, y_edges, time, redshift, num_stars
            
    except Exception as e:
        print(f"ERROR processing {snapshot_file}: {str(e)}")
        return np.zeros((resolution, resolution)), np.linspace(-20, 20, resolution), np.linspace(-20, 20, resolution), 0, 0, 0

# Get snapshot files
snapshot_pattern = "snap*.hdf5"
snapshot_files = sorted(glob.glob("output/"+snapshot_pattern))
print(f"Found {len(snapshot_files)} snapshot files matching pattern: {snapshot_pattern}")

if len(snapshot_files) == 0:
    print("No snapshot files found! Please check the file pattern and directory.")
    exit(1)

# Create output directory
os.makedirs("temp_diagnostics", exist_ok=True)

# Process all snapshots once, storing the data for animation
print("Processing all snapshots and saving diagnostic frames...")
all_data = []

for i, filename in enumerate(snapshot_files):
    # Extract snapshot number from filename
    try:
        snap_num = int(filename.split("_")[-1].split(".")[0])
    except:
        snap_num = i
    
    # Create temperature map
    temp_map, x_edges, y_edges, time, redshift, num_stars = create_temperature_map(filename)
    
    # Store for animation
    all_data.append((temp_map, x_edges, y_edges, time, redshift, num_stars, snap_num))
    
    # Save diagnostic frame
    plt.figure(figsize=(12, 10))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    plt.imshow(temp_map.T, origin='lower', extent=extent,
               norm=LogNorm(vmin=1e4, vmax=1e7), 
               cmap='nipy_spectral', interpolation='gaussian', aspect='auto')
    plt.colorbar(label='Temperature [K]', format='%.1e')
    plt.xlabel('x [kpc]', fontsize=14)
    plt.ylabel('y [kpc]', fontsize=14)
    plt.title(f'Gas Temperature Map - Snapshot {snap_num} - z={redshift:.2f} - Stars: {num_stars}', fontsize=16)
    plt.grid(True, color='white', alpha=0.3, linestyle=':')
    
    plt.savefig(f"temp_diagnostics/frame_{i:04d}.png", dpi=150)
    plt.close()

# Set up the figure for animation
fig, ax = plt.subplots(figsize=(12, 10))

# Get the first frame data
first_data = all_data[0]
temp_map, x_edges, y_edges, time, redshift, num_stars, snap_num = first_data

# Initialize the plot
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
im = ax.imshow(temp_map.T, origin='lower', extent=extent,
               norm=LogNorm(vmin=1e4, vmax=1e7), 
               cmap='nipy_spectral', interpolation='gaussian', aspect='auto')
cbar = fig.colorbar(im, label='Temperature [K]', format='%.1e')
ax.set_xlabel('x [kpc]', fontsize=14)
ax.set_ylabel('y [kpc]', fontsize=14)
title = ax.set_title(f'Gas Temperature Map - Snapshot {snap_num} - z={redshift:.2f} - Stars: {num_stars}', fontsize=16)
time_text = ax.text(0.05, 0.95, f"Scale Factor: {time:.3f}", transform=ax.transAxes,
                   fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
ax.grid(True, color='white', alpha=0.3, linestyle=':')

def update(frame):
    # Get pre-processed data for this frame
    temp_map, x_edges, y_edges, time, redshift, num_stars, snap_num = all_data[frame]
    
    # Clear the axis
    ax.clear()
    
    # Create a new image with the updated data
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(temp_map.T, origin='lower', extent=extent,
                  norm=LogNorm(vmin=1e4, vmax=1e7), 
                  cmap='nipy_spectral', interpolation='gaussian', aspect='auto')
    
    # Add grid
    ax.grid(True, color='white', alpha=0.3, linestyle=':')
    
    # Update labels and title
    ax.set_xlabel('x [kpc]', fontsize=14)
    ax.set_ylabel('y [kpc]', fontsize=14)
    title = ax.set_title(f'Gas Temperature Map - Snapshot {snap_num} - z={redshift:.2f} - Stars: {num_stars}', fontsize=16)
    time_text = ax.text(0.05, 0.95, f"Scale Factor: {time:.3f}", transform=ax.transAxes,
                      fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
    
    return [im, title, time_text]

# Create animation using pre-processed data
print("\nCreating animation...")
ani = animation.FuncAnimation(fig, update, frames=len(all_data), 
                              blit=False, interval=200)

# Save the animation
print("Saving animation to temperature_evolution.mp4...")
try:
    ani.save("temperature_evolution.mp4", writer='ffmpeg', dpi=150)
    print("Animation saved successfully!")
except Exception as e:
    print(f"Error saving animation: {str(e)}")
    print("Check the individual frames in the temp_diagnostics folder.")

plt.show()
print("Done!")