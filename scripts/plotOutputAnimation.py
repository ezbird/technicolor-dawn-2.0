import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Define colors for each PartType
part_colors = {
    'PartType0': {'name': 'Gas', 'color': 'blue', 'size': 2},
    'PartType1': {'name': 'Dark Matter', 'color': 'black', 'size': 2},
    'PartType2': {'name': 'Disk', 'color': 'brown', 'size': 3},
    'PartType3': {'name': 'Bulge', 'color': 'orange', 'size': 3},
    'PartType4': {'name': 'Stars', 'color': 'red', 'size': 5},
    'PartType5': {'name': 'Black Holes', 'color': 'purple', 'size': 3},
    'PartType6': {'name': 'Dust', 'color': 'green', 'size': 5}
}

# Snapshot files
snapshot_files = sorted(glob.glob("../output/snap*.hdf5"))

# Create output folder for frames
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Loop through each snapshot and create a figure
for frame, filename in enumerate(snapshot_files):
    fig, ax = plt.subplots(figsize=(8, 8))
    with h5py.File(filename, 'r') as f:
        for part, props in part_colors.items():
            if part in f:
                coords = f[part]['Coordinates'][:]
                ax.scatter(coords[:, 0], coords[:, 1],
                           s=props['size'],
                           c=props['color'],
                           label=props['name'],
                           alpha=0.7)

        time = f['Header'].attrs['Time']
        redshift = 1.0 / time - 1.0
        snap_num = int(filename.split("_")[-1].split(".")[0])
        num_stars = len(f['PartType4/Coordinates']) if 'PartType4' in f else 0

    ax.set_title(f'Particle Map - Snapshot {snap_num} - z={redshift:.2f} - Stars: {num_stars}', fontsize=16)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("_
