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

def read_multifile_snapshot(snapdir):
    """
    Read a multi-file Gadget-4 snapshot from a directory.
    Returns combined data for all particle types and header info.
    """
    # Find all .hdf5 files in the snapshot directory
    snap_files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    
    if not snap_files:
        print(f"Warning: No HDF5 files found in {snapdir}")
        return None, None, None
    
    # Initialize storage for combined data
    combined_data = {}
    header = None
    
    # Read data from all files
    for snap_file in snap_files:
        with h5py.File(snap_file, 'r') as f:
            # Read header from first file
            if header is None:
                header = dict(f['Header'].attrs)
            
            # Read particle data for each type
            for part_type in part_colors.keys():
                if part_type in f:
                    coords = f[part_type]['Coordinates'][:]
                    
                    if part_type not in combined_data:
                        combined_data[part_type] = []
                    
                    combined_data[part_type].append(coords)
    
    # Concatenate coordinates for each particle type
    final_data = {}
    for part_type, coord_list in combined_data.items():
        if coord_list:
            final_data[part_type] = np.concatenate(coord_list, axis=0)
    
    return final_data, header, snap_files[0]  # Return first file for naming

# Find snapshot directories
snapshot_dirs = sorted(glob.glob("../output/snapdir_*"))

if not snapshot_dirs:
    print("No snapshot directories found. Looking for legacy single-file snapshots...")
    # Fallback to single-file snapshots if no directories found
    snapshot_files = sorted(glob.glob("../output/snap*.hdf5"))
    use_multifile = False
else:
    use_multifile = True

# Create output folder for frames
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

if use_multifile:
    print(f"Found {len(snapshot_dirs)} multi-file snapshot directories")
    
    # Loop through each snapshot directory
    for frame, snapdir in enumerate(snapshot_dirs):
        print(f"Processing {snapdir}...")
        
        # Read multi-file snapshot
        particle_data, header, sample_file = read_multifile_snapshot(snapdir)
        
        if particle_data is None:
            continue
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot each particle type
        for part_type, props in part_colors.items():
            if part_type in particle_data:
                coords = particle_data[part_type]
                ax.scatter(coords[:, 0], coords[:, 1],
                          s=props['size'],
                          c=props['color'],
                          label=props['name'],
                          alpha=0.7)
        
        # Extract snapshot info
        time = header['Time']
        redshift = 1.0 / time - 1.0
        snap_num = int(os.path.basename(snapdir).split("_")[-1])
        num_stars = len(particle_data.get('PartType4', [])) if 'PartType4' in particle_data else 0
        
        ax.set_title(f'Particle Map - Snapshot {snap_num} - z={redshift:.2f} - Stars: {num_stars}', fontsize=16)
        ax.set_xlabel("X [kpc]")
        ax.set_ylabel("Y [kpc]")
        ax.legend(loc='upper right')
        
        # Save the figure
        frame_path = os.path.join(output_folder, f"frame_{frame:03d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved frame {frame:03d} for snapshot {snap_num}")

else:
    print(f"Using legacy single-file format. Found {len(snapshot_files)} snapshots")
    
    # Original single-file logic
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
        ax.set_ylabel("Y [kpc]")
        ax.legend(loc='upper right')

        # Save the figure
        frame_path = os.path.join(output_folder, f"frame_{frame:03d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

print("Finished generating frames!")