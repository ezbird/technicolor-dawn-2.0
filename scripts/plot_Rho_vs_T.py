import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.colors import LogNorm
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Constants
UnitMass = 1.989e43     # g (≈1e10 M☉)
UnitLen  = 3.085678e21  # cm (1 kpc)
k_B      = 1.380649e-16 # erg/K
m_H      = 1.6735575e-24# g
mu       = 0.6
gamma    = 5.0/3.0
G        = 6.674e-8     # cm³ g⁻¹ s⁻²

def read_gas_data_multifile(snapdir):
    """
    Read gas data from multi-file Gadget-4 snapshot.
    Returns combined density, internal energy, header, and parameters.
    """
    # Find all .hdf5 files in the snapshot directory
    snap_files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    
    if not snap_files:
        print(f"Warning: No HDF5 files found in {snapdir}")
        return None, None, None, None
    
    # Initialize storage for combined data
    densities = []
    internal_energies = []
    header = None
    parameters = None
    
    # Read data from all files in parallel using list comprehension for I/O
    for snap_file in snap_files:
        with h5py.File(snap_file, 'r') as f:
            # Read header and parameters from first file
            if header is None:
                header = dict(f['Header'].attrs)
                if 'Parameters' in f:
                    parameters = dict(f['Parameters'].attrs)
            
            # Read gas particle data if present
            if 'PartType0' in f:
                if 'Density' in f['PartType0']:
                    densities.append(f['PartType0/Density'][...])
                if 'InternalEnergy' in f['PartType0']:
                    internal_energies.append(f['PartType0/InternalEnergy'][...])
    
    # Combine data from all files
    combined_density = np.concatenate(densities) if densities else None
    combined_internal_energy = np.concatenate(internal_energies) if internal_energies else None
    
    return combined_density, combined_internal_energy, header, parameters

def process_single_snapshot(args):
    """
    Process a single snapshot (designed for multiprocessing).
    """
    snapfile, use_multifile = args
    
    try:
        start_time = time.time()
        
        if use_multifile:
            # Handle multi-file snapshots
            rho_comov, u_code, hdr, params = read_gas_data_multifile(snapfile)
            
            if rho_comov is None or u_code is None:
                return f"Skipped {snapfile} - no gas data found"
                
            snap_name = os.path.basename(snapfile)
            
        else:
            # Handle single-file snapshots
            with h5py.File(snapfile, 'r') as f:
                hdr = dict(f['Header'].attrs)
                params = dict(f['Parameters'].attrs) if 'Parameters' in f else {}
                
                if 'PartType0' not in f:
                    return f"Skipped {snapfile} - no gas particles"
                    
                rho_comov = f['PartType0/Density'][...]
                u_code = f['PartType0/InternalEnergy'][...]
                
            snap_name = snapfile.split("/")[-1]
        
        # Common processing for both file types
        a = hdr['Time']
        redshift = 1.0 / a - 1.0
        
        # --- compute physical density (g/cm³) ---
        rho_phys_cu = rho_comov / a**3
        rho_cgs     = rho_phys_cu * (UnitMass / UnitLen**3)

        # --- compute temperature (K) ---
        u_cgs  = u_code * (1e5)**2                        # UnitVel=1e5 cm/s
        T      = u_cgs * (gamma - 1.0) * mu * m_H / k_B

        # Use numpy's histogram2d for faster computation than matplotlib's hist2d
        log_rho = np.log10(rho_cgs)
        log_T = np.log10(T)
        
        # Define bins
        rho_bins = np.linspace(log_rho.min(), log_rho.max(), 200)
        T_bins = np.linspace(log_T.min(), log_T.max(), 200)
        
        # Fast 2D histogram
        hist, rho_edges, T_edges = np.histogram2d(log_rho, log_T, bins=[rho_bins, T_bins])
        hist = hist.T  # Transpose for correct orientation
        
        # --- plot ---
        fig, ax = plt.subplots(figsize=(8,6))
        
        # Use pcolormesh for faster plotting than hist2d
        X, Y = np.meshgrid(rho_edges, T_edges)
        im = ax.pcolormesh(X, Y, hist, norm=LogNorm(vmin=1), cmap='plasma', shading='auto')

        # Add threshold lines
        # 1) physical threshold from CritPhysDensity
        cpd = hdr.get("CritPhysDensity", 0.0)
        if cpd > 0:
            ax.axvline(
                np.log10(cpd),
                linestyle='--', color='k',
                label='CritPhysDensity'
            )

        # 2) code-unit threshold PhysDensThresh → physical
        pdt = hdr.get("PhysDensThresh", 0.0)
        if pdt > 0:
            thr_pdt = pdt * (UnitMass/UnitLen**3) / a**3
            ax.axvline(
                np.log10(thr_pdt),
                linestyle='-.', color='w',
                label='PhysDensThresh→phys'
            )

        # 3) overdensity threshold CritOverDensity → physical
        cod = hdr.get("CritOverDensity", 0.0)
        if cod > 0 and params:
            H0_cgs    = params['HubbleParam'] * 100 * 1e5 / (1000 * UnitLen)
            rho_crit0 = 3 * H0_cgs**2 / (8 * np.pi * G)
            rho_b0    = params['OmegaBaryon'] * rho_crit0
            thr_cod   = cod * rho_b0 / a**3
            ax.axvline(
                np.log10(thr_cod),
                linestyle=':', color='gray',
                label='CritOverDensity→phys'
            )

        ax.set_xlabel('log10(Density [g/cm³])')
        ax.set_ylabel('log10(Temperature [K])')
        ax.set_title(f'ρ–T   {snap_name}   (z={redshift:.2f})')
        ax.legend(loc='upper left', fontsize='small')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Gas Count (log scale)')

        plt.tight_layout()
        
        # Save as PNG
        if use_multifile:
            snap_num = int(os.path.basename(snapfile).split("_")[-1])
        else:
            snap_num = int(snap_name.split("_")[-1].split(".")[0])
        
        output_filename = f"rho_T_snap_{snap_num:03d}_z{redshift:.2f}.png"
        plt.savefig("rho_T_frames/"+output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        elapsed = time.time() - start_time
        return f"Saved: {output_filename} ({elapsed:.1f}s, {len(rho_comov):,} particles)"
        
    except Exception as e:
        return f"Error processing {snapfile}: {str(e)}"

def main():
    # Find snapshot directories first, then fall back to single files
    snapshot_dirs = sorted(glob.glob("../output/snapdir_*"))
    snapshot_files = sorted(glob.glob("../output/snapshot_*.hdf5"))

    if snapshot_dirs:
        use_multifile = True
        snapshots = snapshot_dirs
        print(f"Found {len(snapshot_dirs)} multi-file snapshot directories")
    elif snapshot_files:
        use_multifile = False
        snapshots = snapshot_files
        print(f"Found {len(snapshot_files)} single-file snapshots")
    else:
        print("No snapshots found!")
        return

    # Prepare arguments for parallel processing
    args_list = [(snap, use_multifile) for snap in snapshots]
    
    # Determine number of cores to use (leave 1 core free)
    n_cores = max(1, cpu_count() - 1)
    print(f"Using {n_cores} cores for parallel processing")
    
    start_total = time.time()
    
    # Process snapshots in parallel
    with Pool(n_cores) as pool:
        results = pool.map(process_single_snapshot, args_list)
    
    # Print results
    for result in results:
        print(result)
    
    total_time = time.time() - start_total
    print(f"\nTotal processing time: {total_time:.1f}s")
    print("Finished processing all snapshots!")

if __name__ == "__main__":
    main()