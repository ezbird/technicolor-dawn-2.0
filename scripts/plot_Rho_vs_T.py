import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.colors import LogNorm

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
    
    # Read data from all files
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
    exit(1)

for snapfile in snapshots:
    if use_multifile:
        # Handle multi-file snapshots
        print(f"\n=== Processing {snapfile} ===")
        
        rho_comov, u_code, hdr, params = read_gas_data_multifile(snapfile)
        
        if rho_comov is None or u_code is None:
            print(f"Skipping {snapfile} - no gas data found")
            continue
            
        snap_name = os.path.basename(snapfile)
        
    else:
        # Handle single-file snapshots (original logic)
        print(f"\n=== Processing {snapfile} ===")
        
        with h5py.File(snapfile, 'r') as f:
            hdr = dict(f['Header'].attrs)
            params = dict(f['Parameters'].attrs) if 'Parameters' in f else {}
            
            if 'PartType0' not in f:
                print(f"Skipping {snapfile} - no gas particles")
                continue
                
            rho_comov = f['PartType0/Density'][...]
            u_code = f['PartType0/InternalEnergy'][...]
            
        snap_name = snapfile.split("/")[-1]
    
    # Common processing for both file types
    a = hdr['Time']
    
    # --- dump SF parameters ---
    print("CritPhysDensity:", hdr.get("CritPhysDensity", 0.0))
    print("PhysDensThresh:  ", hdr.get("PhysDensThresh",   0.0))
    print("CritOverDensity: ", hdr.get("CritOverDensity",  0.0))
    print(f"Number of gas particles: {len(rho_comov)}")

    # --- compute physical density (g/cm³) ---
    rho_phys_cu = rho_comov / a**3
    rho_cgs     = rho_phys_cu * (UnitMass / UnitLen**3)

    # --- compute temperature (K) ---
    u_cgs  = u_code * (1e5)**2                        # UnitVel=1e5 cm/s
    T      = u_cgs * (gamma - 1.0) * mu * m_H / k_B

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8,6))
    hist = ax.hist2d(
        np.log10(rho_cgs), np.log10(T),
        bins=200, norm=LogNorm(), cmap='plasma'
    )

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
    ax.set_title(f'ρ–T   {snap_name}   (a={a:.4f})')
    ax.legend(loc='upper left', fontsize='small')

    cbar = plt.colorbar(hist[3], ax=ax)
    cbar.set_label('Gas Count (log scale)')

    plt.tight_layout()
    plt.show()

print("Finished processing all snapshots!")