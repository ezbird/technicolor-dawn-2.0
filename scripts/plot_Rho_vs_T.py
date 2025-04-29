import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Constants
k_B = 1.380649e-16     # Boltzmann constant in erg/K
m_H = 1.6735575e-24    # Mass of hydrogen atom in g
mu = 0.6               # Mean molecular weight for fully ionized gas
gamma = 5.0 / 3.0      # Adiabatic index for monoatomic gas

# Gather snapshot files
snapshots = sorted(glob.glob("../output/snapshot_*.hdf5"))

# Output directory for PNGs
output_dir = "rho_T_frames"
os.makedirs(output_dir, exist_ok=True)

for snapfile in snapshots:
    with h5py.File(snapfile, "r") as snap:
        header = dict(snap["Header"].attrs)
        a = header["Time"]
        
        # Unit conversions
        UnitMass = 1.989e43       # grams per code mass unit
        UnitLen = 3.085678e24     # cm per code length unit
        UnitVel = 1e5             # cm/s per code velocity unit
        
        # Load gas data
        rho_code = np.array(snap["PartType0/Density"]) * a**3
        rho_cgs = rho_code * (UnitMass / UnitLen**3)
        
        u_code = np.array(snap["PartType0/InternalEnergy"])
        u_cgs = u_code * UnitVel**2  # erg/g
        
        # Compute temperature
        T = u_cgs * (gamma - 1.0) * mu * m_H / k_B
        
        # Plot
        fig, ax = plt.subplots(figsize=(8,7))
        scatter = ax.scatter(rho_cgs, T, c=np.log10(T), cmap='plasma', s=3, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Density [g/cm³]")
        ax.set_ylabel("Temperature [K]")
        snapname = os.path.basename(snapfile)
        ax.set_title(f"ρ–T Diagram\n{snapname} (a={a:.4f})")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log10(Temperature [K])')

        plt.tight_layout()
        savepath = os.path.join(output_dir, f"{snapname}.png")
        plt.savefig(savepath, dpi=150)
        plt.close(fig)

        print(f"[INFO] Saved {savepath}")
