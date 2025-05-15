import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import LogNorm

# Constants
k_B = 1.380649e-16     # Boltzmann constant in erg/K
m_H = 1.6735575e-24     # Mass of hydrogen atom in g
mu = 0.6               # Mean molecular weight for fully ionized gas
gamma = 5.0 / 3.0       # Adiabatic index for monoatomic gas

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
        UnitLen = 3.085678e21     # cm per code length unit
        UnitVel = 1e5             # cm/s per code velocity unit

        # Load gas data
        rho_code = np.array(snap["PartType0/Density"]) # * a**3
        rho_cgs = rho_code * (UnitMass / UnitLen**3)

        u_code = np.array(snap["PartType0/InternalEnergy"])
        u_cgs = u_code * UnitVel**2  # erg/g

        # Compute temperature
        T = u_cgs * (gamma - 1.0) * mu * m_H / k_B

        # Plot
        fig, ax = plt.subplots(figsize=(8,7))
        h = ax.hist2d(
            np.log10(rho_cgs),
            np.log10(T),
            bins=200,
            cmap='plasma',
            norm=LogNorm()
        )


        ax.set_xlabel("log10(Density (g/cm³))")
        ax.set_ylabel("log10(Temperature (K))")
        snapname = os.path.basename(snapfile)
        ax.set_title(f"\u03c1–T\n{snapname} (a={a:.4f})")
        ax.grid(True, which="both", ls="--", alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Gas Count (log scale)')

        plt.tight_layout()
        savepath = os.path.join(output_dir, f"{snapname}.png")
        plt.savefig(savepath, dpi=150)
        plt.close(fig)

        print(f"[INFO] Saved {savepath}")
