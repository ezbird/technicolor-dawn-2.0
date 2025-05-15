import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm

# Constants
UnitMass = 1.989e43     # g (≈1e10 M☉)
UnitLen  = 3.085678e21  # cm (1 kpc)
k_B      = 1.380649e-16 # erg/K
m_H      = 1.6735575e-24# g
mu       = 0.6
gamma    = 5.0/3.0
G        = 6.674e-8     # cm³ g⁻¹ s⁻²

snapshots = sorted(glob.glob("../output/snapshot_*.hdf5"))
for snapfile in snapshots:
    with h5py.File(snapfile, 'r') as f:
        hdr = dict(f['Header'].attrs)
        params = dict(f['Parameters'].attrs)
        a   = hdr['Time']

        # --- dump SF parameters ---
        print(f"\n=== {snapfile} ===")
        print("CritPhysDensity:", hdr.get("CritPhysDensity", 0.0))
        print("PhysDensThresh:  ", hdr.get("PhysDensThresh",   0.0))
        print("CritOverDensity: ", hdr.get("CritOverDensity",  0.0))

        # --- compute physical density (g/cm³) ---
        rho_comov   = f['PartType0/Density'][...]
        rho_phys_cu = rho_comov / a**3
        rho_cgs     = rho_phys_cu * (UnitMass / UnitLen**3)

        # --- compute temperature (K) ---
        u_code = f['PartType0/InternalEnergy'][...]       # code units
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
        if cod > 0:
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
        ax.set_title(f'ρ–T   {snapfile.split("/")[-1]}   (a={a:.4f})')
        ax.legend(loc='upper left', fontsize='small')

        cbar = plt.colorbar(hist[3], ax=ax)
        cbar.set_label('Gas Count (log scale)')

        plt.tight_layout()
        plt.show()
