import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm

# Constants
k_B     = 1.380649e-16      # erg/K
m_H     = 1.6735575e-24     # g
mu      = 0.6               # mean molecular weight
gamma   = 5.0 / 3.0         # adiabatic index
G       = 6.674e-8          # cm^3 g^-1 s^-2

# Gather snapshots
snapshots = sorted(glob.glob("../output/snapshot_*.hdf5"))

# Loop
for snapfile in snapshots:
    with h5py.File(snapfile, "r") as snap:
        hdr = dict(snap["Header"].attrs)
        params = dict(snap["Parameters"].attrs)
        a   = hdr["Time"]

        # code → physical density conversion
        UnitMass = 1.989e43         # g
        UnitLen  = 3.085678e21      # cm   (1 kpc)
        rho_comov_code = snap["PartType0/Density"][...]
        rho_phys_code  = rho_comov_code / a**3
        rho_cgs        = rho_phys_code * (UnitMass/UnitLen**3)

        # internal energy → temperature
        u_code = snap["PartType0/InternalEnergy"][...]
        u_cgs  = u_code * (1e5)**2          # UnitVel=1e5 cm/s
        T = u_cgs*(gamma-1.0)*mu*m_H/k_B

        # --- compute thresholds in physical cgs ---
        # 1) direct physical thresh from CritPhysDensity (if >0)
        CritPhy = hdr.get("CritPhysDensity", 0.0)    # g/cm3

        # 2) code-unit thresh from PhysDensThresh
        PhysDT  = hdr.get("PhysDensThresh", 0.0)     # in code density units
        thr_from_PhysDT = PhysDT * (UnitMass/UnitLen**3) / a**3

        # 3) overdensity thresh → physical
        #    ρ_b,0 = OmegaBaryon * ρ_crit,0
        H0_cgs     = params["HubbleParam"]*100*1e5 / (1000*UnitLen)  # s^-1
        rho_crit0  = 3*H0_cgs**2/(8*np.pi*G)
        rho_b0     = params["OmegaBaryon"] * rho_crit0
        OD        = hdr.get("CritOverDensity", 0.0)
        thr_from_OD = OD * rho_b0 / a**3

        # --- optional: mask out SF gas (uncomment if you want) ---
        # sfr = snap["PartType0/StarFormationRate"][...]
        # mask = sfr == 0
        # rho_plot = rho_cgs[mask]
        # T_plot   = T[mask]

        # otherwise plot everything
        rho_plot = rho_cgs
        T_plot   = T

        # --- make the plot ---
        fig, ax = plt.subplots(figsize=(8,7))
        h = ax.hist2d(np.log10(rho_plot), np.log10(T_plot),
                      bins=200, cmap="plasma", norm=LogNorm())

        # overplot the three thresholds
        lines = [
            (np.log10(CritPhy),      "CritPhysDensity", "k--"),
            (np.log10(thr_from_PhysDT),"PhysDensThresh","w--"),
            (np.log10(thr_from_OD),   "CritOverDensity","gray--"),
        ]
        for x, label, style in lines:
            if np.isfinite(x):
                ax.axvline(x, ls=style, label=label)

        ax.set_xlabel("log10(Density [g/cm³])")
        ax.set_ylabel("log10(Temperature [K])")
        fname = snapfile.split("/")[-1]
        ax.set_title(f"\u03C1–T   {fname}   (a={a:.4f})")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="upper left")

        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label("Gas Count (log scale)")

        plt.tight_layout()
        out = f"rho_T_frames/{fname}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {out}")
