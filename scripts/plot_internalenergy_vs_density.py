import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_redshift(snapshot_file):
    with h5py.File(snapshot_file, "r") as f:
        header = f["Header"]
        redshift = header.attrs["Redshift"]
    return redshift

def main(snapshot_file):
    redshift = get_redshift(snapshot_file)

    with h5py.File(snapshot_file, "r") as f:
        u = f["PartType0/InternalEnergy"][:]
        rho = f["PartType0/Density"][:]

    log_u = np.log10(u)
    log_rho = np.log10(rho)

    plt.figure(figsize=(8, 6))
    plt.scatter(log_rho, log_u, s=1, alpha=0.5, label="Gas Particles")

    hot_mask = u > 1e5
    if np.any(hot_mask):
        plt.scatter(np.log10(rho[hot_mask]), np.log10(u[hot_mask]), 
                    color='red', s=3, alpha=0.6, label="Hot Gas (u > 1e5)")

    plt.xlabel(r"log$_{10}$(Density)")
    plt.ylabel(r"log$_{10}$(Internal Energy)")
    plt.title(f"Internal Energy vs Density (z = {redshift:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_internal_energy_vs_density.py <snapshot_file>")
        sys.exit(1)
    
    snapshot_file = sys.argv[1]
    main("../output/"+snapshot_file)
