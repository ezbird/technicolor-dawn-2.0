import h5py
import numpy as np
import matplotlib.pyplot as plt

# Replace with your snapshot path
snapshot_path = "../output/snapshot_023.hdf5"

with h5py.File(snapshot_path, "r") as f:
    u = f["PartType0/InternalEnergy"][:]       # Specific internal energy [code units]
    rho = f["PartType0/Density"][:]            # Density [code units]

# Convert to log scale
log_u = np.log10(u)
log_rho = np.log10(rho)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(log_rho, log_u, s=1, alpha=0.5, label="Gas Particles")

# Highlight extreme-energy cases
hot_mask = u > 1e5
plt.scatter(np.log10(rho[hot_mask]), np.log10(u[hot_mask]), 
            color='red', s=3, alpha=0.6, label="Hot Gas (u > 1e5)")

plt.xlabel(r"log$_{10}$(Density)")
plt.ylabel(r"log$_{10}$(Internal Energy)")
plt.title("Internal Energy vs Density (Gas)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
