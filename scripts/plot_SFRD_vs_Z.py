import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --------- Configuration ---------
SNAPSHOT_FOLDER = "../output/"    # <-- Change this to wherever your snapshots are saved!
OUTPUT_FOLDER = "SFR"     # Folder to save the plots
# ----------------------------------

# Constants
gamma_minus_one = 5.0/3.0 - 1.0  # assuming ideal monoatomic gas
mean_molecular_weight = 0.6      # for ionized primordial gas
proton_mass_cgs = 1.6726e-24
boltzmann_cgs = 1.3806e-16

# Make output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Find all snapshot files
snapshot_files = sorted(glob.glob(os.path.join(SNAPSHOT_FOLDER, "snapshot_*.hdf5")))

# Initialize cumulative storage
all_redshifts = []
all_SFRs = []
all_Temps = []

for snap in snapshot_files:
    with h5py.File(snap, 'r') as f:
        header = f['Header'].attrs
        redshift = header['Redshift']
        
        if 'PartType0' not in f:
            continue
        
        gas = f['PartType0']
        
        if 'InternalEnergy' not in gas:
            continue
        
        u = gas['InternalEnergy'][:]
        
        # Check if ElectronAbundance exists
        if 'ElectronAbundance' in gas:
            ne = gas['ElectronAbundance'][:]
            mu = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * ne)
        else:
            mu = mean_molecular_weight
        
        # Convert Internal Energy to Temperature
        Temp = (gamma_minus_one) * u * mu * proton_mass_cgs / boltzmann_cgs
        
        mean_Temp = np.mean(Temp)
        
        # Star formation rate
        if 'StarFormationRate' in gas:
            sfr = gas['StarFormationRate'][:]
            mean_SFR = np.mean(sfr)
        else:
            mean_SFR = 0.0
        
        all_redshifts.append(redshift)
        all_SFRs.append(mean_SFR)
        all_Temps.append(mean_Temp)
        
        # --- Make the 2-panel plot ---
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        axs[0].plot(all_redshifts, all_SFRs, 'bo-')
        axs[0].set_xlabel('Redshift')
        axs[0].set_ylabel('Mean SFR [Msun/yr]')
        axs[0].invert_xaxis()
        axs[0].set_title('Mean Star Formation Rate')
        
        axs[1].plot(all_redshifts, all_Temps, 'ro-')
        axs[1].set_xlabel('Redshift')
        axs[1].set_ylabel('Mean Gas Temperature [K]')
        axs[1].invert_xaxis()
        axs[1].set_title('Mean Gas Temperature')
        
        plt.suptitle(f"Snapshot: {os.path.basename(snap)}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        outname = os.path.join(OUTPUT_FOLDER, os.path.basename(snap).replace('.hdf5', '.png'))
        plt.savefig(outname)
        plt.close()

print(f"Done! All plots saved in '{OUTPUT_FOLDER}/' folder.")
