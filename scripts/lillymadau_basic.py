import h5py
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def get_sfr_density(snapshot_file):
    with h5py.File(snapshot_file, 'r') as f:
        header = f['Header'].attrs
        redshift = header.get('Redshift', header.get('REDSHIFT', None))
        # Attempt to read SFR from gas or star particles
        try:
            sfr = f['PartType0']['StarFormationRate'][()]
        except KeyError:
            sfr = f['PartType4']['StarFormationRate'][()]
        boxsize = header.get('BoxSize', None)
        volume = boxsize**3 if boxsize is not None else np.nan
        sfr_density = np.sum(sfr) / volume
    return redshift, sfr_density

def load_sfr_density(folder):
    files = sorted(glob.glob(os.path.join(folder, 'snapshot_*.hdf5')))
    data = []
    for f in files:
        try:
            data.append(get_sfr_density(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if data:
        z, rho = zip(*data)
        return np.array(z), np.array(rho)
    else:
        return np.array([]), np.array([])

# Paths to your two runs:
folder1 = '~/gadget-3.27/output'
folder2 = '../output'

z1, rho1 = load_sfr_density(folder1)
z2, rho2 = load_sfr_density(folder2)

plt.figure()
plt.plot(z1, rho1, label='Run 1')
plt.plot(z2, rho2, label='Run 2')
plt.xlabel('Redshift (z)')
plt.ylabel('Cosmic SFR Density [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.show()

