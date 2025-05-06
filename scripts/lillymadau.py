import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

def process_folder(folder):
    """
    Reads Gadget-4 snapshots from 'folder', computes cosmic star formation rate density
    in Msun/yr/Mpc^3 for each snapshot.
    Returns arrays of redshift and SFR density.
    """
    # Find snapshot files
    snapshots = sorted(glob.glob(os.path.join(folder, "snap*.hdf5")))
    redshifts = []
    sfr_densities = []
    
    for snapfile in snapshots:
        with h5py.File(snapfile, 'r') as snap:
            header = snap['Header'].attrs
            a = header['Time']               # scale factor
            z = header['Redshift']
            box_code = header['BoxSize']     # code units
            # Read unit conversions from header if present, otherwise default
            UnitMass = header.get('UnitMass_in_g', 1.989e43)  # grams per code mass unit
            UnitTime = header.get('UnitTime_in_s', 3.15576e16)  # s per code time unit
            UnitLength = header.get('UnitLength_in_cm', 3.085678e21)  # cm per code length unit
            # Compute volume in Mpc^3
            box_cm = box_code * UnitLength
            box_Mpc = box_cm / (3.085678e24)
            volume_Mpc3 = box_Mpc**3
            
            # Sum SFR from gas particles
            sfr_code = snap['PartType0']['StarFormationRate'][:]  # code units: mass/time
            sfr_g_per_s = np.sum(sfr_code) * UnitMass / UnitTime  # grams per second
            sfr_Msun_per_yr = sfr_g_per_s / 1.989e33 * (UnitTime / (3600*24*365))
            
            # Cosmic SFR density
            sfr_density = sfr_Msun_per_yr / volume_Mpc3
            
            redshifts.append(z)
            sfr_densities.append(sfr_density)
    
    # Convert to numpy arrays and sort by decreasing redshift
    redshifts = np.array(redshifts)
    sfr_densities = np.array(sfr_densities)
    sort_idx = np.argsort(redshifts)[::-1]
    return redshifts[sort_idx], sfr_densities[sort_idx]

def main():
    parser = argparse.ArgumentParser(description="Plot Lilly-Madau SFR density from Gadget-4 snapshots.")
    parser.add_argument("folder1", help="Path to first snapshot folder")
    parser.add_argument("--folder2", help="Path to second snapshot folder (optional)", default=None)
    args = parser.parse_args()
    
    plt.figure()
    
    z1, s1 = process_folder(args.folder1)
    plt.semilogy(z1, s1, label=os.path.basename(args.folder1))
    
    if args.folder2:
        z2, s2 = process_folder(args.folder2)
        plt.semilogy(z2, s2, label=os.path.basename(args.folder2))
    
    plt.gca().invert_xaxis()
    plt.xlabel("Redshift")
    plt.ylabel("SFR Density (Msun/yr/Mpc^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
