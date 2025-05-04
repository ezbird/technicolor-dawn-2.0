'''
  Makes header arrays to be length 7 (for dust, PartType6)

  Converts MassTable from physical units to internal units (10^10 M_solar/h)

  Adds a missing InternalEnergy field to gas particles (with a hardcoded value of 200).
'''

import h5py
import numpy as np
import shutil

# Input and output file paths
infile = "IC_gadget3_32_100kpc.hdf5"
outfile = "IC_gadget4_32_100kpc.hdf5"

# Add a uniform InternalEnergy value to gas particles (PartType0)
# A value of 5 would be T = 5 × (5.0 × 10^-10) × (2/3) × 1.67 × 10^-24 × 1.22 / 1.38 × 10^-16 ≈ 50,000K
# A value of 0.01 would be ~100K
# A value of 0.1 would be ~1000K
# Aye! or, 200 should be about 10^4 Kelvin, ideal for starts a sim
u_gas = 200  # Internal energy in Gadget units

# Create a copy of the input file so we don't modify the original
shutil.copy(infile, outfile)

# Open the copied HDF5 file for modification
with h5py.File(outfile, "r+") as f:
    header = f["Header"].attrs

    def pad_array(name, target_len=7):
        """Ensure header arrays have exactly 7 elements (1 per PartType).
        Pad with zeros or truncate as needed."""
        if name in header:
            data = header[name]
            if len(data) < target_len:
                padded = np.zeros(target_len, dtype=data.dtype)
                padded[:len(data)] = data
                header[name] = padded
                print(f"Padded {name}: {data} → {padded}")
            elif len(data) > target_len:
                trimmed = data[:target_len]
                header[name] = trimmed
                print(f"Trimmed {name}: {data} → {trimmed}")

    # Pad the common header attributes
    pad_array("NumPart_ThisFile")
    pad_array("NumPart_Total")
    pad_array("NumPart_Total_HighWord")
    pad_array("MassTable")

# Convert MassTable from solar masses to Gadget internal units: 10^10 M_sun/h
mass_unit_scale = 1    # 1e9  # Convert M_sun → 10^10 M_sun/h
                       # This is only needed if we change the param: UnitLength_in_cm         3.085678e21        ;  Mpc / h is 3.085678e24; 1.0 kpc / h is 3.085678e21


with h5py.File(outfile, "r+") as f:
    mt = f['Header'].attrs['MassTable']
    fixed = mt * mass_unit_scale
    f['Header'].attrs['MassTable'] = fixed
    print(f"Fixed MassTable: {mt} → {fixed}")

with h5py.File(outfile, "r+") as f:
    if 'PartType0' in f:
        N = f['PartType0']['Coordinates'].shape[0]
        # Remove existing InternalEnergy if present
        if 'InternalEnergy' in f['PartType0']:
            del f['PartType0']['InternalEnergy']
        # Add uniform internal energy to all gas particles
        f['PartType0'].create_dataset('InternalEnergy', data=np.full(N, u_gas))
        print(f"Assigned InternalEnergy = {u_gas:.2e} to {N} gas particles")