import h5py
import numpy as np
import shutil

infile = "ics_arepo.hdf5"
outfile = "ics_from_music_32.hdf5"

# Copy the file so we don't modify the original
shutil.copy(infile, outfile)

with h5py.File(outfile, "r+") as f:
    header = f["Header"].attrs

    def pad_array(name, target_len=7):
        if name in header:
            data = header[name]
            if len(data) < target_len:
                padded = np.zeros(target_len, dtype=data.dtype)
                padded[:len(data)] = data
                header[name] = padded
                print(f"Padded {name}: {data} → {padded}")
            elif len(data) > target_len:
                header[name] = data[:target_len]
                print(f"Trimmed {name} to {target_len} entries.")

    pad_array("NumPart_ThisFile")
    pad_array("NumPart_Total")
    pad_array("NumPart_Total_HighWord")
    pad_array("MassTable")


# MUSIC creates the masses in physical units (solar masses) and Gadget is expecting its internal units 10^10 Msolar/h
mass_unit_scale = 10**9  # Convert M☉ to 1e10 M☉/h

with h5py.File(outfile, "r+") as f:
    mt = f['Header'].attrs['MassTable']
    fixed = mt * mass_unit_scale
    f['Header'].attrs['MassTable'] = fixed
    print(f"Fixed MassTable: {mt} → {fixed}")

# There is no InternalEnergy for the gas, so cooling breaks. Let's add it in...
u_gas = 200  # Internal energy per unit mass, Gadget units

with h5py.File(outfile, "r+") as f:
    if 'PartType0' in f:
        N = f['PartType0']['Coordinates'].shape[0]
        if 'InternalEnergy' in f['PartType0']:
            del f['PartType0']['InternalEnergy']  # overwrite safely
        f['PartType0'].create_dataset('InternalEnergy', data=np.full(N, u_gas))
        print(f"Assigned InternalEnergy = {u_gas:.2e} to {N} gas particles")


