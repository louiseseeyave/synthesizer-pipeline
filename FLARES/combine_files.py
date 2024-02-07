import h5py
import numpy as np

region = "25"

snaps = ["000_z015p000", "001_z014p000", "002_z013p000", "003_z012p000",
         "004_z011p000", "005_z010p000" , "006_z009p000", "007_z008p000",
         "008_z007p000" , "009_z006p000", "010_z005p000"]

data_dir = "/cosma7/data/dp004/dc-seey1/modules/synthesizer-pipeline/FLARES/data_indices"

master_f = h5py.File(f"data_indices/flares_photometry_indices_{region}.hdf5", 'a')

for snap in snaps:

    fname = f"flares_photometry_indices_{region}_{snap}.hdf5"

    snap_f = h5py.File(f"{data_dir}/{fname}", 'r')
    
    snap_f.copy(snap_f[f"{region}/{snap}"], master_f, f"{region}/{snap}")
