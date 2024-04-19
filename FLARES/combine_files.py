import h5py
import numpy as np

regions = np.array(['00','01','02','03','04','05','06',
                    '07','08','09','10','11','12','13',
                    '14','15','16','17','18','19','20',
                    '21','22','23','24','25','26','27',
                    '28','29','30','31','32','33','34',
                    '35','36','37','38','39'])

snaps = ["000_z015p000", "001_z014p000", "002_z013p000", "003_z012p000",
         "004_z011p000", "005_z010p000" , "006_z009p000", "007_z008p000",
         "008_z007p000" , "009_z006p000", "010_z005p000"]

data_dir = "/cosma7/data/dp004/dc-seey1/modules/synthesizer-pipeline/FLARES/data_indices"

master_f = h5py.File(f"data_indices/flares_photometry_bpass-v2p3_no-cloudy_alpha-0p6.hdf5", 'a')

for rg in regions:
    for snap in snaps:

        # fname = f"flares_photometry_indices_{region}_{snap}.hdf5"
        fname = f"flares_photometry_bpass-v2p3_no-cloudy_alpha-0p6_{rg}_{snap}.hdf5"
        
        snap_f = h5py.File(f"{data_dir}/{fname}", 'r')
        
        snap_f.copy(snap_f[f"{rg}/{snap}"], master_f, f"{rg}/{snap}")
