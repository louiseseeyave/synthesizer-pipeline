import numpy as np

from synthesizer.grid import Grid
from synthesizer.sed import Sed
from synthesizer.filters import FilterCollection
from synthesizer.load_data.load_simba import load_Simba


if __name__ == "__main__":

    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,100.0"
    grid_dir = "../../synthesizer_data/grids/"
    grid = Grid(grid_name, grid_dir=grid_dir, read_lines=False)

    # define a filter collection object
    fs = [f"SLOAN/SDSS.{f}" for f in ['u', 'g', 'r', 'i', 'z']]
    fs += ['GALEX/GALEX.FUV', 'GALEX/GALEX.NUV']
    fs += [f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']]
    fs += [f'2MASS/2MASS.{f}' for f in ['J', 'H', 'Ks']]
    fs += [f'HST/ACS_HRC.{f}'
           for f in ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']]
    fs += [f'HST/WFC3_IR.{f}'
           for f in ['F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']]

    tophats = {
        "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
        "UV2800": {"lam_eff": 2800, "lam_fwhm": 300},
    }

    fc = FilterCollection(
        filter_codes=fs, tophat_dict=tophats, new_lam=grid.lam
    )

    gals = load_Simba(
        directory=("/cosma7/data/dp004/dc-love2/codes/"
                   "simba_dusty_quiescent/data"),
        snap_name="snap_m100n1024_144.hdf5",
        caesar_name="Groups/m100n1024_144.hdf5",
    )

    # filter galaxies by stellar mass
    stellar_masses = np.array([np.log10(_g.stellar_mass.value)
                               for _g in gals])

    _specs = np.vstack(
        [
            _g.get_spectra_incident(
                grid,
                # tau_v_ISM=0.33,
                # tau_v_BC=0.67
            )._lnu
            for _g in gals[:10]
        ]
    )

    _specs = Sed(lam=grid.lam, lnu=_specs)
