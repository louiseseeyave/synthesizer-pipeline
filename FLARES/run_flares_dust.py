import numpy as np
import h5py
import time
from functools import partial

import matplotlib.pyplot as plt
from unyt import Myr

from schwimmbad import MultiPool

from synthesizer.grid import Grid
from synthesizer.sed import Sed
from synthesizer.filters import FilterCollection
from synthesizer.load_data.load_flares import load_FLARES
from synthesizer.kernel_functions import kernel
from synthesizer.conversions import lnu_to_absolute_mag
from synthesizer.dust.attenuation import PowerLaw


if __name__ == "__main__":

    master_file = '/cosma7/data/dp004/dc-love2/codes/flares/data/flares.hdf5'

    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c17.03"
    grid_dir = "../../synthesizer_data/grids/"
    grid = Grid(grid_name, grid_dir=grid_dir, read_lines=False)

    kern = kernel()

    # define a filter collection object
    fs = [f"SLOAN/SDSS.{f}" for f in ['u', 'g', 'r', 'i', 'z']]
    # fs += ['GALEX/GALEX.FUV', 'GALEX/GALEX.NUV']
    # fs += [f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']]
    # fs += [f'2MASS/2MASS.{f}' for f in ['J', 'H', 'Ks']]
    # fs += [f'HST/ACS_HRC.{f}'
    #        for f in ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']]
    # fs += [f'HST/WFC3_IR.{f}'
    #        for f in ['F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']]

    tophats = {
        "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
        "UV2800": {"lam_eff": 2800, "lam_fwhm": 300},
    }

    fc = FilterCollection(
        filter_codes=fs,
        tophat_dict=tophats,
        new_lam=grid.lam
    )

    gals = load_FLARES(
        master_file=master_file,
        region='00',
        tag='010_z005p000',
    )

    def get_spectra(_gal, grid, fc, age_pivot=10. * Myr):
        """
        Helper method for spectra generation

        Args:
            _gal (gal type)
            grid (grid type)
            fc (FilterCollection type)
            age_pivot (float)
                split between young and old stellar populations, units Myr
        """

        spec = {}

        # Get nebular spectra for each star particle
        young_spec_part = \
            _gal.stars.get_particle_spectra_nebular(grid, young=age_pivot)
        old_spec_part = \
            _gal.stars.get_particle_spectra_incident(grid, old=age_pivot)

        # Sum up integrated spectra
        young_spec = young_spec_part.sum()
        old_spec = old_spec_part.sum()
        
        # Save intrinsic stellar spectra
        spec['incident'] = young_spec + old_spec

        # Simple screen model
        spec['screen'] = spec['incident'].apply_attenuation(tau_v=0.33)
        
        # Charlot & Fall attenuation model
        young_spec_attenuated = young_spec.apply_attenuation(tau_v=0.33 + 0.67)
        old_spec_attenuated = old_spec.apply_attenuation(tau_v=0.33)
        spec['CF00'] = young_spec_attenuated + old_spec_attenuated

        # Gamma model (modified version of Lovell+19)
        gamma = _gal.screen_dust_gamma_parameter()

        young_spec_attenuated = young_spec.apply_attenuation(
            tau_v=gamma * (0.33 + 0.67)
        )
        old_spec_attenuated = old_spec.apply_attenuation(
            tau_v=gamma * 0.33
        )

        spec['gamma'] = young_spec_attenuated + old_spec_attenuated

        # LOS model (Vijayan+21)
        tau_v = _gal.calculate_los_tau_v(kappa=0.0795, kernel=kern.get_kernel(), force_loop=True)
        
        plt.hist(tau_v)
        plt.show()

        young_spec_attenuated = young_spec_part.apply_attenuation(
            tau_v=tau_v + (_gal.stars.metallicities / 0.01)
        )
        old_spec_attenuated = old_spec_part.apply_attenuation(tau_v=tau_v)

        spec['los'] = young_spec_attenuated.sum() + old_spec_attenuated.sum()

        return spec


    spec = get_spectra(gals[0], grid, fc)

    # Debugging plot
    # plt.loglog(young_spec.lam, young_spec.lnu, label='young')
    # plt.loglog(old_spec.lam, old_spec.lnu, label='old')
    plt.loglog(spec['incident'].lam, spec['incident'].lnu, label='combined')
    plt.loglog(spec['screen'].lam, spec['screen'].lnu, label='screen')
    plt.loglog(spec['CF00'].lam, spec['CF00'].lnu, label='CF00')
    plt.loglog(spec['gamma'].lam, spec['gamma'].lnu, label='gamma')
    plt.loglog(spec['los'].lam, spec['los'].lnu, label='los')
    plt.xlim(1e1, 9e4)
    # plt.ylim(1e27,1e36)
    plt.legend()
    plt.show()
 
    # start = time.time()
    # 
    # _f = partial(get_spectra, grid=grid, fc=fc)
    # with MultiPool(4) as pool:
    #     dat = pool.map(_f, gals)

    # end = time.time()
    # print(f'{end - start:.2f}')

    # with h5py.File('simba.hdf5', 'w') as hf:
    #     hf.create_group('spectra')
    #     hf.create_dataset('spectra/young_stellar', data=np.array(young_spec.lnu))
    #     hf.create_dataset('spectra/old_stellar', data=np.array(old_spec.lnu))
    #     hf.create_dataset('spectra/wavelength', data=np.array(young_spec.lam))
    #     # hf.create_dataset('mags', data=np.array(mags))

