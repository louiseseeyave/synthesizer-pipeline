import numpy as np
import h5py
import time
import argparse
from functools import partial

import matplotlib.pyplot as plt
from unyt import Myr
from astropy.cosmology import Planck13

from schwimmbad import MultiPool

from synthesizer.grid import Grid
from synthesizer.sed import Sed, combine_list_of_seds
from synthesizer.filters import FilterCollection
from synthesizer.load_data.load_flares import load_FLARES
from synthesizer.kernel_functions import Kernel
from synthesizer.conversions import lnu_to_absolute_mag
from synthesizer.dust.attenuation import PowerLaw


def get_spectra(_gal, grid, age_pivot=10. * Myr):
    """
    Helper method for spectra generation

    Args:
        _gal (gal type)
        grid (grid type)
        age_pivot (float)
            split between young and old stellar populations, units Myr
    """

    spec = {}

    dtm = _gal.dust_to_metal_vijayan19()
    
    # Get young pure stellar spectra (integrated)
    young_spec = \
        _gal.stars.get_spectra_incident(grid, young=age_pivot)
    
    # Get pure stellar spectra for all old star particles
    old_spec_part = \
        _gal.stars.get_particle_spectra_incident(grid, old=age_pivot)
    
    # Sum and save old and young pure stellar spectra
    old_spec = old_spec_part.sum()

    spec['stellar'] = old_spec + young_spec

    # Get nebular spectra for each star particle
    young_neb_spec_part = \
        _gal.stars.get_particle_spectra_nebular(grid, young=age_pivot)

    # Sum and save intrinsic stellar spectra
    young_neb_spec = young_neb_spec_part.sum()

    
    # Save intrinsic stellar spectra
    spec['intrinsic'] = young_neb_spec + old_spec

    # Simple screen model
    spec['screen'] = spec['intrinsic'].apply_attenuation(tau_v=0.33)
    
    # Charlot & Fall attenuation model
    young_spec_attenuated = young_neb_spec.apply_attenuation(tau_v=0.33 + 0.67)
    old_spec_attenuated = old_spec.apply_attenuation(tau_v=0.33)
    spec['CF00'] = young_spec_attenuated + old_spec_attenuated

    # Gamma model (modified version of Lovell+19)
    gamma = _gal.screen_dust_gamma_parameter()

    young_spec_attenuated = young_neb_spec.apply_attenuation(
        tau_v=gamma * (0.33 + 0.67)
    )
    old_spec_attenuated = old_spec.apply_attenuation(
        tau_v=gamma * 0.33
    )

    spec['gamma'] = young_spec_attenuated + old_spec_attenuated

    # LOS model (Vijayan+21)
    tau_v = _gal.calculate_los_tau_v(kappa=0.0795, kernel=kern.get_kernel(), force_loop=True)
    
    # plt.hist(np.log10(tau_v))
    # plt.show()

    young_spec_attenuated = young_neb_spec_part.apply_attenuation(
        tau_v=tau_v + (_gal.stars.metallicities / 0.01)
    )
    old_spec_attenuated = old_spec_part.apply_attenuation(tau_v=tau_v)

    spec['los'] = young_spec_attenuated.sum() + old_spec_attenuated.sum()

    return spec

def set_up_filters():
    
    # define a filter collection object
    fs = [f"SLOAN/SDSS.{f}" for f in ['u', 'g', 'r', 'i', 'z']]
    fs += ['GALEX/GALEX.FUV', 'GALEX/GALEX.NUV']
    fs += [f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']]
    fs += [f'2MASS/2MASS.{f}' for f in ['J', 'H', 'Ks']]
    fs += [f'HST/ACS_HRC.{f}'
           for f in ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']]
    fs += [f'HST/WFC3_IR.{f}'
           for f in ['F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']]

    fs += [f'JWST/NIRCam.{f}' 
            for f in [
                'F070W', 'F090W', 'F115W', 'F140M', 'F150W',
                'F162M', 'F182M', 'F200W', 'F210M', 'F250M',
                'F277W', 'F300M', 'F356W', 'F360M', 'F410M',
                'F430M', 'F444W', 'F460M', 'F480M']]
    
    fs += [f'JWST/MIRI.{f}' 
            for f in [
                'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W',
                'F2100W', 'F2550W', 'F560W', 'F770W']]

    tophats = {
        "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
        "UV2800": {"lam_eff": 2800, "lam_fwhm": 300},
    }

    fc = FilterCollection(
        filter_codes=fs,
        tophat_dict=tophats,
        new_lam=grid.lam
    )

    return fc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the synthesizer FLARES pipeline")

    rg = "39"
    snap = "010_z005p000"

    parser.add_argument(
        "-region",
        type=str,
        help="FLARES region",
        default = rg
    )
    
    parser.add_argument(
        "-tag",
        type=str,
        help="FLARES snapshot tag",
        default = snap
    )
    
    parser.add_argument(
        "-master-file",
        type=str,
        required=False,
        help="FLARES master file",
        default = '/cosma7/data/dp004/dc-love2/codes/flares/data/flares.hdf5'
    )
    
    parser.add_argument(
        "-grid-name",
        type=str,
        required=False,
        help="Synthesizer grid file",
        default = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03"
    )
    
    parser.add_argument(
        "-grid-directory",
        type=str,
        required=False,
        help="Synthesizer grid directory",
        default = "/cosma7/data/dp004/dc-seey1/modules/synthesizer-sam/grids/"
    )
    
    parser.add_argument(
        "-output",
        type=str,
        required=False,
        help="Output file",
        default = f"./flares_photometry_{rg}_{snap}.hdf5"
    )
    
    parser.add_argument(
        "-nprocs",
        type=int,
        required=False,
        help="Number of threads",
        default = 1
        # default = 10
    )

    args = parser.parse_args()

    grid = Grid(
        args.grid_name,
        grid_dir=args.grid_directory,
        read_lines=False
    )

    kern = Kernel()

    # fc = set_up_filters()
    fc = FilterCollection(path="filter_collection.hdf5")

    gals = load_FLARES(
        master_file=args.master_file,
        region=args.region,
        tag=args.tag,
    )

    print(f"Number of galaxies: {len(gals)}")

    # spec = get_spectra(gals[100], grid, fc)

    # Debugging plot
    # plt.loglog(young_spec.lam, young_spec.lnu, label='young')
    # plt.loglog(old_spec.lam, old_spec.lnu, label='old')
    # plt.loglog(spec['incident'].lam, spec['incident'].lnu, label='combined')
    # plt.loglog(spec['screen'].lam, spec['screen'].lnu, label='screen')
    # plt.loglog(spec['CF00'].lam, spec['CF00'].lnu, label='CF00')
    # plt.loglog(spec['gamma'].lam, spec['gamma'].lnu, label='gamma')
    # plt.loglog(spec['los'].lam, spec['los'].lnu, label='los')
    # plt.xlim(1e1, 9e4)
    # plt.ylim(1e25,1e34)
    # plt.legend()
    # plt.show()
 
    start = time.time()

    # Get list of ParticleGalaxy objects
    _f = partial(get_spectra, grid=grid)
    with MultiPool(args.nprocs) as pool:
        dat = pool.map(_f, gals)

    print('Got list of ParticleGalaxy objects for this region and snap.')
    print('Each object has the following keys:')
    print(dat[0].keys())

    
    # Combine list of dicts into dict with single Sed objects
    specs = {}
    for key in dat[0].keys():
        specs[key] = combine_list_of_seds([_dat[key] for _dat in dat])

    end = time.time()
    print(f'Spectra generation: {end - start:.2f}')

    print('Spec dictionary has the following keys:')
    print(specs.keys())
    

    # Calculate photometry (observer frame fluxes and luminosities)
    fluxes = {}
    luminosities = {}
    
    start = time.time()
    for key in dat[0].keys():
        print(f'Key: {key}')
        specs[key].get_fnu(cosmo=Planck13, z=gals[0].redshift)
        fluxes[key] = specs[key].get_photo_fluxes(fc)
        print(f'fluxes[key].photometry: shape {np.array(fluxes[key].photometry).shape}')
        print(f'fluxes[key].filters: {[f.filter_code for f in fluxes[key].filters]}')
        # print(fluxes[key].photometry)
        luminosities[key] = specs[key].get_photo_luminosities(fc)
        print(f'luminosities[key].photometry: shape {np.array(luminosities[key].photometry).shape}')
        print(f'luminosities[key].filters: {[f.filter_code for f in luminosities[key].filters]}')
        # print(luminosities[key].photometry)

    end = time.time()
    print(f'Photometry calculation: {end - start:.2f}')

    # Save spectra, fluxes and luminosities
    with h5py.File(args.output, 'w') as hf:

        # Use Region/Tag structure
        grp = hf.require_group(f'{args.region}/{args.tag}')

        # Loop through different spectra / dust models
        for key in dat[0].keys():
            sbgrp = grp.require_group('SED')
            dset = sbgrp.create_dataset(f'{str(key)}', data=specs[key].lnu)
            dset.attrs['Units'] = str(specs[key].lnu.units)

            sbgrp = grp.require_group('Fluxes')
            # Create separate groups for different instruments
            for i, f in enumerate(fluxes[key].filters):
                dset = sbgrp.create_dataset(
                    f'{str(key)}/{f.filter_code}',
                    data=fluxes[key][f.filter_code]
                    # data=fluxes[key].photometry[:,i]
                )

                dset.attrs['Units'] = str(fluxes[key].photometry.units)


            sbgrp = grp.require_group('Luminosities')
            # Create separate groups for different instruments
            for i, f in enumerate(luminosities[key].filters):
                dset = sbgrp.create_dataset(
                    f'{str(key)}/{f.filter_code}',
                    data=luminosities[key][f.filter_code]
                    # data=luminosities[key].photometry[:,i]
                )

                dset.attrs['Units'] = str(luminosities[key].photometry.units)

