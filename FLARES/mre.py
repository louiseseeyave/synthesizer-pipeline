import numpy as np
import h5py
import time
import argparse
import sys

import matplotlib.pyplot as plt
from unyt import Myr, Msun, yr, angstrom
from astropy.cosmology import Planck13

from synthesizer.grid import Grid
from synthesizer.sed import Sed, combine_list_of_seds
from synthesizer.filters import FilterCollection
from synthesizer.load_data.load_flares import load_FLARES
from synthesizer.kernel_functions import Kernel
from synthesizer.particle.galaxy import Galaxy


def test(_gal, grid, filters, age_pivot=None):

    if age_pivot:
        sed = _gal.stars.get_particle_spectra_incident(grid, old=age_pivot)
    else:
        sed = _gal.stars.get_particle_spectra_incident(grid, grid_assignment_method="cic")

    print('Successfully obtained SEDs')

    # Calculate the observed SED in nJy
    sed.get_fnu(Planck13, _gal.redshift, igm=False)
    
    # Get UV luminosity
    lum = _gal.stars.particle_spectra["incident"].get_photo_luminosities(filters)
    luv = lum['UV1500']

    # print(f'luv.shape: {luv.shape}')
    print(f'log10luv: {np.log10(luv)}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the synthesizer FLARES pipeline")

    parser.add_argument(
        "region",
        type=str,
        help="FLARES region",
        default='39',
    )
    
    parser.add_argument(
        "tag",
        type=str,
        help="FLARES snapshot tag",
        default='005_z010p000',
    )
    
    parser.add_argument(
        "-master-file",
        type=str,
        required=False,
        help="FLARES master file",
        # default = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5',
        default = '/cosma7/data/dp004/dc-love2/codes/flares/data/flares.hdf5'
    )
    
    parser.add_argument(
        "-grid-name",
        type=str,
        required=False,
        help="Synthesizer grid file",
        default = "bpass-2.2.1-bin_chabrier03-0.1,300.0"
        # default = "test_grid"
        # default = "bpass-2.3-bin_chabrier03-0.1,300.0_alpha0.0"
    )
    
    parser.add_argument(
        "-grid-directory",
        type=str,
        required=False,
        help="Synthesizer grid directory",
        default = "../../synthesizer-sam/grids/"
    )
    
    parser.add_argument(
        "-output",
        type=str,
        required=False,
        help="Output file",
        default = f"./flares_photometry_test.hdf5"
    )
    
    parser.add_argument(
        "-nprocs",
        type=int,
        required=False,
        help="Number of threads",
        default = 1
    )

    args = parser.parse_args()

    grid = Grid(
        args.grid_name,
        grid_dir=args.grid_directory,
        read_lines=False,
    )
    
    # We only want the UV continuum (1000-3500A)
    lam = grid.lam
    ok_lam = (lam> 1000) & (lam < 3500)
    print(f'Wavelength arr had {len(lam)} elements.')
    print(f'Now it has {len(lam[ok_lam])} elements.')
    print(f'Arr: {[lam[ok_lam]]}')
    
    grid = Grid(
        args.grid_name,
        grid_dir=args.grid_directory,
        read_lines=False,
        lam_lims=(1000 * angstrom, 3500 * angstrom),
        # new_lam=lam[ok_lam],
    )
    
    fc_fname = '/cosma7/data/dp004/dc-seey1/modules/synthesizer-pipeline/FLARES/filter_collection_uv1500.hdf5'
    fc = FilterCollection(path=fc_fname)

    gals = load_FLARES(
        master_file=args.master_file,
        region=args.region,
        tag=args.tag,
    )

    n_gals = len(gals)
    print(f"Number of galaxies: {n_gals}")

    # If there are no galaxies in this snap, create dummy file
    if n_gals==0:
        print('No galaxies. Saving dummy file.')
        save_dummy_file(args.output, args.region, args.tag,
                        [f.filter_code for f in fc])
        sys.exit()

    # Loop over galaxies
    for i, gal in enumerate(gals):

        print('------------------------------------------')
        print(f'Galaxy {i}:')
        print(f'Contains {gal.stars.nstars} stellar particles')
        print(f'Metallicities: {gal.stars.metallicities}')
        ages = (gal.stars.ages).to('Myr')
        print(f'Ages: {ages}')
        print(f'Initial masses: {gal.stars.initial_masses}')
        age_pivot = 4. * Myr
        # print(f'With pivot old = {age_pivot}...')
        # old = ages > age_pivot
        # print(f'{np.sum(old)} star particles above age pivot.')
        # young = ages < age_pivot
        # print(f'{np.sum(young)} star particles below age pivot.')
        # test(gal, grid, fc, age_pivot=age_pivot)
        print('Without pivot...')
        test(gal, grid, fc, age_pivot=None)
        if i==0:
            break

    # Identify problematic particles
    print('------------------------------------------')
    print('Now testing a galaxy with a single stellar particle below the age pivot:')
    galaxy = Galaxy(redshift=gal.redshift)
    # galaxy.load_stars(
    #     np.array([1859402.40044147]) * Msun,
    #     np.array([2.31792238*10**6]) * yr,
    #     np.array([0.0013345]),
    # )
    # galaxy.load_stars(
    #     np.array([1869298.57591167]) * Msun,
    #     np.array([3.12331021*10**6]) * yr,
    #     np.array([1.03978673e-04]),
    # )
    galaxy.load_stars(
        np.array([1869298.57591167]) * Msun,
        np.array([0.43335279*10**6]) * yr,
        np.array([7.00157660e-04]),
    )
    test(galaxy, grid, fc, age_pivot=None)
