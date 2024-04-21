import numpy as np
import h5py
import time
import argparse
import sys

import matplotlib.pyplot as plt
from unyt import Myr
from astropy.cosmology import Planck13

# from mpi4py import MPI

from synthesizer.grid import Grid
from synthesizer.sed import Sed, combine_list_of_seds
from synthesizer.filters import FilterCollection
from synthesizer.load_data.load_flares import load_FLARES
from synthesizer.kernel_functions import Kernel


def test(_gal, grid, age_pivot=None):

    if age_pivot:
        sed = _gal.stars.get_particle_spectra_incident(grid, old=age_pivot)
    else:
        sed = _gal.stars.get_particle_spectra_incident(grid)

    print('Successfully obtained SEDs')


def get_spectra(_gal, grid, age_pivot=10. * Myr):

    """
    Helper method for spectra generation

    Args:
        _gal (gal type)
        grid (grid type)
        age_pivot (float)
            split between young and old stellar populations, units Myr
    """

    
    # Skip over galaxies that have no stellar particles
    if _gal.stars.nstars==0:
        print('There are no stars in this galaxy.')
        return None

    spec = {}

    dtm = _gal.dust_to_metal_vijayan19()
    
    # Get young pure stellar spectra (integrated)
    young_spec = \
        _gal.stars.get_spectra_incident(grid, young=age_pivot)
    
    # Get pure stellar spectra for all old star particles
    old_spec_part = \
        _gal.stars.get_particle_spectra_incident(grid) # , old=age_pivot)
    
    # Sum and save old and young pure stellar spectra
    old_spec = old_spec_part.sum()

    spec['stellar'] = old_spec + young_spec

    return spec


def get_lum_weighted_metallicity(_gal, grid, filters, age_pivot=10*Myr):

    """
    Get UV luminosity of each stellar particle and use it to obtain
    the weighted metallicity of the galaxy.

    Args:
        _gal (gal type)
        grid (grid type)
    """
    
    # Skip over galaxies that have no stellar particles
    if _gal.stars.nstars==0:
        print('There are no stars in this galaxy.')
        return None

    print(f'There are {_gal.stars.nstars} stars in this galaxy.')
    
    # Get particle metallicity
    metallicity =_gal.stars.metallicities
    print('metallicity:', metallicity)
    print('ages:', _gal.stars.ages)

    # Check stellar ages
    ages = (_gal.stars.ages).to('Myr')
    old = ages > age_pivot
    print(f'{np.sum(old)} star particles above age pivot.')
    young = ages < age_pivot
    print(f'{np.sum(young)} star particles below age pivot.')

    # if my_rank==0:
    #     print('ending a')
    #     sys.exit()

    # Get pure stellar spectra for all star particles
    sed = _gal.stars.get_particle_spectra_incident(grid) #, old=age_pivot)
    
    # if my_rank==0:
    #     print('ending a')
    #     sys.exit()
    
    # Calculate the observed SED in nJy
    sed.get_fnu(Planck13, _gal.redshift, igm=False)
    
    # Get UV luminosity
    lum = \
        _gal.stars.particle_spectra["incident"].get_photo_luminosities(filters)
    luv = lum['UV1500']

    print(f'luv.shape: {luv.shape}')
    print(f'log10luv: {np.log10(luv)}')
    
    # Calculate UV luminosity-weighted metallicity
    weighted_metallicity = np.dot(metallicity, luv)/np.sum(luv)

    # print(f'Weighted metallicity: {weighted_metallicity.value}')

    return weighted_metallicity.value


def save_dummy_file(h5_file, region, tag, filters,
                    keys=['stellar']):

    """
    Save a dummy hdf5 file with the expected hdf5 structure but
    containing no data. Useful if a snapshot contains no galaxies.

    Args
        h5_file (str):
            file to be written
        region (str):
            region e.g. '00'
        tag (str):
            snapshot label e.g. '000_z015p000'
        filters:
            filter collection
        keys:
            spectra / dust models
    """

    with h5py.File(h5_file, 'w') as hf:
        
        # Use Region/Tag structure
        grp = hf.require_group(f'{region}/{tag}')

        # Loop through different spectra / dust models
        for key in keys:
            
            sbgrp = grp.require_group('SED')
            dset = sbgrp.create_dataset(f'{str(key)}', data=[])
            dset.attrs['Units'] = 'erg/(Hz*s)'


if __name__ == "__main__":

    # Set up MPI
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()
    if my_rank==0:
        print("World size: "+str(world_size))

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
        default = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5',
        # default = '/cosma7/data/dp004/dc-love2/codes/flares/data/flares.hdf5'
    )
    
    parser.add_argument(
        "-grid-name",
        type=str,
        required=False,
        help="Synthesizer grid file",
        default = "bpass-2.2.1-bin_chabrier03-0.1,300.0"
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
        new_lam=lam[ok_lam],
    )
    
    # kern = Kernel()

    # Save and load filter collection
    fc_fname = 'filter_collection_uv1500.hdf5'
    # fc = save_filters(fname=fc_fname)
    fc = FilterCollection(path=fc_fname)

    gals = load_FLARES(
        master_file=args.master_file,
        region=args.region,
        tag=args.tag,
    )

    n_gals = len(gals)
    if my_rank==0:
        print(f"Number of galaxies: {n_gals}")

    # If there are no galaxies in this snap, create dummy file
    if n_gals==0:
        if my_rank==0:
            print('No galaxies. Saving dummy file.')
            save_dummy_file(args.output, args.region, args.tag,
                            [f.filter_code for f in fc])
        sys.exit()

    # Divide workload between processors using number of stellar particles
    # Array to track number of stellar particles assigned to each processor
    ranks_npart = np.zeros(world_size)
    # Indices of the galaxy assigned to particular processor
    my_inds = []
    # Loop over every galaxy
    for ind in range(n_gals):
        # Get number of stellar particles in this galaxy
        gal = gals[ind]
        n_stars = len(gal.stars.initial_masses)
        # Get rank with lowest number of stellar particles
        min_ind = np.argmin(ranks_npart)
        # Assign galaxy to that rank
        ranks_npart[min_ind] += n_stars
        if min_ind==my_rank:
            my_inds.append(ind)

    # Print workload of each processor
    print(f'Rank {my_rank}: working on {len(my_inds)} galaxies')

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

    # dat = []
    lum_metallicity = np.array([])
    # young_metallicity = np.array([])

    # Loop over the galaxies allocated to rank
    # print('Getting spectra...')
    for gal_idx in my_inds:

        gal = gals[gal_idx]

        # Get spectra
        _spec = get_spectra(gal, grid=grid)
        if _spec==None:
            continue
        dat.append(_spec)

        # # Get UV luminosity-weighted metallicity
        # _lum_metallicity = \
        #     get_lum_weighted_metallicity(gal, grid=grid, filters=fc)
        # lum_metallicity = np.append(lum_metallicity, _lum_metallicity)

        # if my_rank==0:
        #     print('ending a')
        #     sys.exit()

        # # Get initial mass-weighted metallicity for young stars
        # _young_metallicity = get_young_mass_weighted_metallicity(gal)
        # young_metallicity = np.append(young_metallicity, _young_metallicity)

    # Collate galaxies on rank 0
    if my_rank==0:
        # world_dat = np.empty(n_gals, dtype=object)
        # world_dat[my_inds] = dat
        world_lum_metallicity = np.zeros(n_gals, dtype=np.float32)
        world_lum_metallicity[my_inds] = lum_metallicity
        # world_young_metallicity = np.zeros(n_gals, dtype=np.float32)
        # world_young_metallicity[my_inds] = young_metallicity
        for i in range(1, world_size):
            # rank_dat = world_comm.recv(source=i, tag=1)
            rank_inds = world_comm.recv(source=i, tag=2)
            # world_dat[rank_inds] = rank_dat
            rank_lum_metallicity = world_comm.recv(source=i, tag=3)
            world_lum_metallicity[rank_inds] = rank_lum_metallicity
            # rank_young_metallicity = world_comm.recv(source=i, tag=4)
            # world_young_metallicity[rank_inds] = rank_young_metallicity
        print(f'Collected data from all {len(world_dat)} galaxies.')
    else:
        # world_comm.send(dat, dest=0, tag=1)
        world_comm.send(my_inds, dest=0, tag=2)
        world_comm.send(lum_metallicity, dest=0, tag=3)
        # world_comm.send(young_metallicity, dest=0, tag=4)

    end = time.time()
    print(f'Spectra generation: {end - start:.2f}')

    # If there are no galaxies in this snap, create dummy file
    if my_rank==0:
        if len(world_dat)==0:
            print('Galaxies have no stellar particles. Saving dummy file.')
            save_dummy_file(args.output, args.region, args.tag,
                            [f.filter_code for f in fc])
            sys.exit()

    if my_rank==0:
        print('ending a')
        sys.exit()

    # Save data
    if my_rank==0:

        # Combine list of dicts into dict with single Sed objects
        specs = {}
        start = time.time()
        for key in dat[0].keys():
            specs[key] = combine_list_of_seds([_dat[key] for _dat in world_dat])
        end = time.time()
        print(f'Combining spectra: {end - start:.2f}')

        # Calculate photometry (observer frame fluxes and luminosities)
        # fluxes = {}
        # luminosities = {}
    
        # start = time.time()
        # for key in dat[0].keys():
            # specs[key].get_fnu(cosmo=Planck13, z=gals[0].redshift)
            # fluxes[key] = specs[key].get_photo_fluxes(fc)
            # luminosities[key] = specs[key].get_photo_luminosities(fc)

        # end = time.time()
        # print(f'Photometry calculation: {end - start:.2f}')

        # Save spectra, fluxes and luminosities
        print('Saving hdf5...')
        with h5py.File(args.output, 'w') as hf:

            # Use Region/Tag structure
            grp = hf.require_group(f'{args.region}/{args.tag}/Galaxy')
            
            # Loop through different spectra / dust models
            for key in dat[0].keys():

                sbgrp = grp.require_group('SED')
                # Include wavelength array corresponding to SEDs
                if key==list(dat[0].keys())[0]:
                    # We only want the UV continuum (1000-3500A)
                    lam_dat = specs[key].lam # Angstroms
                    # ok = (lam_dat > 1000) & (lam_dat < 3500)
                    # lam_dat = lam_dat[ok]
                    lam = sbgrp.create_dataset(f'wavelength', data=lam_dat)
                    lam.attrs['Units'] = str(specs[key].lam.units)
                # Add corresponding SED data
                sed_dat = specs[key].lnu
                # sed_dat = np.transpose(np.transpose(sed_dat)[ok])
                dset = sbgrp.create_dataset(f'{str(key)}', data=sed_dat)
                dset.attrs['Units'] = str(specs[key].lnu.units)

            # Add metallicities
            dset = grp.create_dataset(
                'Metallicity/UV1500WeightedStellarZ', data=world_lum_metallicity)
            dset.attrs['Units'] = 'Dimensionless'
            dset = grp.create_dataset(
                'Metallicity/YoungMassWeightedStellarZ', data=world_young_metallicity)
            dset.attrs['Units'] = 'Dimensionless'

            # Master file properties
            with h5py.File(args.master_file, 'r') as mf:

                # Add stellar properties

                # Properties in the master file to be copied over
                h5_dirs = [
                    'Galaxy/SFR_aperture/30',
                    'Galaxy/Mstar_aperture/30',
                    'Galaxy/Metallicity/MassWeightedStellarZ',
                ]

                for h5_dir in h5_dirs:
                    mf.copy(
                        mf[f"{args.region}/{args.tag}/{h5_dir}"],
                        hf,
                        f"{args.region}/{args.tag}/{h5_dir}"
                    )

    print('All done!')
 
