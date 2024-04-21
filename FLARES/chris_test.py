import numpy as np
from synthesizer.grid import Grid
from synthesizer.particle.stars import Stars

grid_dir = "/cosma7/data/dp004/dc-seey1/modules/synthesizer-sam/grids"
grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0.hdf5"
grid = Grid(grid_name, grid_dir=grid_dir, read_lines=False)

stars = Stars(
    ages=10**np.array([6.80148435, 6.86897098, 7.19931891, 6.49461512, 5.6368416, 7.33821453, 7.86737916, 7.18474303, 7.38661522, 7.44604221, 7.48734393, 7.10764301]), 
    metallicities=np.array([1.72812573e-03, 1.33368056e-04, 1.07661654e-04, 1.03978673e-04, 7.00157660e-04, 2.78041092e-08, 3.39667400e-04, 1.48250647e-05, 1.40332331e-05, 2.15445732e-04, 1.76718662e-04, 1.04383791e-04]),
    initial_masses=np.array([1872523.71688373, 1850028.49320881, 1850280.24134226, 1869298.57591167, 1861203.49098928, 1850007.24741258, 1867570.09942085, 1850007.24741258, 1850007.24741258, 1850524.56799895, 1850648.69583584, 1850007.24741258]),
)
#ages = np.array([1e4]),#, 1e5, 1e5, 1e5]),
#metallicities = np.array([1e-6]),#, 1e-4, 1e-3, 1e-4]),


args = stars._prepare_sed_args(
    grid=grid,
    fesc=0.,
    spectra_type='incident',
    mask=np.array([True] * 12),
    grid_assignment_method='cic'
)

print(args)

from synthesizer.extensions.particle_spectra import compute_particle_seds
from synthesizer.extensions.integrated_spectra import compute_integrated_sed

# lnu_particle = compute_integrated_sed(*args)
lnu_particle = compute_particle_seds(*args)
