aout = [0.2857, 0.3076, 0.3333, 0.3636, 0.4000]
stages = autostages(aout, astart=0.05, N=40)
unitary = True

seed = 121

nc = 2 * 1500
pm_nc_factor = 2

boxsize = 2 * 1380

use_non_gaussianity = False
fnl = 0

power_kedges = arange(1e-3, 3e0, 1e-3)
