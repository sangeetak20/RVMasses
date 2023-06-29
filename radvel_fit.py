import os
import pandas as pd
import numpy as np
import radvel
import argparse
from matplotlib import rcParams

rcParams.update({'font.size': 16})

hiresdata = pd.read_csv('T001246_4pl_data.csv')
harpsdata = pd.read_csv('../TOI 1246/HARPSN_2/TOI-1246-tng_harpn-0028-srv-rvs.dat',delimiter='     ',header=None)
harpsdata = harpsdata[[0,1,2]]
harpsdata.columns = ['time', 'mnvel', 'errvel']
harpsdata['tel'] = 'harps-n'

data = pd.concat([hiresdata, harpsdata])
t = np.array(data.time)
vel = np.array(data.mnvel)
errvel = np.array(data.errvel)
tel = np.array(data.tel)
telgrps = data.groupby('tel').groups
instnames = telgrps.keys()

starname = 'T001246'
nplanets = 5
fitting_basis = 'per tc secosw sesinw k'
bjd0 = 0.
planet_letters = {1: 'b', 2: 'c', 3: 'd', 4: 'e', 5:'f'}

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets, basis='per tc e w k', planet_letters=planet_letters) # initialize Parameters object
anybasis_params['per1'] = radvel.Parameter(value=4.307412)
anybasis_params['tc1'] = radvel.Parameter(value=2458686.564819)
anybasis_params['e1'] = radvel.Parameter(value=0, vary=True)
anybasis_params['w1'] = radvel.Parameter(value=0.000000)
anybasis_params['k1'] = radvel.Parameter(value=3.400000)

anybasis_params['per2'] = radvel.Parameter(value=5.903194)
anybasis_params['tc2'] = radvel.Parameter(value=2458683.474609)
anybasis_params['e2'] = radvel.Parameter(value=0, vary=True)
anybasis_params['w2'] = radvel.Parameter(value=0.000000)
anybasis_params['k2'] = radvel.Parameter(value=3.400000)

anybasis_params['per3'] = radvel.Parameter(value=18.652357)#18.65590
anybasis_params['tc3'] = radvel.Parameter(value=2458688.940308) #2458688.9653
anybasis_params['e3'] = radvel.Parameter(value=0, vary=True)
anybasis_params['w3'] = radvel.Parameter(value=0.000000)
anybasis_params['k3'] = radvel.Parameter(value=3.400000)

anybasis_params['per4'] = radvel.Parameter(value=37.9198) #37.9216
anybasis_params['tc4'] = radvel.Parameter(value=2458700.72042) #2458700.7134
anybasis_params['e4'] = radvel.Parameter(value=0)
anybasis_params['w4'] = radvel.Parameter(value=0.000000)
anybasis_params['k4'] = radvel.Parameter(value=3.400000)

anybasis_params['per5'] = radvel.Parameter(value=95)
anybasis_params['tc5'] = radvel.Parameter(value=2459460)
anybasis_params['e5'] = radvel.Parameter(value=0, vary=True)
anybasis_params['w5'] = radvel.Parameter(value=0.000000)
anybasis_params['k5'] = radvel.Parameter(value=3.400000)

# anybasis_params['per5'] = radvel.Parameter(value=75)
# anybasis_params['tc5'] = radvel.Parameter(value=2459405)
# anybasis_params['e5'] = radvel.Parameter(value=0)
# anybasis_params['w5'] = radvel.Parameter(value=0.000000)
# anybasis_params['k5'] = radvel.Parameter(value=3.400000)

time_base = 2458989.783463
anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
anybasis_params['curv'] = radvel.Parameter(value=0.0)

anybasis_params['gamma_j'] = radvel.Parameter(value=0.5, vary=True, linear=True)
anybasis_params['jit_j'] = radvel.Parameter(value=3.21, vary=True)
anybasis_params['gamma_harps-n'] = radvel.Parameter(value=1, vary=True, linear=True)
anybasis_params['jit_harps-n'] = radvel.Parameter(value=2.98, vary=True)


params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)
params['dvdt'].vary = True
params['curv'].vary = False

mod = radvel.RVModel(params, time_base=time_base)

mod.params['per1'].vary = False
mod.params['tc1'].vary = False
mod.params['secosw1'].vary = True
mod.params['sesinw1'].vary = True
mod.params['per2'].vary = False
mod.params['tc2'].vary = False
mod.params['secosw2'].vary = True
mod.params['sesinw2'].vary = True
mod.params['per3'].vary = False
mod.params['tc3'].vary = False
mod.params['secosw3'].vary = True
mod.params['sesinw3'].vary = True
mod.params['per4'].vary = False
mod.params['tc4'].vary = False
mod.params['secosw4'].vary = True
mod.params['sesinw4'].vary = True
mod.params['per5'].vary = True
mod.params['tc5'].vary = True
mod.params['secosw5'].vary = True
mod.params['sesinw5'].vary = True
# mod.params['per6'].vary = True
# mod.params['tc6'].vary = True
# mod.params['secosw6'].vary = False
# mod.params['sesinw6'].vary = False
mod.params['dvdt'].vary = True
mod.params['curv'].vary = True
mod.params['jit_j'].vary = True
mod.params['jit_harps-n'].vary = True


# priors = [radvel.prior.EccentricityPrior(nplanets,upperlims=[0.536,0.234,0.1897,0.377]), radvel.prior.PositiveKPrior(nplanets), radvel.prior.Gaussian('per1', 18.652357, 0.00665), radvel.prior.Gaussian('per2', 4.307412,  0.001355), radvel.prior.Gaussian('per3', 5.903194,  0.000474), radvel.prior.Gaussian('per4', 37.919841, 0.001333), radvel.prior.HardBounds('jit_j', 0.0, 20.0), radvel.prior.HardBounds('jit_harps', 0.0, 20.0)]radvel.prior.Gaussian('tc3', 2458688.9653, 0.0090),radvel.prior.Gaussian('tc4',2458700.7134, 0.0089),
priors = [radvel.prior.EccentricityPrior(nplanets), radvel.prior.PositiveKPrior(nplanets), radvel.prior.Gaussian('per3', 18.652357, 0.00665), radvel.prior.Gaussian('per1', 4.307412,  0.001355), radvel.prior.Gaussian('per2', 5.903194,  0.000474),  radvel.prior.Gaussian('per4',  37.919841, 0.001333),radvel.prior.HardBounds('jit_j', -20.0, 20.0), radvel.prior.HardBounds('jit_harps-n', -20.0, 20.0), radvel.prior.HardBounds('per5', 40,300)]
 #, radvel.prior.EccentricityPrior([4],upperlims=[0.377])] radvel.prior.Gaussian('per4', 37.919841, 0.001333), radvel.prior.HardBounds('per5', 40,300)
#radvel.prior.HardBounds('per5', 40,300),
stellar = dict(mstar=0.8683, mstar_err=0.05)
