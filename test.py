from halo import *
import random
import corner
import numpy as np
import matplotlib.pyplot as plt 

def process_result(sampler, ndim):
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    samples[:,0] = np.log10(((pc / solMass) ** 1.5) * (10 ** 18) * np.sqrt(3 / (4*np.pi*rho_c*delta)) * (samples[:,0]) ** 3 / (G ** (3/2))) # solMass
    result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    return result

def read_data(lines):
    data = []
    flag = 0
    for idx, line in enumerate(lines):
        line = line.split()
        if line[0] is '#' and flag == 0:
            continue
        elif line[0] is not '#':
            flag = 1
            data.append([float(i) for i in line])
        elif flag == 1:
            break
    return np.array(data), idx

name = 'DDO161'

out_file = open(name + "_result", "w+")

rc_file = name + ".txt"
with open(rc_file) as f:
    
    lines = f.readlines()
    observed, idx = read_data(lines)
    
    lines = lines[idx:]
    bulge, idx = read_data(lines)

    lines = lines[idx:]
    disk, idx = read_data(lines)

    lines = lines[idx:]
    gas, idx = read_data(lines)

    r, v_obs, v_obs_err = observed[:,0], observed[:,1], observed[:,2]
    v_stars = np.sqrt(bulge[:,1]**2 + disk[:,1]**2)    
    v_gas = gas[:,1]

M_stellar = 0.548 * (10 ** 9) # solMass
M_gas = 1.33 * 1.378 * (10 ** 9) # solMass

# for NFW

ndim, nwalkers = 3, 100

NFW_pos = []
for i in range(nwalkers):
    p = np.array([10 ** random.uniform(np.log10(10.0),np.log10(500.0)), \
                10 ** random.uniform(np.log10(1.0),np.log10(100.0)), random.uniform(0.1,1.0)])
    NFW_pos.append(p)

NFWHalo = NFW_Halo(r, v_obs, v_obs_err, v_gas, v_stars, M_gas, M_stellar)
NFWHalo.set_prior_range(rng=[(10., 500.), (1., 100.), (0.1, 1.0)])
sampler = NFWHalo.run_mcmc(NFW_pos, thread=40)
result = process_result(sampler, ndim)

V_vir_mcmc, c_vir_mcmc, mass_to_light_mcmc = result
V_vir_mcmc, V_vir_mcmc_err = V_vir_mcmc[0], V_vir_mcmc[1]
c_vir_mcmc = c_vir_mcmc[0] 
mass_to_light_mcmc = mass_to_light_mcmc[0]
print(V_vir_mcmc, V_vir_mcmc_err, c_vir_mcmc, mass_to_light_mcmc, file=out_file)