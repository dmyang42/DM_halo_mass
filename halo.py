# -*- coding: utf-8 -*-
                                                                                                                                                    
#                                              88      ,ad88PPP88ba,      ad88888ba          88 888888888888 88        88  
#     ,d                                       88     d8"  .ama.a "8a    d8"     "8b         88      88      88        88  
#     88                                       88    d8'  ,8P"88"  88    Y8,                 88      88      88        88  
#   MM88MMM ,adPPYba,  8b,dPPYba,   ,adPPYba,  88    88  .8P  8P   88    `Y8aaaaa,           88      88      88        88  
#     88   a8"     "8a 88P'    "8a a8"     "8a 88    88  88   8'   8P      `"""""8b,         88      88      88        88  
#     88   8b       d8 88       d8 8b       d8 88    88  8B ,d8 ,ad8'            `8b         88      88      88        88  
#     88,  "8a,   ,a8" 88b,   ,a8" "8a,   ,a8" 88    "8a "88P"888P"      Y8a     a8P 88,   ,d88      88      Y8a.    .a8P  
#     "Y888 `"YbbdP"'  88`YbbdP"'   `"YbbdP"'  88     `Y8aaaaaaaad8P      "Y88888P"   "Y8888P"       88       `"Y8888Y"'   
#                     88                                """""""""                                                         
#                     88                                                                                                  

import random
import emcee
import numpy as np
from scipy import integrate
from multiprocessing import Pool

G = 6.6738 * (10 ** (-11)) # m^3 kg^-1 s^-2
h = 0.7
rho_c = 2.77536627 * (10 ** (11)) * (h ** 2) # solMass Mpc^-3
solMass = 1.9885 * (10 ** 30) # kg
pc = 3.0856776 * (10 ** 16) # m
delta = 93.6 # factor

class Halo():
    
    def __init__(self, r, v_obs, v_obs_err, v_gas, v_stellar, M_gas, M_stellar):
        """
        r - 半径 (kpc)
        v_obs - 观测到的旋转速度 (km / s)
        v_obs_err - 旋转速度的误差 (km / s)
        v_gas - 观测到的气体component (km / s)
        v_stellar - 观测到的stellarcomponent (km / s)
        M_gas - 气体质量 (solMass)
        M_stellar - stellar质量 (solMass)
        """
        self.r = r
        self.v_obs = v_obs
        self.v_obs_err = v_obs_err
        self.v_gas = v_gas
        self.v_stellar = v_stellar
        self.M_stellar = M_stellar
        self.M_gas = M_gas

        self.ndim = 3

    def _mass_integration(self, ra, args):
        """
        used in circular velocity function
        args[0]预留给mass to light
        """
        return "sbpp"

    def _circular_velocity(self, radius, args):
        """
        通过速度Model计算某半径处的旋转速度
        ars = density profile中的参数
        """
        mass_to_light = args[0]

        v_c = []
        for ra in radius:
            idx = np.where(self.r == ra)[0][0]
            gas = self.v_gas[idx]
            stars = self.v_stellar[idx]
            # I should use astropy.unit
            # But now I dont want to modify anything

            halo = self._mass_integration(ra, args[1:])
            v_c.append(np.sqrt(gas**2 + mass_to_light * stars**2 + halo**2))
        return np.array(v_c)

    def _get_args(self, theta):
        """
        used in lnlike function
        """
        args = ("s", "b", "p", "p")
        return args

    def _lnlike(self, theta, x, y, yerr):
        """
        计算theta下model给出的速度, 然后计算似然值
        """
        args = self._get_args(theta)
        model = self._circular_velocity(x, args=args)
        for i in model:
            if np.isnan(i):
                return -np.inf

        sigma2 = yerr ** 2
        return -0.5 * np.sum((y-model)**2/sigma2)

    def set_prior_range(self, rng):
        self.theta_range = rng
        
    def _lnprior(self, theta, rng):
        """
        参数的先验分布
        """
        # fixed sequence
        V_vir, c_vir, mass_to_light = theta[0], theta[1], theta[2]
        M_vir = ((pc / solMass) ** 1.5) * (10 ** 18) * np.sqrt(3 / (4*np.pi*rho_c*delta)) * (V_vir) ** 3 / (G ** (3/2)) # solMass
        _M_stellar = mass_to_light * self.M_stellar # solMass
        _M_gas = self.M_gas # solMass
        
        if (_M_stellar + _M_gas) / M_vir > 0.2:
            return -np.inf

        for i in range(len(theta)):
            if rng[i][0] < theta[i] < rng[i][1]:
                continue
            else:
                return -np.inf
        return 0.0        
        
    def _lnprob(self, theta, x, y, yerr):
        """
        计算后验概率
        """
        lp = self._lnprior(theta, self.theta_range)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._lnlike(theta, x, y, yerr)

    def run_mcmc(self, pos, nwalkers=100, iteration=2000, thread=1):
        with Pool(thread) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob, args=(self.r, self.v_obs, self.v_obs_err), pool=pool)
            sampler.run_mcmc(pos0=pos, N=iteration)
        return sampler

"""
-    To construct a new density profile
-    New mass_integration function & get_args functions should be given

-    And nothing else

MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMM:  . . . ,M. .. . . .MMMMMMMMMM
MMMMMMM ................ . . . .MMMMMMMM
MMMMMM.. .MMMMM.........MMMM7. ..MMMMMMM
MMMMM  . .MMMMM.. ......MMMMM. . .MMMMMM
MMMMM....=MMMM...MMMMM..?MMMM?....MMMMMM
MMMMM . .NMMMM.  MMMMM ..MMMMD....MMMMMM
MMMMM....=MMMM ..MMMMM..?MMMM+....MMMMMM
MMMMM . ..MMMMMMMMMMMMMMMMMMM ....MMMMMM
MMMMMM. . .MMMMMMMM.MMMMMMMM.....MMMMMMM
MMMMMMM . . .+MMI    .OMM+.. .  MMMMMMMM
MMMMMMMMM,.      ..M . .......MMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
"""

class NFW_Halo(Halo):
    """
    NFW density profile
    """
    def _mass_integration(self, ra, args):
        """
        args = (r_s, rho_s)
        """
        r_s = args[0]
        rho_s = args[1]
        return 0.001 * np.sqrt((solMass / pc) * 0.001 * (np.log(1+ra/r_s)-(ra/r_s)/(1+ra/r_s)) * (4*np.pi*G*rho_s*(r_s**3)) / (ra))

    def _get_args(self, theta):
        """
        theta = ()
        """
        V_vir, c_vir, mass_to_light = theta
        r_s = ((pc / solMass) ** 0.5) * (10 ** 9) * V_vir / (c_vir * np.sqrt((4*np.pi*G*rho_c*delta)/3)) # kpc
        M_vir = ((pc / solMass) ** 1.5) * (10 ** 18) * np.sqrt(3 / (4*np.pi*rho_c*delta)) * (V_vir) ** 3 / (G ** (3/2)) # solMass 
        rho_s = M_vir / (4*np.pi*(r_s**3)*(np.log(1+c_vir)-c_vir/(1+c_vir))) # solMass kpc^-3
        
        args = (mass_to_light, r_s, rho_s)
        return args

class D14_Halo(Halo):
    """
    D14 density profile
    """
    def _d14_mass(self, x, alpha, beta, gamma, rs):
        return 4 * np.pi * (x ** 2) / ((x/rs)**gamma * (1+(x/rs)**alpha)**((beta-gamma)/alpha))
    
    def _mass_integration(self, ra, args):
        """
        args = (r_s, rho_s, alpha, beta, gamma)
        """
        r_s, rho_s, alpha, beta, gamma = args[0], args[1], args[2], args[3], args[4]
        return 0.001 * np.sqrt((solMass / pc) * 0.001 * G * rho_s * \
                integrate.quad(self._d14_mass, 0, ra, args=(alpha, beta, gamma, r_s))[0] / (ra))

    def _get_args(self, theta):
        V_vir, c_vir, mass_to_light = theta

        R_vir = ((pc / solMass) ** 0.5) * (10 ** 9) * V_vir / np.sqrt((4*np.pi*G*rho_c*delta)/3) # kpc
        r_m2 = ((pc / solMass) ** 0.5) * (10 ** 9) * V_vir / (c_vir * np.sqrt((4*np.pi*G*rho_c*delta)/3)) # kpc
        M_vir = ((pc / solMass) ** 1.5) * (10 ** 18) * np.sqrt(3 / (4*np.pi*rho_c*delta)) * (V_vir) ** 3 / (G ** (3/2)) # solMass 
        
        _M_stellar = mass_to_light * self.M_stellar # solMass
        _M_gas = self.M_gas # solMass
        M_halo = M_vir - _M_stellar - _M_gas # solMass
        X = np.log10(_M_stellar / M_halo)
        if X > -1.3:
            X = -1.3
        alpha = 2.94 - np.log10((10**(X+2.33))**(-1.08)+(10**(X+2.33))**(2.39))
        beta = 4.23 + 1.34 * X + 0.26 * (X ** 2)
        gamma = -0.06 - np.log10((10**(X+2.56))**(-0.68)+(10**(X+2.56)))

        r_s = r_m2 * ((beta -2)/(2-gamma))**(1/alpha)
        rho_s = M_vir / integrate.quad(self._d14_mass, 0, R_vir, args=(alpha, beta, gamma, r_s))[0] # solMass kpc^-3        
        
        args = (mass_to_light, r_s, rho_s, alpha, beta, gamma)
        return args

class cNFW_Halo(Halo):
    """
    cNFW density profile
    """
    def __init__(self, r, v_obs, v_obs_err, v_gas, v_stellar, M_gas, M_stellar):
        Halo.__init__(self, r, v_obs, v_obs_err, v_gas, v_stellar, M_gas, M_stellar)
        self.ndim = 4

    def _cored_NFW_mass(self, x, b, rs):
        return 4 * np.pi * (x ** 2) / ((1+b*(x/rs))*((1+(x/rs))**2))

    def _mass_integration(self, ra, args):
        """
        args = (r_s, rho_s, b)
        """
        r_s, rho_s, b = args[0], args[1], args[2]
        return 0.001 * np.sqrt((solMass / pc) * 0.001 * G * rho_s * \
                               integrate.quad(self._cored_NFW_mass, 0, ra, args=(b, r_s))[0] / (ra))
    
    def _get_args(self, theta):
        V_vir, c_vir, mass_to_light, b = theta
        R_vir = ((pc / solMass) ** 0.5) * (10 ** 9) * V_vir / np.sqrt((4*np.pi*G*rho_c*delta)/3) # kpc
        r_s = ((pc / solMass) ** 0.5) * (10 ** 9) * V_vir / (c_vir * np.sqrt((4*np.pi*G*rho_c*delta)/3)) # kpc
        M_vir = ((pc / solMass) ** 1.5) * (10 ** 18) * np.sqrt(3 / (4*np.pi*rho_c*delta)) * (V_vir) ** 3 / (G ** (3/2)) # solMass 
        rho_s = M_vir / integrate.quad(self._cored_NFW_mass, 0, R_vir, args=(b, r_s))[0] # solMass kpc^-3        
        
        args = (mass_to_light, r_s, rho_s, b)
        return args

class Einasto_Halo(Halo):
    """
    Einasto density profile
    """
    def __init__(self, r, v_obs, v_obs_err, v_gas, v_stellar, M_gas, M_stellar):
        Halo.__init__(self, r, v_obs, v_obs_err, v_gas, v_stellar, M_gas, M_stellar)
        self.ndim = 4

    def _einasto_mass(self, x, alpha, rs):
        return 4 * np.pi * (x ** 2) * np.exp(-2/alpha*((x/rs)**alpha)-1)

    def _mass_integration(self, ra, args):
        """
        args = (r_s, rho_s, alpha)
        """
        r_s, rho_s, alpha = args[0], args[1], args[2]
        return 0.001 * np.sqrt((solMass / pc) * 0.001 * G * rho_s * \
                               integrate.quad(self._einasto_mass, 0, ra, args=(alpha, r_s))[0] / (ra))

    def _get_args(self, theta):
        V_vir, c_vir, mass_to_light, alpha = theta
        R_vir = ((pc / solMass) ** 0.5) * (10 ** 9) * V_vir / np.sqrt((4*np.pi*G*rho_c*delta)/3) # kpc
        r_s = ((pc / solMass) ** 0.5) * (10 ** 9) * V_vir / (c_vir * np.sqrt((4*np.pi*G*rho_c*delta)/3)) # kpc
        M_vir = ((pc / solMass) ** 1.5) * (10 ** 18) * np.sqrt(3 / (4*np.pi*rho_c*delta)) * (V_vir) ** 3 / (G ** (3/2)) # solMass 

        rho_s = M_vir / integrate.quad(self._einasto_mass, 0, R_vir, args=(alpha, r_s))[0] # solMass kpc^-3
         
        args = (mass_to_light, r_s, rho_s, alpha)
        return args