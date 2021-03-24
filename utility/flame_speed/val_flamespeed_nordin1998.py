# -*- coding: utf-8 -*-

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
p = ct.one_atm * 10  # pressure [Pa]
Tin = 500.0  # unburned gas temperature [K]

n_exp = 6
conds = np.zeros((n_exp, 10))
conds[:, 0] = np.linspace(0.7, 1.3, num=n_exp)

mech_master = "../../mechanism/mech_41s168r.yaml"
mech_sk = "../../mechanism/mech_34s121r.yaml"
mech_op = "../../mechanism/mech_34s121r_op.yaml"
# mech_sk26 = "../../mechanism/gri30_sk26.yaml"
# mech_sk27 = "../../mechanism/gri30_sk27.yaml"

width = 0.03  # m

def get_flame_speed(mech, n_exp, conds, col):
    for i in range(n_exp):
        phi = conds[i, 0]
        fuel2air = 11.0
        reactants = 'C7H16:{:.2f}, O2:1.0, N2:3.76'.format(phi/fuel2air)
        
        # Solution object used to compute mixture properties
        gas = ct.Solution(mech)
        gas.TPX = Tin, p, reactants
        
        # Flame object
        f = ct.FreeFlame(gas, width=width)
        f.set_refine_criteria(ratio=2, slope=0.05, curve=0.05, prune=0)
        
        # if i > 1:
        #     f.restore('master.xml', 'solution')
        #     f.solve(loglevel=1, auto=False)
        # else:
        f.solve(loglevel=0, auto=True)
        # f.save('master.xml', 'solution',
        #    'solution with multicomponent transport')
            
        print('\n {} {:d} phi {:.2f}, sl = {:4f} m/s\n'.format(mech, i, phi, f.velocity[0]))
        
        conds[i, col] = f.velocity[0] * 100.0
        
    return conds
    
get_flame_speed(mech_master, n_exp, conds, 1)
get_flame_speed(mech_sk, n_exp, conds, 2)
get_flame_speed(mech_op, n_exp, conds, 3)
# get_flame_speed(mech_sk26, n_exp, conds, 4)
# get_flame_speed(mech_sk27, n_exp, conds, 5)


plt.plot(conds[:, 0], conds[:, 1], '-', label='Nordin1998_41s')
plt.plot(conds[:, 0], conds[:, 2], 'o', label='sk34')
plt.plot(conds[:, 0], conds[:, 3], 's', label='sk34_op')
# plt.plot(conds[:, 0], conds[:, 4], 'o', label='sk26')
# plt.plot(conds[:, 0], conds[:, 5], 's', label='sk27')
plt.xlabel('Equivilence ratio')
plt.ylabel('Flame Speed [cm/s]')
plt.title('{:.1f} atm, Tu {:.0f} K C7H16/Air'.format(p/ct.one_atm, Tin))
plt.legend(loc='best')
plt.savefig('flamespeed.png')
plt.show()