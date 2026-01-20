"""
Script to read planet evolution data from generated 
.npz files and plot mass ratio vs period,
semimajor axis vs mass, and period vs mass.
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


datafile = 'planet_evo/'
N = 7000
fontsize = 15

a=[[ ] for _ in range(2)]
M=[[ ] for _ in range(2)]
P=[[ ] for _ in range(2)]
q=[[ ] for _ in range(2)]

for i in range(0,N):
    filename = datafile +'%06d_'%i+'planet.npz'
    if (os.path.exists(filename)):
        print(i)
        load_data_disk = np.load(filename, allow_pickle=True)
        load_data_planet = np.load(filename, allow_pickle=True)
        load_data_disk.keys()
        load_data_planet.keys()
        Mci = load_data_planet['Mc']
        Mei = load_data_planet['Me']
        ai  = load_data_planet['a']
        print(ai)
        star_massi = load_data_planet['star_mass']

        # Store initial data
        M[0].append(Mci[0]+Mei[0])
        a[0].append(ai[0])
        P[0].append(ai[0]**(3/2)*365.24*star_massi**-0.5)
        q[0].append((Mci[0]+Mei[0])/star_massi*3e-6)
        
        # Store final data
        M[1].append(Mci[-1]+Mei[-1])
        a[1].append(ai[-1])
        P[1].append(ai[-1]**(3/2)*365.24*star_massi**-0.5)
        q[1].append((Mci[-1]+Mei[-1])/star_massi*3e-6)


a_init=np.array(a[0])[0]
M_init=np.array(M[0])[0]
P_init=np.array(P[0])[0]
q_init=np.array(q[0])[0]

a_final=np.array(a[1])[0]
M_final=np.array(M[1])[0]
P_final=np.array(P[1])[0]
q_final=np.array(q[1])[0]

cmap = matplotlib.colormaps['cool']
cmap_colours = cmap(np.linspace(0.15, 1, len(a_init)))

fig = plt.figure(0,figsize=(8,8))
j = 0
plt.scatter(P_init, q_init, marker='o', c=cmap_colours, s=70)
plt.scatter(P_final, q_final, marker='X', c=cmap_colours, s=70)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-1,10**4)
plt.xlabel('period[day]', fontsize = fontsize)
plt.ylabel(r'Mass ratio', fontsize = fontsize)
plt.savefig('pps_P_q.png', dpi=500)
plt.close()

fig = plt.figure(1,figsize=(8,8))
plt.scatter(a_init, M_init, marker='o', c=cmap_colours, s=70)
plt.scatter(a_final, M_final, marker='X', c=cmap_colours, s=70)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-2.5,10**5)
plt.xlabel('sma[au]', fontsize = fontsize)
plt.ylabel(r'Mass[$M_\oplus$]', fontsize = fontsize)
plt.savefig('pps_a_M.png', dpi=500)
plt.close()

fig = plt.figure(2,figsize=(8,8))
plt.scatter(P_init, M_init, marker='o', c=cmap_colours, s=70)
plt.scatter(P_final, M_final, marker='X', c=cmap_colours, s=70)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-1,10**5)
plt.xlabel('Period[day]', fontsize = fontsize)
plt.ylabel(r'Mass[$M_\oplus$]', fontsize = fontsize)
plt.savefig('pps_P_M.png', dpi=500)
plt.close()