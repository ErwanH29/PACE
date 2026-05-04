from amuse.lab import *
import natsort
import glob
import numpy as np

import matplotlib.pyplot as plt

all_files = natsort.natsorted(glob.glob("test_run/cluster_data/*"))
init = read_set_from_file(all_files[0])
fin  = read_set_from_file(all_files[-1])

dR_arr = [ ]
dR_rel_arr = [ ]
for i in init:
    target = fin[fin.key == i.key]
    dR = (target.Rdisk - i.Rdisk)
    if not target or i.Rdisk <= (0 | units.au):
        continue

    dR_arr.append(dR[0].value_in(units.au))
    dR_rel_arr.append(dR[0]/i.Rdisk)
    if dR == (0 | units.au):
        continue
    print(dR.in_(units.au), target.Rdisk/i.Rdisk)
    
for d in [dR_arr, dR_rel_arr]:
    sorted_d = np.sort(d)
    cdf = np.arange(1, len(sorted_d)+1) / (len(sorted_d))

    plt.plot(sorted_d, cdf)
    plt.show()