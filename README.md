# PACE (Planet population synthesis and their Architecture evolution in star Cluster Environments)

## 1. Code Description:
PACE is a planet population synthesis code designed to study the formation and long-term evolution of planetary systems in clustered stellar environments. It incorporates the standard pebble accretion paradigm and considers disk evolution by taking into account the external field radiative environment. Physical modules are bridged using [VENICE](https://arxiv.org/abs/2407.20332) and the code is implemented using the [AMUSE](https://github.com/amusecode/amuse) framework. A formal documentation of PACE with scientific results can be found [here](https://www.aanda.org/articles/aa/full_html/2024/09/aa51051-24/aa51051-24.html).


## 2. Directory Description
- `amuse_vader_src.zip`: Required version of Vader to run code.
- `shared_src/venice_src/`: Directory hosting VENICE. This is algorithm used to bridge the physics together.
- `shared_src/`: Directory hosting functions used to evolve the formation history of planets.
- `test_run/`: Directory hosting main script to run simulation and tests.

NOTE: To ensure PACE works, make sure to replace `amuse_vader` inside `amuse/src/` with the inflated version of `amuse_vader_src.zip`. The current version of AMUSE does not build the required Vader version to allow for migration.

## 3. Running Instructions

To run PACE, the following directories are required:
- `shared_src/cluster_data/`: This hosts your cluster particle files in hdf5 format.
- `test_run/planet_evo/`: This will host all files tracking the evolution of your planet embryos.

1. Go to `test_run/` directory: `cd test_run/`
2. Execute python script: `python3 run_pebb_OL18_map.py`

Upon executing python3 run_pebb_OL18_map.py, the script will store all information on the clusters radiative field and generate random disk parameters. The random disk parameters are then stored in `../random_IC_file.npz` for later reading.

Run python3 venice_pps_run.py to run the default pps model. Parameters are basically set in this file. We trace the planets and save them in file planet_data.npz, run venice_pps_plot.py to plot the data.
The files named module*.py are building different physical processes. For example, module_migration.py builds migration process. (is the name module good?)
The file extra_funcs.py defines some extra functions used in the files. parames.py defile some constants used in OL18.py.

However, venice.py, symmetric_matrix.py are src files of Venice (Wilhlem et al. in prep). OL18.py is the source file to calculate pebble accretion \citep{Ormel & Liu 2018}.

## 4. Some Notes
- In `test_run/run_pebb_OL18_map.py`:
    - Change line 40: This will point the script toward the cluster evolution files.
    - Change line 50 + 51: These are the initial and final snapshot of cluster evolution.
    - Change line 133: This will change the number of cores you will parallelise over.
    - Change line 174: This will flag whether you are running with a SLURM scheduler or not.
    - Change line 175: This will change the initial separation between embryos.
    - Change line 176: This will change depending if you are using a work station or SLURM scheduling.
    - Change line 200: This will set the number of prospective embryos per disk.

## 5. Test run
key file for vader: PACE_vader/PACE/amuse_vader_src/vader should be in the amuse vader community (amuse/src/amuse/community/vader/src/prob). The worker file should be included. Maybe the easiest (but most risky) way is to replace the vader file in the community.
compile: make vader.code
If succeed, run: python3 test_run.py. In principle, it should form planet. You can monitor the growth and migration of planet by compile read_planet_evo.py
Key function in the python script: run_single_pps (locates in venice_pps_setup_pebb_vader_OL18.py)

