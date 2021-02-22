#!/bin/bash -l

# Standard output and error:
#SBATCH -o /ptmp/aenge/mask_children/code/slurm/tjob.out.%j
#SBATCH -e /ptmp/aenge/mask_children/code/slurm/tjob.err.%j
# Initial working directory:
#SBATCH -D /ptmp/aenge/mask_children/code/
# Job Name:
#SBATCH -J mask_children_sing
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

# Request some main memory per node in units of MB:
#SBATCH --mem=32000

# Wall clock limit:
#SBATCH --time=24:00:00
# E-mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=enge@cbs.mpg.de

# Run the program:
module load singularity
singularity exec --no-home --bind /ptmp/aenge/mask_children:/home/mask_children \
    ../mask_children_latest.sif /home/mask_children/code/sing_inside.sh
#cd /home/mask_children/code
#. /opt/miniconda-latest/etc/profile.d/conda.sh
#conda activate nimare
##export NUMEXPR_MAX_THREADS=64
#JUPYTER_PARAMS=(--to=notebook --execute --inplace \
#    --ExecutePreprocessor.kernel_name=python3 \
#    --ExecutePreprocessor.timeout=-1)
#jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb01_ale.ipynb
#srun jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb02_subtraction.ipynb
#srun jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb03_adults.ipynb
#srun jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb04_sdm.ipynb

#srun python3 nb04_sdm.py "$SLURM_CPUS_PER_TASK"

#singularity shell --no-home --bind /ptmp/aenge/mask_children:/home/mask_children mask_children_latest.sif
