#!/bin/bash -l

# Standard output and error:
#SBATCH -o /ptmp/aenge/mask_children/code/slurm/tjob.out.%j
#SBATCH -e /ptmp/aenge/mask_children/code/slurm/tjob.err.%j
# Initial working directory:
#SBATCH -D /ptmp/aenge/mask_children/code/
# Job Name:
#SBATCH -J mask_children
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
# Wall clock limit:
#SBATCH --time=24:00:00
# E-mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=enge@cbs.mpg.de

# Run the program:
module load anaconda
JUPYTER_PARAMS=(--to=notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1)
srun jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb01_ale.ipynb
srun jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb02_subtraction.ipynb
