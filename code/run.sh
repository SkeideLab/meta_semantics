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
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=enge@cbs.mpg.de
#
# Wall clock limit:
#SBATCH --time=24:00:00

# Run the program:
module load anaconda
srun jupyter nbconvert --to=notebook --ExecutePreprocessor.kernel_name=python3 \
--ExecutePreprocessor.timeout=-1 --execute nb01_ale.ipynb "$SLURM_CPUS_PER_TASK"
