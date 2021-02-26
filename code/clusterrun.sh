#!/bin/bash -l

# Standard output and error:
#SBATCH -o /ptmp/aenge/mask_children/code/slurm/tjob.out.%j
#SBATCH -e /ptmp/aenge/mask_children/code/slurm/tjob.err.%j
# Initial working directory:
#SBATCH -D /ptmp/aenge/mask_children/
# Job Name:
#SBATCH -J mask_children_sing
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
# Request main memory per node in units of MB:
#SBATCH --mem=185000
# Wall clock limit:
#SBATCH --time=24:00:00
# E-mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=enge@cbs.mpg.de

# Run the program:

# Pull the docker image, if necessary
module load singularity

# Run the shell script (which executes the notebooks) in the container
singularity exec --no-home --bind /ptmp/aenge/mask_children:/home/mask_children \
    mask_children_latest.sif /home/mask_children/code/runall.sh

# For debugging in the terminal
#singularity shell --no-home --bind /ptmp/aenge/mask_children:/home/mask_children mask_children_latest.sif