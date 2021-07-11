#!/bin/bash -l

# Standard output and error
#SBATCH -o /ptmp/aenge/meta_semantics/code/slurm/tjob.out.%j
#SBATCH -e /ptmp/aenge/meta_semantics/code/slurm/tjob.err.%j

# Initial working directory
#SBATCH -D /ptmp/aenge/meta_semantics/

# Job Name
#SBATCH -J meta_semantics

# Queue (Partition)
#SBATCH --partition=general

# Number of nodes and MPI tasks per node
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40

# Request main memory per node in units of MB
#SBATCH --mem=185000

# Wall clock limit
#SBATCH --time=24:00:00

# E-mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=enge@cbs.mpg.de

# Load singularity
module load singularity

# Create shorthand command for executing scripts inside the container
EXEC=(
    srun -N1 -n1 singularity exec
        --bind /ptmp/aenge/meta_semantics:/home/jovyan/meta_semantics
        --home /ptmp/aenge/home
        --pwd /home/jovyan/meta_semantics/code
        meta_semantics_latest.sif
)

# First, perform only the actual ALE analyses
"${EXEC[@]}" python3 nb01_ale.py

# Then, perform all of the other analyses in parallel
"${EXEC[@]}" python3 nb02_subtraction.py &
    "${EXEC[@]}" python3 nb03_adults.py &
    "${EXEC[@]}" python3 nb04_sdm.py &
    "${EXEC[@]}" python3 nb05_jackknife.py &
    "${EXEC[@]}" python3 nb06_fsn.py all 4 &
    "${EXEC[@]}" python3 nb06_fsn.py all 3 &
    "${EXEC[@]}" python3 nb06_fsn.py all 3 &
    "${EXEC[@]}" python3 nb06_fsn.py knowledge,relatedness,objects 10 &
    "${EXEC[@]}" python3 nb09_supplement.py
wait

# Create output tables and figures
"${EXEC[@]}" python3 nb07_tables.py &
    "${EXEC[@]}" python3 nb08_figures.py
