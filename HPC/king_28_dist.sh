#!/bin/bash
#SBATCH --account=owner-guest
#SBATCH --partition=kingspeak-guest
#SBATCH --job-name=lambda_limit
#SBATCH --time=10:00:00
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=32G
#SBATCH --output=lambda_limit-%j.out
#SBATCH --error=lambda_limit-%j.err
##SBATCH --mail-type=FAIL,END
##SBATCH --mail-user=jplundquist@gmail.com
#SBATCH --exclude=kp292

set -euo pipefail

module purge
module load deeplearning/2025.4
module load openmpi

m=$(( SLURM_CPUS_PER_TASK*2))
export NUMBA_NUM_THREADS="$m"              # or 28
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export OMP_PROC_BIND=false
export OMP_PLACES=threads

# Apptainer cache can be shared
export APPTAINER_CACHEDIR=/scratch/general/vast/$USER/apptainer_cache
mkdir -p "$APPTAINER_CACHEDIR"

# PRTE/OpenMPI temp/session should be node-local
unset TMPDIR
export PRTE_MCA_prte_tmpdir_base=/tmp
export PRTE_MCA_prte_silence_shared_fs=1

cd /uufs/chpc.utah.edu/common/home/u0446071/lambda_corr-proj/HPC

export L0_N=${L0_N:-325}

echo "JOBID=$SLURM_JOB_ID"
echo "NodeList=$SLURM_JOB_NODELIST"
srun -n $SLURM_NTASKS hostname

# minimal container sanity
singularity exec /uufs/chpc.utah.edu/sys/installdir/r8/deeplearning/2025.4/deeplearning_2025.4.sif python -V

# minimal MPI + container sanity (same launcher as your real job)
mpirun -np $SLURM_NTASKS hostname
mpirun -np 2 singularity exec /uufs/chpc.utah.edu/sys/installdir/r8/deeplearning/2025.4/deeplearning_2025.4.sif python -V

mpirun -np $SLURM_NTASKS --bind-to none \
  singularity exec --nv /uufs/chpc.utah.edu/sys/installdir/r8/deeplearning/2025.4/deeplearning_2025.4.sif \
  python -u fit_beta_distr.py
