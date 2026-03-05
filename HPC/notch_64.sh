#!/bin/bash
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --job-name=lambda_limit
#SBATCH --time=5:00:00
#SBATCH --nodes=120
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=lambda_limit-%j.out
#SBATCH --error=lambda_limit-%j.err
##SBATCH -C rom
##SBATCH -C "skl|csl|icl|srp"
##SBATCH --exclude=notch108,notch109,notch110,notch373,notch374,notch296,notch314,notch[111-128]

set -euo pipefail

export PYTHONNOUSERSITE=1
unset PYTHONPATH
module purge
module load mpich/4.2.1
#module load openmpi
#module load deeplearning/2025.4   # only if you still need it for other things

#module purge
#module load openmpi
module load miniconda3   # replace with the actual module name you see
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate l0_mpich

#export OMPI_MCA_pml=ucx
#export OMPI_MCA_osc=ucx
## Optional: keep TCP out of the picture
#export OMPI_MCA_btl="^tcp"

## avoid link-local TCP disasters
#export OMPI_MCA_oob_tcp_if_exclude=lo,169.254.0.0/16
#export OMPI_MCA_btl_tcp_if_exclude=lo,169.254.0.0/16

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

echo "SLURM_NTASKS=$SLURM_NTASKS SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
srun --mpi=list

srun --mpi=pmi2 -n $SLURM_NTASKS python -c "from mpi4py import MPI; import socket; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size(), socket.gethostname())"
srun --mpi=pmi2 -n 1 python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
echo "PMI env check (expect ranks 0..):"
srun --mpi=pmi2 -n $SLURM_NTASKS bash -lc 'echo host=$(hostname) proc=$SLURM_PROCID PMI_RANK=${PMI_RANK:-NA} PMI_SIZE=${PMI_SIZE:-NA} PMIX_RANK=${PMIX_RANK:-NA}'

# minimal container sanity
#singularity exec /uufs/chpc.utah.edu/sys/installdir/r8/deeplearning/2025.4/deeplearning_2025.4.sif python -V

# minimal MPI + container sanity (same launcher as your real job)
#mpirun -np $SLURM_NTASKS hostname
#mpirun -np 2 singularity exec /uufs/chpc.utah.edu/sys/installdir/r8/deeplearning/2025.4/deeplearning_2025.4.sif python -V

#mpirun -np $SLURM_NTASKS --bind-to none \
#  singularity exec --nv /uufs/chpc.utah.edu/sys/installdir/r8/deeplearning/2025.4/deeplearning_2025.4.sif \
#  python -u fit_beta_distr.py

#mpirun -np $SLURM_NTASKS python -u fit_beta_distr.py
srun --mpi=pmi2 -n $SLURM_NTASKS python -u fit_beta_distr.py
