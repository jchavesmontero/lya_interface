#!/bin/bash
# Cobaya MCMC on Perlmutter CPU nodes, resubmitting itself until it converges.
#
# One chain per MPI rank, THREADS OpenMP threads per chain. Choose the split
# with scripts/thread_scaling.py on a compute node first. Perlmutter CPU nodes
# have 128 physical cores (256 hardware threads): --cpus-per-task counts
# hardware threads, so it is 2 x THREADS for one thread per core.
#
# Each job first queues its successor (starting after this one ends, whatever
# the reason, e.g. the time limit), then runs; Cobaya resumes from its
# checkpoint. When the run has converged the successor is cancelled. At most
# MAX_JOBS jobs are chained.
#
# Requirements on Perlmutter: an environment with cup1d, lya_interface,
# ForestFlow, LaCE and Cobaya, and mpi4py built against Cray MPICH:
#   MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
#
# Submit from anywhere:
#   sbatch scripts/nersc/mcmc_job.sh
#
#SBATCH --job-name=lya_mcmc
#SBATCH --account=EDIT_ACCOUNT
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --output=%x-%j.out

set -euo pipefail

# ---------------------------------------------------------------- edit these
THREADS=16                         # OpenMP threads per chain: cpus-per-task / 2
LYA_INTERFACE=$HOME/igmhub/lya_interface
RUN_DIR=$LYA_INTERFACE/output/hires_forestflow/cobaya_mcmc   # holds cobaya_mcmc.yaml
MAX_JOBS=10
activate_environment() {
    module load python
    conda activate lace            # environment with the IGMHub packages
}
# ---------------------------------------------------------------------------

activate_environment
JOB_INDEX=${JOB_INDEX:-1}
SCRIPT=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2}')
RUN="python $LYA_INTERFACE/scripts/run_mcmc.py $RUN_DIR/cobaya_mcmc.yaml"

if $RUN --status; then
    echo "already converged: nothing to do"
    exit 0
fi

NEXT=""
if [ "$JOB_INDEX" -lt "$MAX_JOBS" ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,JOB_INDEX=$((JOB_INDEX + 1)) "$SCRIPT")
    echo "job $JOB_INDEX of at most $MAX_JOBS; next job $NEXT queued"
else
    echo "job $JOB_INDEX is the last one (MAX_JOBS=$MAX_JOBS)"
fi

export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun --cpu-bind=cores $RUN || true   # a time-limit kill still leaves a checkpoint

if $RUN --status && [ -n "$NEXT" ]; then
    scancel "$NEXT"
    echo "converged: cancelled job $NEXT"
fi
