#!/bin/bash
# Cobaya MCMC on Perlmutter CPU nodes: production (resubmitting itself until
# it converges) or a short test of the whole path in the debug queue.
#
# One chain per MPI rank, THREADS OpenMP threads per chain. Perlmutter CPU
# nodes have 128 physical cores (256 hardware threads): --cpus-per-task counts
# hardware threads. thread_scaling.py found no gain from threads (0.134 s per
# call for 1-32), so chains use one thread; 64 chains per node leave each
# chain two cores (less memory-bandwidth contention than 128 chains).
#
# Production: each job first queues its successor (starting after this one
# ends, whatever the reason, e.g. the time limit), then runs; Cobaya resumes
# from its checkpoint. When the run has converged the successor is cancelled.
# At most MAX_JOBS jobs are chained.
#
#   sbatch scripts/nersc/mcmc_job.sh
#
# Submit from the lya_interface checkout: logs go to logs/lya_mcmc-<jobid>.out,
# with a copy in RUN_DIR/logs (every 10 min and at the end of each job).
#
# Test (MODE=test): debug queue, 64 chains of TEST_SAMPLES accepted points in
# RUN_DIR/test/ (overwritten), with convergence checks every 5d accepted points
# so the checkpoint and the learned covariance are written; no resubmission.
# Ends with a status report that reloads the chains.
#
#   sbatch --qos=debug --time=00:30:00 --export=ALL,MODE=test scripts/nersc/mcmc_job.sh
#
# Requirements on Perlmutter: an environment with cup1d, lya_interface,
# ForestFlow, LaCE and Cobaya, and mpi4py built against Cray MPICH:
#   MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
#
#SBATCH --job-name=lya_mcmc
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# ---------------------------------------------------------------- edit these
THREADS=1                          # OpenMP threads per chain
LYA_INTERFACE=/global/cfs/cdirs/desicollab/users/jjchaves/packages/lya_interface
# holds cobaya_mcmc.yaml, in the data folder
RUN_DIR=$CFS/data/DESIY1_QMLE3__Karacayli2022/global_opt/forest_mpg_gp/cobaya_mcmc
MAX_JOBS=10
TEST_SAMPLES=1000                  # accepted points per chain in MODE=test (~10 min)
CONDA_BIN=/global/homes/j/jjchaves/miniconda3/bin
CONDA_ENV=lace                     # environment with the IGMHub packages
activate_environment() {
    # activate script of the installation itself: no conda init or module needed;
    # conda's activation scripts may use unset variables, so relax set -u for it
    set +u
    source "$CONDA_BIN/activate" "$CONDA_ENV"
    set -u
}
# ---------------------------------------------------------------------------

MODE=${MODE:-production}
activate_environment
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export OMP_PLACES=threads
# Cobaya locks <output>.input.yaml.locked against two runs writing the same
# output. File locks are unreliable on CFS (DVS mounts): they fail spuriously
# or survive the process. Jobs here never overlap (the successor starts after
# this one ends; the test overwrites its own folder), so locking is not needed.
export COBAYA_USE_FILE_LOCKING=False
export OMP_PROC_BIND=spread
PYTHON=$CONDA_PREFIX/bin/python    # the environment's interpreter, by full path
RUN="$PYTHON $LYA_INTERFACE/scripts/run_mcmc.py $RUN_DIR/cobaya_mcmc.yaml"
source "$LYA_INTERFACE/scripts/nersc/job_log.sh"
job_header "MCMC $MODE, $SLURM_NTASKS chains x $THREADS threads, job ${JOB_INDEX:-1} of at most $MAX_JOBS"

if [ "$MODE" = "test" ]; then
    srun --cpu-bind=cores $RUN --test --test-samples "$TEST_SAMPLES" --learn-every 5d
    echo; echo "files in $RUN_DIR/test:"; ls -la "$RUN_DIR/test"
    echo; $RUN --test --status || true   # exit code 1 = not converged, expected for a test
    exit 0
fi

JOB_INDEX=${JOB_INDEX:-1}
SCRIPT=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2}')

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

srun --cpu-bind=cores $RUN || true   # a time-limit kill still leaves a checkpoint

if $RUN --status && [ -n "$NEXT" ]; then
    scancel "$NEXT"
    echo "converged: cancelled job $NEXT"
fi
