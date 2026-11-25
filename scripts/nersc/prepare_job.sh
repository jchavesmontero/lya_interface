#!/bin/bash
# Prepare the MCMC run folder on Perlmutter: best fit (minimized only if
# missing), Gauss-Newton errors, proposal covariance, production YAML and
# best-fit figure (scripts/prepare_mcmc.py). One process on a shared CPU node.
#
#   sbatch scripts/nersc/prepare_job.sh                              # reuse a saved best fit
#   sbatch --export=ALL,FORCE=1 scripts/nersc/prepare_job.sh         # redo everything
#
# The run folder is <$CFS/data>/<out_folder of the native cup1d YAML>/cobaya_mcmc,
# the RUN_DIR of mcmc_job.sh. Log: prepare_mcmc-<jobid>.out in the submission folder.
#
#SBATCH --job-name=prepare_mcmc
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# ---------------------------------------------------------------- edit these
THREADS=1                          # OpenMP threads: thread_scaling.py found no gain from more
LYA_INTERFACE=/global/cfs/cdirs/desicollab/users/jjchaves/packages/lya_interface
CONFIG=$LYA_INTERFACE/examples/hires_forestflow_evaluate.yaml
CONDA_BIN=/global/homes/j/jjchaves/miniconda3/bin
CONDA_ENV=lace                     # environment with the IGMHub packages
# ---------------------------------------------------------------------------

set +u  # conda's activation scripts may use unset variables
source "$CONDA_BIN/activate" "$CONDA_ENV"
set -u
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

EXTRA=""
if [ "${FORCE:-0}" = "1" ]; then
    EXTRA="--force"
fi
echo "config $CONFIG, threads $THREADS, force=${FORCE:-0}"
"$CONDA_PREFIX/bin/python" "$LYA_INTERFACE/scripts/prepare_mcmc.py" "$CONFIG" $EXTRA
