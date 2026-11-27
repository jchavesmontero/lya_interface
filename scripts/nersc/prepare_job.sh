#!/bin/bash
# Prepare the MCMC run folder on Perlmutter (scripts/prepare_mcmc.py): best fit,
# Gauss-Newton errors, proposal covariance, production YAML and best-fit figure.
# One process in the debug queue (minimization + errors take a few minutes).
#
# By default the fit is redone, starting from the saved best fit if there is
# one (the safe choice after any change of code or configuration). To keep a
# saved best fit and its errors and only rewrite the covariances and the YAML:
#
#   sbatch scripts/nersc/prepare_job.sh                     # redo the fit (default)
#   sbatch --export=ALL,REUSE=1 scripts/nersc/prepare_job.sh
#
# Submit from the lya_interface checkout: the log goes to
# logs/prepare_mcmc-<jobid>.out, and a copy to RUN_DIR/logs.
#
#SBATCH --job-name=prepare_mcmc
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# ---------------------------------------------------------------- edit these
THREADS=1                          # OpenMP threads: thread_scaling.py found no gain from more
LYA_INTERFACE=/global/cfs/cdirs/desicollab/users/jjchaves/packages/lya_interface
# CONFIG (required): the interface YAML, e.g.
#   sbatch --export=ALL,CONFIG=$PWD/examples/dr1_cm2026_base_evaluate.yaml scripts/nersc/prepare_job.sh
# RUN_DIR (optional): the run folder, the same as in mcmc_job.sh; by default
# prepare_mcmc.py derives it from the native out_folder of CONFIG.
CONFIG=${CONFIG:-}
RUN_DIR=${RUN_DIR:-}
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

if [ -z "$CONFIG" ]; then
    echo "set CONFIG: sbatch --export=ALL,CONFIG=<interface yaml>[,RUN_DIR=<folder>] scripts/nersc/prepare_job.sh"
    exit 1
fi
source "$LYA_INTERFACE/scripts/nersc/job_log.sh"
EXTRA="--force"
if [ "${REUSE:-0}" = "1" ]; then
    EXTRA=""
fi
job_header "prepare run folder; config $CONFIG; ${EXTRA:-reuse saved best fit}"
if [ -n "$RUN_DIR" ]; then
    EXTRA="$EXTRA --run-dir $RUN_DIR"
fi
"$CONDA_PREFIX/bin/python" "$LYA_INTERFACE/scripts/prepare_mcmc.py" "$CONFIG" $EXTRA
