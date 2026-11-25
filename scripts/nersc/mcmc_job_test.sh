#!/bin/bash
# The production MCMC (mcmc_job.sh) for 30 minutes in the debug queue, to check
# that everything works before launching the long self-resubmitting jobs:
#
#   sbatch scripts/nersc/mcmc_job_test.sh     # check logs/lya_mcmc_test-<jobid>.out
#   sbatch scripts/nersc/mcmc_job.sh          # then the real thing
#
# Same chains, settings and code path as mcmc_job.sh (it runs that script):
# it starts or resumes chains/mcmc, so its 30 minutes count for the run. The
# only differences are the queue, the time limit and that it queues no
# successor (MAX_JOBS=1). Not to be confused with MODE=test of mcmc_job.sh,
# a short separate chain in test/.
#
# Submit from the lya_interface checkout; keep the resources below equal to
# those of mcmc_job.sh.
#
#SBATCH --job-name=lya_mcmc_test
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MAX_JOBS=1 bash "$SLURM_SUBMIT_DIR/scripts/nersc/mcmc_job.sh"
