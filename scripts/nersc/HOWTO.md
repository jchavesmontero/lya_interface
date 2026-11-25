# Hires MCMC on NERSC: what to run, in order

Everything from the lya_interface checkout on Perlmutter:

```bash
cd /global/cfs/cdirs/desicollab/users/jjchaves/packages/lya_interface
source /global/homes/j/jjchaves/miniconda3/bin/activate lace
RUN_DIR=$CFS/data/DESIY1_QMLE3__Karacayli2022/global_opt/forest_mpg_gp/cobaya_mcmc
```

## 0. Update the code

```bash
for repo in lya_interface cup1d ForestFlow LaCE; do git -C ../$repo pull; done
```

## 1. Best fit and covariance (debug queue, a few minutes)

```bash
sbatch scripts/nersc/prepare_job.sh
```

Check when it ends:

- `logs/prepare_mcmc-<jobid>.out`: chi2 and PTE, parameters near prior bounds,
  the last line says `next: test the sampler ...`
- `$RUN_DIR/best_fit.png`: model vs data and residuals

## 2. Test of the sampler (debug queue, ~15 min)

```bash
sbatch --qos=debug --time=00:30:00 --export=ALL,MODE=test scripts/nersc/mcmc_job.sh
```

Check `logs/lya_mcmc-<jobid>.out`: acceptance rate (roughly 0.1-0.4), time per
step, and the final status report (chains reloaded, `covmat` written).

## 3. Production

Every production run lives in its own numbered folder `$RUN_DIR/chain_<n>/`,
with frozen copies of the YAML and proposal covariance it started from (later
prepare jobs never change it). Jobs never write outside their chain folder.

Start a new chain with 30 min in the debug queue (no resubmission), to check
that everything works:

```bash
sbatch --export=ALL,NEW_CHAIN=1 scripts/nersc/mcmc_job_test.sh
```

Check `logs/lya_mcmc_test-<jobid>.out`: the chain folder in the header
(e.g. `chain_1`), no errors, chains written (`ls -lt $RUN_DIR/chain_1 | head`),
and after ~5 min a convergence check with `Updated covariance matrix of
proposal pdf`. Then the long run, which resumes the latest chain and
resubmits itself until converged:

```bash
sbatch scripts/nersc/mcmc_job.sh
```

Follow it (`<n>`: the chain number):

```bash
squeue --me                                                     # running and queued jobs
tail -f logs/lya_mcmc-<jobid>.out                               # current job
python scripts/run_mcmc.py $RUN_DIR/chain_<n>/cobaya_mcmc.yaml --status   # R-1 so far (seconds)
# add --detailed to also count every chain and reload them (minutes for long chains)
```

Copies of all logs: `$RUN_DIR/logs/`.

Diagnostics of the latest chain so far (login node, about a minute; stuck
chains, R-1 of each parameter, mass at prior edges, learned proposal):

```bash
python scripts/chain_diagnostics.py $RUN_DIR        # another chain: --prefix chain_<n>/mcmc
cat $RUN_DIR/diagnostics/chain_<n>/summary.txt
```

To look at the figures on the laptop, copy only that folder (small):

```bash
rsync -av jjchaves@perlmutter.nersc.gov:<RUN_DIR>/diagnostics <local RUN_DIR>/
```

## If something changed

- Code, priors or configuration changed: start again from step 1 (redoes the
  fit), then step 3 with `NEW_CHAIN=1`: a new `chain_<n+1>`; old chains are
  left untouched.
- Only want new covariances or YAML from the saved best fit:
  `sbatch --export=ALL,REUSE=1 scripts/nersc/prepare_job.sh` (affects new
  chains only).
- A production job stopped (time limit, node failure): `sbatch scripts/nersc/mcmc_job.sh`
  again; it resumes the latest chain from its checkpoint (`--export=ALL,CHAIN=<n>`
  for another one).
- Restart a running production with new code or sampler settings: cancel the
  running job **and** its queued successor, then submit again (never two job
  chains on the same chain folder: they would write the same files):
  ```bash
  scancel --me --name=lya_mcmc
  sbatch scripts/nersc/mcmc_job.sh
  ```
