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

## 3. Production (resubmits itself until converged)

```bash
sbatch scripts/nersc/mcmc_job.sh
```

Follow it:

```bash
squeue --me                                              # running and queued jobs
tail -f logs/lya_mcmc-<jobid>.out                        # current job
python scripts/run_mcmc.py $RUN_DIR/cobaya_mcmc.yaml --status   # R-1 so far
```

Chains: `$RUN_DIR/chains/`. Copies of all logs: `$RUN_DIR/logs/`.

## If something changed

- Code, priors or configuration changed: start again from step 1, which redoes
  the fit. Move the old `$RUN_DIR/chains/` aside before step 3, otherwise the
  production run resumes it.
- Only want new covariances or YAML from the saved best fit:
  `sbatch --export=ALL,REUSE=1 scripts/nersc/prepare_job.sh`
- A production job stopped (time limit, node failure): `sbatch scripts/nersc/mcmc_job.sh`
  again; it resumes from the checkpoint.
- Restart a running production with new code or sampler settings: cancel the
  running job **and** its queued successor, then submit again (never two job
  chains at once: they would write the same chains). It resumes from the
  checkpoint:
  ```bash
  scancel --me --name=lya_mcmc
  sbatch scripts/nersc/mcmc_job.sh
  ```
