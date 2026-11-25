# Sourced by the Slurm jobs in this folder (after LYA_INTERFACE and RUN_DIR are
# set): a header with what is run, and copies of the job log in RUN_DIR/logs.
#
# Slurm writes the log to logs/<job name>-<job id>.out in the submission folder
# (the lya_interface checkout); it is copied to RUN_DIR/logs every 10 minutes
# and when the job ends, so the run folder keeps the log of every job.

job_header() {  # $1: what the job does
    echo "================================================================"
    echo "job:        $SLURM_JOB_NAME $SLURM_JOB_ID ($1)"
    echo "date:       $(date -Is)"
    echo "nodes:      ${SLURM_JOB_NODELIST:-?}, tasks ${SLURM_NTASKS:-1}, qos ${SLURM_JOB_QOS:-?}, partition ${SLURM_JOB_PARTITION:-?}"
    echo "run folder: $RUN_DIR"
    echo "python:     $(command -v python)"
    local root repo dir dirty
    root=$(dirname "$LYA_INTERFACE")
    for repo in lya_interface cup1d ForestFlow LaCE; do
        dir=$root/$repo
        if [ -d "$dir/.git" ]; then
            dirty=""
            if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
                dirty=" (uncommitted changes)"
            fi
            echo "$(printf '%-12s' "$repo:") $(git -C "$dir" rev-parse --abbrev-ref HEAD) $(git -C "$dir" rev-parse --short HEAD)$dirty"
        else
            echo "$(printf '%-12s' "$repo:") not found at $dir"
        fi
    done
    echo "================================================================"
}

JOB_LOG=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ StdOut=/{print $2}')

copy_log() {
    mkdir -p "$RUN_DIR/logs"
    cp -f "$JOB_LOG" "$RUN_DIR/logs/" 2>/dev/null || true
}

# periodic copies, so a job killed at its time limit still leaves a recent one
( while sleep 600; do copy_log; done ) &
LOG_COPIER=$!
trap 'kill "$LOG_COPIER" 2>/dev/null || true; copy_log' EXIT
