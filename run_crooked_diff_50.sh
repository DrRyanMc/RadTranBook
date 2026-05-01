#!/bin/bash
#$ -M rmcclarr@nd.edu
#$ -m abe
#$ -q long@@mcclarren
#$ -pe smp 50
#$ -N crooked_diff_50

CHECKPOINT="crooked_pipe_checkpoint_50g_refined_168x354.npz"
DONE_FLAG="${CHECKPOINT%.npz}.done"
SCRIPT="$( cd "$( dirname "$0" )" && pwd )/$( basename "$0" )"

# Don't re-run if a prior execution already finished cleanly.
if [ -f "$DONE_FLAG" ]; then
    echo "Run already completed ($DONE_FLAG exists). Exiting."
    exit 0
fi

# SGE sends SIGTERM before the hard kill (walltime / admin qdel).
# Resubmit while a checkpoint is available so work isn't lost.
resubmit() {
    echo "SIGTERM received – resubmitting job from checkpoint."
    if [ -f "$CHECKPOINT" ]; then
        qsub "$SCRIPT"
    else
        echo "No checkpoint found; cannot resubmit."
    fi
    exit 1
}
trap resubmit SIGTERM

module load python
cd ~/RadTranBook

BASE_ARGS="--use-refined-mesh --n-groups 50 --n-threads 50 --checkpoint-every 5"

if [ -f "$CHECKPOINT" ]; then
    echo "Restarting from checkpoint: $CHECKPOINT"
    python3 nonEquilibriumDiffusion/problems/crooked_pipe_multigroup_noneq.py \
        $BASE_ARGS \
        --restart-file "$CHECKPOINT"
else
    python3 nonEquilibriumDiffusion/problems/crooked_pipe_multigroup_noneq.py \
        $BASE_ARGS
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    touch "$DONE_FLAG"
    echo "Completed successfully."
elif [ -f "$CHECKPOINT" ]; then
    echo "Exited with code $EXIT_CODE – resubmitting from checkpoint."
    qsub "$SCRIPT"
else
    echo "Exited with code $EXIT_CODE and no checkpoint; not resubmitting."
fi
