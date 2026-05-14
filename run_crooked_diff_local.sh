#!/bin/bash
# run_crooked_diff_local.sh
#
# Runs the multigroup non-equilibrium diffusion crooked-pipe solver on the
# local machine, restarting automatically from checkpoint whenever the
# process exits non-zero (crash, Ctrl-C, etc.).
#
# Usage:
#   bash run_crooked_diff_local.sh          # fresh or auto-restart
#   bash run_crooked_diff_local.sh --clean  # delete checkpoint/done flag and start fresh

# ─── Configuration ─────────────────────────────────────────────────────────
N_GROUPS=2
N_THREADS=2        # set to the number of physical cores you want to use

# Checkpoint filename matches what the Python script auto-generates:
#   crooked_pipe_checkpoint_<N_GROUPS>g_refined_<nx>x<ny>.npz
# For a refined mesh built from 60 coarse r-cells × 210 coarse z-cells the
# solver produces a 168×354 fine mesh, so:
CHECKPOINT="crooked_pipe_checkpoint_${N_GROUPS}g_refined_168x354.npz"
DONE_FLAG="${CHECKPOINT%.npz}.done"

BASE_ARGS="--use-refined-mesh \
           --n-groups ${N_GROUPS} \
           --n-threads ${N_THREADS} \
           --dt-initial 1e-4 \
           --dt-max 0.005 \
           --dt-growth 1.1 \
           --bc-t-start 0.5 \
           --bc-t-end 0.5 \
           --bc-ramp-time 1.0 \
           --output-times 0.001,0.01,0.1,1,5,10,20,50,100,200 \
           --checkpoint-every 10 \
           --max-steps 10"

# ─── Optional --clean flag ──────────────────────────────────────────────────
if [ "${1}" = "--clean" ]; then
    echo "Removing checkpoint and done flag for a clean restart."
    rm -f "$CHECKPOINT" "$DONE_FLAG"
fi

# ─── Skip if already finished cleanly ──────────────────────────────────────
if [ -f "$DONE_FLAG" ]; then
    echo "Run already completed ($DONE_FLAG exists)."
    echo "Pass --clean to start over."
    exit 0
fi

cd "$(dirname "$0")"

MAX_RESTARTS=10000   # safety valve: stop after this many consecutive crashes

for attempt in $(seq 1 $MAX_RESTARTS); do
    if [ -f "$CHECKPOINT" ]; then
        echo ""
        echo "=== Attempt $attempt: restarting from $CHECKPOINT ==="
        python3 nonEquilibriumDiffusion/problems/crooked_pipe_multigroup_noneq.py \
            $BASE_ARGS \
            --restart-file "$CHECKPOINT" \
            --checkpoint-file "$CHECKPOINT"
    else
        echo ""
        echo "=== Attempt $attempt: starting fresh run ==="
        python3 nonEquilibriumDiffusion/problems/crooked_pipe_multigroup_noneq.py \
            $BASE_ARGS \
            --checkpoint-file "$CHECKPOINT"
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        touch "$DONE_FLAG"
        echo ""
        echo "Run completed successfully."
        exit 0
    fi

    # Exit code 75 = max_steps reached, checkpoint saved, restart needed.
    if [ $EXIT_CODE -eq 75 ]; then
        echo ""
        echo "Step limit reached (exit 75) – restarting from checkpoint (attempt $((attempt + 1)))…"
        continue
    fi

    echo ""
    echo "Process exited with code $EXIT_CODE."

    if [ ! -f "$CHECKPOINT" ]; then
        echo "No checkpoint found after failure; cannot restart. Exiting."
        exit $EXIT_CODE
    fi

    echo "Checkpoint found – restarting (attempt $((attempt + 1)))…"
done

echo "Reached maximum restart limit ($MAX_RESTARTS). Exiting."
exit 1
