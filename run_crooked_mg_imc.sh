#!/bin/bash
#$ -M rmcclarr@nd.edu
#$ -m abe
#$ -q long@@mcclarren
#$ -pe smp 10
#$ -N crooked_mg_imc

# ─── Configuration ─────────────────────────────────────────────────────────
N_GROUPS=10
NR=60
NZ=210
NMax=1000000*$N_GROUPS  # keep number of particles per group constant as N_GROUPS changes
#make Ntotal half NMax and make sure it is an integer.
Ntotal=$(( (NMax) / 2 ))

# Checkpoint filename must match what the Python script auto-generates:
#   crooked_pipe_mg_imc_checkpoint_<N_GROUPS>g_<mesh_tag>_<nr_actual>x<nz_actual>.pkl
# For a uniform 60x210 mesh with N_GROUPS groups that is:
CHECKPOINT="crooked_pipe_mg_imc_checkpoint_${N_GROUPS}g_refined_${NR}x${NZ}.pkl"
DONE_FLAG="${CHECKPOINT%.pkl}.done"

SCRIPT="$( cd "$( dirname "$0" )" && pwd )/$( basename "$0" )"

# ─── Skip if already finished ───────────────────────────────────────────────
if [ -f "$DONE_FLAG" ]; then
    echo "Run already completed ($DONE_FLAG exists). Exiting."
    exit 0
fi

# ─── Resubmit on SIGTERM (SGE walltime / qdel) ─────────────────────────────
resubmit() {
    echo "SIGTERM received – attempting resubmission."
    if [ -f "$CHECKPOINT" ]; then
        echo "  Checkpoint found: $CHECKPOINT"
        qsub "$SCRIPT"
    else
        echo "  No checkpoint found; cannot resubmit."
    fi
    exit 1
}
trap resubmit SIGTERM

# ─── Environment ────────────────────────────────────────────────────────────
module load python
cd ~/RadTranBook

BASE_ARGS="--n-groups ${N_GROUPS} \
           --use-refined-mesh \
           --Ntotal-T-floor .015 \
           --Ntotal ${Ntotal} \
           --Nmax ${NMax} \
           --dt-initial 1e-4 \
           --dt-max 0.01 \
           --dt-growth 1.1 \
           --bc-t-start 0.05 \
           --bc-t-end 0.5 \
           --bc-ramp-time 20.0 \
           --output-times 0.001,0.01,0.1,1,5,10,20,50,100,200,500,1000 \
           --checkpoint-every 10 \
           --max-events 10000000"

# ─── Run (restart if checkpoint exists) ────────────────────────────────────
if [ -f "$CHECKPOINT" ]; then
    echo "Restarting from checkpoint: $CHECKPOINT"
    python3 MG_IMC/crooked_pipe_multigroup_imc.py \
        $BASE_ARGS \
        --restart-file "$CHECKPOINT" \
        --checkpoint-file "$CHECKPOINT"
else
    echo "Starting fresh run."
    python3 MG_IMC/crooked_pipe_multigroup_imc.py \
        $BASE_ARGS \
        --checkpoint-file "$CHECKPOINT"
fi

EXIT_CODE=$?

# ─── Post-run handling ───────────────────────────────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    touch "$DONE_FLAG"
    echo "Run completed successfully."
elif [ -f "$CHECKPOINT" ]; then
    echo "Exited with code $EXIT_CODE – resubmitting from checkpoint."
    qsub "$SCRIPT"
else
    echo "Exited with code $EXIT_CODE and no checkpoint found; not resubmitting."
fi
