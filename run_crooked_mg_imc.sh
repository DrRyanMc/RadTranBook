#!/bin/bash
#$ -M rmcclarr@nd.edu
#$ -m abe
#$ -q long@@mcclarren
#$ -pe smp 64
#$ -N crooked_mg_imc

# ─── Configuration ─────────────────────────────────────────────────────────
N_GROUPS=16
NR=60
NZ=105
#this next line needs to do that calculation and not just be a string so that it is an integer when passed to the Python script.
NMax=$((1000000 * N_GROUPS))  # keep number of particles per group constant as N_GROUPS changes
# make Nboundary half NMax and ensure it is an integer
Ntarget=$((300000 * N_GROUPS))
Nboundary=$((400000 * N_GROUPS))

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
           --Ntotal-T-floor .1 \
           --T-emit-floor 0.1 \
           --mode publication
           --dt-initial 0.01 \
           --dt-max .1 \
           --dt-growth 1.025 \
           --bc-t-start 0.5 \
           --bc-t-end 0.5 \
           --bc-ramp-time 20.0 \
           --output-times 0.0001,0.00013257113655901095,0.00017575106248547912,0.00023299518105153718,0.00030888435964774815,0.00040949150623804275,0.0005428675439323859,0.0007196856730011522,0.0009540954763499944,0.0012648552168552957,0.0016768329368110084,0.0022229964825261957,0.0029470517025518097,0.003906939937054617,0.005179474679231213,0.006866488450042998,0.009102981779915217,0.012067926406393288,0.015998587196060572,0.021209508879201904,0.028117686979742307,0.03727593720314938,0.04941713361323833,0.0655128556859551,0.08685113737513521,0.1151395399326447,0.15264179671752334,0.20235896477251575,0.2682695795279725,0.35564803062231287,0.47148663634573945,0.6250551925273969,0.8286427728546842,1.0985411419875573,1.4563484775012443,1.9306977288832496,2.559547922699533,3.39322177189533,4.498432668969444,5.963623316594637,7.9060432109077015,10.481131341546853,13.89495494373136,18.420699693267164,24.420530945486497,32.3745754281764,42.91934260128778,56.89866029018293,75.43120063354607,100.0 \
           --checkpoint-every 20 \
           --max-events 10000000"

# ─── Run (restart if checkpoint exists) ────────────────────────────────────
if [ -f "$CHECKPOINT" ]; then
    echo "Restarting from checkpoint: $CHECKPOINT"
    python3 MG_IMC/crooked_pipe_multigroup_imc.py \
        $BASE_ARGS \
        --restart-file "$CHECKPOINT" \
        --checkpoint-file "$CHECKPOINT" > "$CHECKPOINT.log" 2>&1
else
    echo "Starting fresh run."
    python3 MG_IMC/crooked_pipe_multigroup_imc.py \
        $BASE_ARGS \
        --checkpoint-file "$CHECKPOINT" > "$CHECKPOINT.log" 2>&1
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
