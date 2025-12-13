#!/bin/bash
# Submit all training jobs to SLURM
# Usage: ./submit_all_jobs.sh [--account=ACCOUNT_NAME]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parse account argument if provided
ACCOUNT_ARG=""
if [ "$1" != "" ]; then
    ACCOUNT_ARG="$1"
fi

echo "Submitting all training jobs..."
if [ "$ACCOUNT_ARG" != "" ]; then
    echo "Using account: $ACCOUNT_ARG"
fi
echo "=================================="

# Submit DNC jobs
echo "Submitting DNC jobs..."
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_dnc_O0.sh
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_dnc_O75.sh
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_dnc_O200.sh

# Submit DNC Greedy jobs
echo "Submitting DNC Greedy jobs..."
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_dnc_greedy_O0.sh
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_dnc_greedy_O75.sh
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_dnc_greedy_O200.sh


# Submit MinMax jobs
echo "Submitting MinMax jobs..."
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_minmax_O0.sh
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_minmax_O75.sh
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_minmax_O200.sh

# Submit Baseline jobs
echo "Submitting Baseline jobs..."
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_baseline_O0.sh
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_baseline_O75.sh
sbatch $ACCOUNT_ARG $SCRIPT_DIR/slurm_train_baseline_O200.sh

echo ""
echo "All jobs submitted! Check status with: squeue -u $USER"

