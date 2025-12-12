#!/bin/bash
# Submit a single training job to SLURM
# Usage: ./submit_single_job.sh <script_name> [--account=ACCOUNT_NAME]
# Example: ./submit_single_job.sh slurm_train_dnc_O75.sh --account=def-username

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$1" == "" ]; then
    echo "Usage: $0 <script_name> [--account=ACCOUNT_NAME]"
    echo "Example: $0 slurm_train_dnc_O75.sh --account=def-username"
    exit 1
fi

SCRIPT_NAME="$1"
ACCOUNT_ARG=""

# Check if account argument is provided
if [ "$2" != "" ]; then
    ACCOUNT_ARG="$2"
fi

if [ ! -f "$SCRIPT_DIR/$SCRIPT_NAME" ]; then
    echo "Error: Script not found: $SCRIPT_DIR/$SCRIPT_NAME"
    exit 1
fi

echo "Submitting job: $SCRIPT_NAME"
if [ "$ACCOUNT_ARG" != "" ]; then
    echo "Using account: $ACCOUNT_ARG"
    sbatch $ACCOUNT_ARG $SCRIPT_DIR/$SCRIPT_NAME
else
    sbatch $SCRIPT_DIR/$SCRIPT_NAME
fi

echo "Job submitted! Check status with: squeue -u $USER"

