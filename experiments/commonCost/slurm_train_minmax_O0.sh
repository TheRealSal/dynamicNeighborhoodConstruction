#!/bin/bash
#SBATCH --job-name=minmax_O0
#SBATCH --account=def-account-name  # Replace with your Compute Canada account
#SBATCH --time=9:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch/%u/ift6162-project/minmax_O0/slurm_%A_%a.out
#SBATCH --error=/scratch/%u/ift6162-project/minmax_O0/slurm_%A_%a.err
#SBATCH --array=0-11

# Define arrays
SEEDS=(42 43 44)
ACTIONS=(20 5)
SCALE_OPTS=("--scale_reward" "")

# Calculate indices
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
ACTION_IDX=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))
SCALE_IDX=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))

# Get values
SEED=${SEEDS[$SEED_IDX]}
N_ACTIONS=${ACTIONS[$ACTION_IDX]}
SCALE_FLAG=${SCALE_OPTS[$SCALE_IDX]}

# Load modules (adjust for your cluster)
module load StdEnv/2023

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# Activate virtual environment
source ~/ift6162_venv/bin/activate

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/../..

# Run training
python experiments/train_single_config.py \
    --algorithm minmax \
    --O 0 \
    --seed $SEED \
    --n_actions $N_ACTIONS \
    $SCALE_FLAG \
    --max_episodes 30000 \
    --output_dir $SCRATCH/ift6162-project

echo "Job completed at $(date)"
