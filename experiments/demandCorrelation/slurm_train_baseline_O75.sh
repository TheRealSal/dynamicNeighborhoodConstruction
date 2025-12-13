#!/bin/bash
#SBATCH --job-name=baseline_O75
#SBATCH --account=def-account-name  # Replace with your Compute Canada account
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/%u/ift6162-project/correlated_demand/baseline_O75/slurm_%A_%a.out
#SBATCH --error=/scratch/%u/ift6162-project/correlated_demand/baseline_O75/slurm_%A_%a.err
#SBATCH --array=0-35

# Define arrays
SEEDS=(42 43 44)
ACTIONS=(20 5)
SCALE_OPTS=("--scale_reward" "")
CORRELATIONS=(0.25 0.5 0.75)

# Calculate indices
# Total combinations = 3 * 2 * 2 * 3 = 36
# Layered calculation:
# Seed changes fastest (0-2)
# Then Actions (0-1)
# Then Scale (0-1)
# Then Correlation (0-2)

SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
ACTION_IDX=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))
SCALE_IDX=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))
CORR_IDX=$(( (SLURM_ARRAY_TASK_ID / 12) % 3 ))

# Get values
SEED=${SEEDS[$SEED_IDX]}
N_ACTIONS=${ACTIONS[$ACTION_IDX]}
SCALE_FLAG=${SCALE_OPTS[$SCALE_IDX]}
CORRELATION=${CORRELATIONS[$CORR_IDX]}

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

# Run baseline (no GPU needed)
python experiments/train_single_config.py \
    --algorithm baseline \
    --O 75 \
    --seed $SEED \
    --n_actions $N_ACTIONS \
    $SCALE_FLAG \
    --demand_dist standard \
    --demand_correlation $CORRELATION \
    --output_dir $SCRATCH/ift6162-project/correlated_demand

echo "Job completed at $(date)"
