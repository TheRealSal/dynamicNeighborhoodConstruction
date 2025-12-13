#!/bin/bash
#SBATCH --job-name=minmax_O75
#SBATCH --account=def-account-name  # Replace with your Compute Canada account
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/scratch/%u/ift6162-project/minmax_O75/slurm_%A_%a.out
#SBATCH --error=/scratch/%u/ift6162-project/minmax_O75/slurm_%A_%a.err
#SBATCH --array=0-2

# Define seeds
SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Load modules (adjust for your cluster)
module load StdEnv/2023

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# Activate virtual environment
source ~/ift6162_venv/bin/activate

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/..

# Run training
python experiments/train_single_config.py \
    --algorithm minmax \
    --O 75 \
    --seed $SEED \
    --n_actions 20 \
    --max_episodes 30000 \
    --output_dir $SCRATCH/ift6162-project

echo "Job completed at $(date)"
