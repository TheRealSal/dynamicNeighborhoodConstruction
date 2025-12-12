#!/bin/bash
#SBATCH --job-name=minmax_O75
#SBATCH --account=def-account-name  # Replace with your Compute Canada account
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --output=/scratch/$USER/ift6162-project/minmax_O75/slurm_%j.out
#SBATCH --error=/scratch/$USER/ift6162-project/minmax_O75/slurm_%j.err

# Load modules (adjust for your cluster)
module load StdEnv/2023

# Activate virtual environment
source ~/ift6162_venv/bin/activate

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/..

# Run training
python experiments/train_single_config.py \
    --algorithm minmax \
    --O 75 \
    --seed 42 \
    --n_items 20 \
    --max_episodes 70000 \
    --output_dir $SCRATCH/ift6162-project

echo "Job completed at $(date)"

