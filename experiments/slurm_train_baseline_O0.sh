#!/bin/bash
#SBATCH --job-name=baseline_O0
#SBATCH --account=def-account-name  # Replace with your Compute Canada account
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/ift6162-project/baseline_O0/slurm_%j.out
#SBATCH --error=/scratch/%u/ift6162-project/baseline_O0/slurm_%j.err

# Load modules (adjust for your cluster)
module load StdEnv/2023

# Activate virtual environment
source ~/ift6162_venv/bin/activate

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/..

# Run baseline (no GPU needed)
python experiments/train_single_config.py \
    --algorithm baseline \
    --O 0 \
    --n_items 20 \
    --output_dir $SCRATCH/ift6162-project

echo "Job completed at $(date)"

