#!/bin/bash
#SBATCH --job-name=minmax_O0
#SBATCH --account=def-account-name  # Replace with your Compute Canada account
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/%u/ift6162-project/minmax_O0/slurm_%j.out
#SBATCH --error=/scratch/%u/ift6162-project/minmax_O0/slurm_%j.err

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
    --O 0 \
    --seed 42 \
    --n_items 20 \
    --max_episodes 70000 \
    --output_dir $SCRATCH/ift6162-project

echo "Job completed at $(date)"

