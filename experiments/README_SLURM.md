# SLURM Job Submission Guide

This guide explains how to run training jobs on Compute Canada using SLURM.

## Files Created

### Training Script
- `train_single_config.py`: Main training script that accepts command-line arguments

### SLURM Batch Scripts (9 total)
- `slurm_train_dnc_O0.sh`, `slurm_train_dnc_O75.sh`, `slurm_train_dnc_O200.sh`
- `slurm_train_minmax_O0.sh`, `slurm_train_minmax_O75.sh`, `slurm_train_minmax_O200.sh`
- `slurm_train_baseline_O0.sh`, `slurm_train_baseline_O75.sh`, `slurm_train_baseline_O200.sh`

### Helper Script
- `submit_all_jobs.sh`: Submits all 9 jobs at once

## Job Configuration

Each job has:
- **Time limit**: 14 hours
- **Memory**: 64GB
- **CPUs**: 56
- **Output directory**: `$SCRATCH/ift6162-project`
- **Job names**: `{algorithm}_O{O}` (e.g., `dnc_O75`, `minmax_O200`, `baseline_O0`)

## Usage

### Option 1: Submit All Jobs at Once

```bash
cd experiments
./submit_all_jobs.sh
```

### Option 2: Submit Individual Jobs

```bash
# Submit a single job
sbatch experiments/slurm_train_dnc_O75.sh

# Or submit multiple
sbatch experiments/slurm_train_dnc_O0.sh
sbatch experiments/slurm_train_dnc_O75.sh
sbatch experiments/slurm_train_dnc_O200.sh
```

### Option 3: Manual Submission with Custom Parameters

```bash
# Run training script directly (for testing)
python experiments/train_single_config.py \
    --algorithm dnc \
    --O 75 \
    --seed 42 \
    --n_items 20 \
    --max_episodes 30000 \
    --output_dir $SCRATCH/ift6162-project
```

## Before Running

### 1. Update Module Load Commands

Edit the SLURM scripts to match your cluster's module system:

```bash
# Check available modules
module avail python
module avail cuda

# Update in scripts if needed
module load python/3.8  # Adjust version
module load cuda/11.7    # Adjust version
```

### 2. Update Virtual Environment Path

Update the virtual environment path in all scripts:

```bash
# Change this line in each script:
source ~/ift6162_venv/bin/activate

# To your actual venv path, e.g.:
source /path/to/your/venv/bin/activate
```

### 3. Verify Output Directory

Ensure the output directory exists and is writable:

```bash
mkdir -p $SCRATCH/ift6162-project
```

## Output Structure

Results will be saved to:

```
$SCRATCH/ift6162-project/
├── dnc_O0/
│   └── seed42/
│       ├── Logs/
│       ├── Checkpoints/
│       └── Results/
│           ├── evaluation_results.json
│           ├── rewards.npy
│           ├── performance.png
│           └── ...
├── dnc_O75/
│   └── seed42/
│       └── ...
├── minmax_O0/
│   └── seed42/
│       └── ...
├── baseline_O0/
│   └── evaluation_results.json
└── ...
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Cancel a job
scancel <job_id>

# View job output
tail -f $SCRATCH/ift6162-project/dnc_O75/slurm_<job_id>.out
```

## Job Names

Each job has a unique name:
- `dnc_O0`, `dnc_O75`, `dnc_O200`
- `minmax_O0`, `minmax_O75`, `minmax_O200`
- `baseline_O0`, `baseline_O75`, `baseline_O200`

## Notes

- All jobs run for 14 hours maximum
- Results are saved incrementally during training
- Checkpoints are saved periodically
- Evaluation results are saved in JSON format

## Troubleshooting

### Job fails immediately
- Check module paths and versions
- Verify virtual environment path
- Check output directory permissions

### Out of memory
- Increase `--mem=16G` in SLURM scripts
- Or reduce `max_episodes` in training

### Module not found
- Run `module avail` to see available modules
- Update module load commands in scripts

### Path errors
- Ensure you're in the correct directory
- Check that `$SLURM_SUBMIT_DIR` points to experiments folder
- Verify project structure is correct

