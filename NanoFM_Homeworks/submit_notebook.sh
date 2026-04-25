#!/usr/bin/env bash
#SBATCH --job-name=notebook_run             # Change as needed
#SBATCH --time=10:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Adjust CPU allocation if needed
#SBATCH --output=interactive_job.out        # Output log file
#SBATCH --error=interactive_job.err         # Error log file

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate nanofm

cd notebooks
jupyter nbconvert --to notebook --execute \
    --ExecuteProcessor.timeout=-1 \
    --output completed_notebook.ipynb \
    CS503_FM_part4_nanoFlowMatching.ipynb