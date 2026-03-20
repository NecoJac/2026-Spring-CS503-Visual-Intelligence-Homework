#!/bin/bash
#SBATCH --job-name=you_job         # Change as needed
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Adjust CPU allocation if needed
#SBATCH --output=interactive_job2.out    # Output log file
#SBATCH --error=interactive_job2.err     # Error log file

export CONFIG_FILE=./cfgs/nanoMaskGIT/tinystories_d8w512.yaml
export WANDB=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
export NUM_GPUS=2

source /home/sjiang/.bashrc
conda activate nanofm
wandb login $WANDB
export WANDB_API_KEY=$WANDB && OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS run_training.py --config $CONFIG_FILE
