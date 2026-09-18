#!/bin/bash

# --- SLURM DIRECTIVES ---
#SBATCH --job-name=GAT_Ensemble      # Name of the job in the queue
#SBATCH --output=/home/fader/logs/ensemble_%j.out # Standard output log (%j inserts the job ID)
#SBATCH --error=/home/fader/logs/ensemble_%j.err  # Error log file
#SBATCH --time=04:00:00               # Walltime limit (HH:MM:SS) - Diffusion takes time!
#SBATCH --partition=gpu               # The partition or queue with GPUs
#SBATCH --gres=gpu:1                  # Request exactly 1 GPU
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --ntasks-per-node=1          # Run 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores (needed for nnU-Net preprocessing)
#SBATCH --mem=32G                     # RAM allocation (at least 32GB recommended for 3D scans)

# --- ENVIRONMENT SETUP ---
# Create the log directory if it doesn't exist yet
export nnUNet_raw_data_base="/path/to/nnUNet_raw_data_base"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export RESULTS_FOLDER="/path/to/RESULTS_FOLDER"


# Load necessary modules provided by your cluster (uncomment and adjust if needed)
# module load cuda/12.1
# module load python/3.10

# Activate your specific conda or virtual environment
# Replace 'my_env' with the environment where torch, SimpleITK, and nnunet are installed


# --- RUN THE EXPERIMENT ---
echo "Starting GATSegDiff + nnU-Net Ensemble..."
echo "Running on node: $SLURM_NODELIST"

# Execute your python file (replace with your actual script name)
python "/path/to/GATnnUnetEnsemble.py"

echo "Job completed."

