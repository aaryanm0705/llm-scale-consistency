#!/bin/bash
#
# SLURM Job Submission Script
# Evaluation Method: Vanilla Sequential (Chainwise)
#
#SBATCH --job-name=consistency_sequential
#SBATCH --output=logs/consistency_sequential_%j.out
#SBATCH --error=logs/consistency_sequential_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "=========================================="
echo "Vanilla Sequential (Chainwise) Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo

module load python/3.10
source activate llm

echo "Starting vanilla sequential evaluation..."
cd /dss/dsshome1/0C/ra96duk2/thesis_llm_evaluation
python scripts/evaluate_consistency_vanilla_sequential.py

echo
echo "Job completed: $(date)"
