#!/bin/bash
#
# SLURM Job Submission Script
# Evaluation Method: Expert One-Shot
# DEPENDENCY: Requires expert zeroshot baseline results in results/ directory
#
#SBATCH --job-name=consistency_expert_oneshot
#SBATCH --output=logs/consistency_expert_oneshot_%j.out
#SBATCH --error=logs/consistency_expert_oneshot_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "=========================================="
echo "Expert One-Shot Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "DEPENDENCY: Requires expert_zeroshot results"
echo

module load python/3.10
source activate llm

echo "Starting expert one-shot evaluation..."
cd "$(dirname "$0")/../.."
python scripts/evaluate_consistency_expert_oneshot.py

echo
echo "Job completed: $(date)"
