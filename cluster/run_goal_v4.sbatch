#!/bin/bash
#SBATCH -J GOAL-bs64
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate goal

export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HOME=/home/scc/pb24111693
export DEEPSPEED_CONFIG_DIR=/home/scc/pb24111693/tmp/.deepspeed
mkdir -p $DEEPSPEED_CONFIG_DIR

cd ~/projects/GOAL

# GPU check
nvidia-smi

# NFS warmup
echo "Warming NFS cache on compute node..."
find ~/GOAL_data/DCI/ -type f 2>/dev/null | wc -l

# Train
echo "Starting training..."
srun --export=ALL python goal.py \
    --dataset datasets/DCI_segment_sim_bbox_del_org.json \
    --batch_size 64 \
    --num_workers 0 \
    --epochs 10 \
    --output_dir finetune_out_v4
