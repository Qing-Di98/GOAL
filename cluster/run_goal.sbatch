#!/bin/bash
#SBATCH -J GOAL-bs16
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate goal

# Fix deepspeed permission issue
export DEEPSPEED_CONFIG_DIR=~/tmp/.deepspeed
mkdir -p $DEEPSPEED_CONFIG_DIR
export HOME=/home/scc/pb24111693

# Wandb offline
export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_MODE=offline

# Fix CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/projects/GOAL

# NFS cache warmup
echo "Warming up NFS cache..."
find ~/GOAL_data/DCI/ -type f 2>/dev/null | wc -l
echo "NFS cache ready."

# Verify GPU
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Train
python goal.py \
    --dataset datasets/DCI_segment_sim_bbox_del_org.json \
    --batch_size 16 \
    --num_workers 0 \
    --epochs 10 \
    --output_dir finetune_out
