#!/bin/bash
# GOAL 环境安装脚本 - 在集群 shell 中执行
set -e

echo "=== 安装 Miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
fi

source $HOME/miniconda3/etc/profile.d/conda.sh

echo "=== 创建 conda 环境 ==="
conda create -n goal python=3.10 -y
conda activate goal

echo "=== 安装 PyTorch (CUDA 12.1) ==="
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== 安装依赖 ==="
pip install transformers==4.46.3 lightning deepspeed wandb opencv-python pillow accelerate

echo "=== 验证 ==="
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "=== 环境安装完成 ==="
