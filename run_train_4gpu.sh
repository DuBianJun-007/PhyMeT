#!/bin/bash
# ============================================================
# MemISTD Small Target Detection - 4 GPU Distributed Training
# ============================================================

# ==================== Configuration ====================
# Modify these paths according to your environment

# Virtual environment path (choose one method)
# Method 1: Python venv
VENV_PATH="./.venv"

# Method 2: Conda environment name
# CONDA_ENV="your_conda_env_name"

# Project root directory
PROJECT_ROOT="./MemISTD"

# Config file path
CONFIG_FILE="./config/memistd_small_target_config.yaml"

# Number of GPUs
NUM_GPUS=4

# ==================== Setup Environment ====================

echo "=========================================="
echo "MemISTD 4-GPU Distributed Training"
echo "=========================================="

# Change to project directory
cd "$PROJECT_ROOT" || exit 1
echo "Project directory: $(pwd)"

# Activate virtual environment
if [ -n "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    . "$VENV_PATH/bin/activate"
elif [ -n "$CONDA_ENV" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    conda activate "$CONDA_ENV"
else
    echo "Warning: No virtual environment specified, using system Python"
fi

# Check Python version
echo "Python version: $(python --version)"

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ==================== Start Training ====================

echo ""
echo "=========================================="
echo "Starting distributed training with $NUM_GPUS GPUs"
echo "Config file: $CONFIG_FILE"
echo "=========================================="
echo ""

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Get Python path from virtual environment
PYTHON_BIN="$VENV_PATH/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi
echo "Using Python: $PYTHON_BIN"

# Start distributed training using virtual environment's torchrun
TORCHRUN_BIN="$VENV_PATH/bin/torchrun"
if [ ! -f "$TORCHRUN_BIN" ]; then
    TORCHRUN_BIN="torchrun"
fi
echo "Using torchrun: $TORCHRUN_BIN"

$TORCHRUN_BIN \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_dist.py \
    --config "$CONFIG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with error code: $?"
    echo "=========================================="
    exit 1
fi
