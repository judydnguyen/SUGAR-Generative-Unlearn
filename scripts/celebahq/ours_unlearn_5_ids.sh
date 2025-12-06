#!/bin/bash
# Script to run unlearning experiment for 5 CelebAHQ identities

set -e  # Exit on error

# Configuration
GPU_ID=0
EXP_NAME="ours_celebahq-forget-5"

# Run unlearning experiment
CUDA_VISIBLE_DEVICES=${GPU_ID} python optimized_unlearn.py \
    --exp "${EXP_NAME}" \
    --inversion goae \
    --inversion_image_path data/exp01/celebahq-forget-5 \
    --valid_dir data/exp01/celebahq-retain-10 \
    --target extra \
    --target_d 20.0 \
    --target_idx 0 \
    --local \
    --nei \
    --batch_size 2 \
    --seed 0 \
    --lr 2e-4 \
    --lmbda 50.0 \
    --iter 1000 \
    --trigger_epochs 50 && {

    # Launch TensorBoard for experiment visualization
    TB_LOG_DIR="experiments/${EXP_NAME}"
    TB_PORT=6006
    
    echo "Starting TensorBoard for experiment: ${EXP_NAME}"
    
    # Try to open TensorBoard in a new terminal window
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "tensorboard --logdir ${TB_LOG_DIR} --port ${TB_PORT}; exec bash" &
    elif command -v xterm &> /dev/null; then
        xterm -e "tensorboard --logdir ${TB_LOG_DIR} --port ${TB_PORT}; exec bash" &
    elif command -v konsole &> /dev/null; then
        konsole -e bash -c "tensorboard --logdir ${TB_LOG_DIR} --port ${TB_PORT}; exec bash" &
    else
        # Fallback: run in background
        tensorboard --logdir "${TB_LOG_DIR}" --port "${TB_PORT}" &
    fi
    
    # Wait for TensorBoard to start, then open browser
    sleep 3
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:${TB_PORT}" 2>/dev/null || true
    elif command -v sensible-browser &> /dev/null; then
        sensible-browser "http://localhost:${TB_PORT}" 2>/dev/null || true
    elif command -v x-www-browser &> /dev/null; then
        x-www-browser "http://localhost:${TB_PORT}" 2>/dev/null || true
    else
        echo "TensorBoard started. Please open http://localhost:${TB_PORT} in your browser."
    fi
}