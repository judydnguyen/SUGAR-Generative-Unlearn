#!/bin/bash

# --------------------------------------------------------------------------------
# ------- Our Approach ----------------------------------------------------------
# --------------------------------------------------------------------------------

set -e  # Exit script on error

GPU_ID=1
EPOCH=300

# Ensure the evaluate directory exists
mkdir -p evaluate/logs
pushd evaluate > /dev/null  # Enter evaluate directory

for n_id in 5; do
    for MODE in ours; do
        echo "Evaluating $MODE with forget $n_id"

        if [ "$MODE" = "ours" ]; then
            MODEL_PATH="../experiments/${MODE}_celebahq-forget-${n_id}_lmbda_50.0/checkpoints/generator_epoch_${EPOCH}.pkl"
        else
            MODEL_PATH="../experiments/${MODE}_celebahq-forget-${n_id}/checkpoints/generator_epoch_${EPOCH}.pkl"
        fi
        # MODEL_PATH="../saved_ckpts/celebahq/${MODE}/forget_${n_id}_generator_epoch_950.pkl"

        RETAIN_DATA_PATH="../data/exp01/celebahq-retain-10"
        FORGET_DATA_PATH="../data/exp01/celebahq-forget-${n_id}"
        EXP_TAG="${MODE}_celebahq_forget_${n_id}"
        LOG_PATH="./logs/${EXP_TAG}.log"

        # Run evaluation script
        CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate_all_metrics.py --exp "$EXP_TAG" \
            --unlearn_model_path "$MODEL_PATH" \
            --inversion goae \
            --inversion_image_path "$FORGET_DATA_PATH" \
            --batch_size 8 \
            --valid_image_path "$RETAIN_DATA_PATH" >> "$LOG_PATH"

        echo "Done evaluating $EXP_TAG, see $LOG_PATH for details"
    done
done

popd > /dev/null  # Exit evaluate directory
