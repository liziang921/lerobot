#!/bin/bash

source /mmfs1/gscratch/krishna/ziangli/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
module load cuda
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POLICY_PATH=${1:-lerobot/pi05_libero_finetuned}
RUN_NAME=$(basename ${POLICY_PATH})
OUTPUT_DIR=./profile/outputs/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}

echo "Profiling policy: ${POLICY_PATH}"
echo "Output: ${OUTPUT_DIR}/pi05.nsys-rep"

nsys profile \
    --trace=cuda,nvtx \
    --output=${OUTPUT_DIR}/pi05 \
    --force-overwrite=true \
    python profile/profile_pi05.py --policy_path ${POLICY_PATH} --n_warmup 3 --n_runs 3
