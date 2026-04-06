#!/bin/bash

source /mmfs1/gscratch/krishna/ziangli/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
module load cuda
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT=${1:-030000}
OUTPUT_DIR=./profile/outputs/${CHECKPOINT}
mkdir -p ${OUTPUT_DIR}

echo "Profiling checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}/warmstart.nsys-rep"

nsys profile \
    --trace=cuda,nvtx \
    --output=${OUTPUT_DIR}/warmstart \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python profile/profile_warmstart.py --checkpoint ${CHECKPOINT} --n_warmup 3 --n_runs 3
