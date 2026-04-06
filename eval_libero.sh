#!/bin/bash

source /mmfs1/gscratch/krishna/ziangli/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT=${1:-030000}
POLICY_PATH=./outputs/pi05_warmstart_libero/checkpoints/${CHECKPOINT}/pretrained_model

lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=${POLICY_PATH} \
  --output_dir=./eval_logs/${CHECKPOINT}/ \
  --policy.n_action_steps=10 \
  --env.max_parallel_tasks=1
