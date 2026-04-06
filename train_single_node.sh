#!/bin/bash
source /mmfs1/gscratch/krishna/ziangli/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
  --num_machines=1 \
  --num_processes=4 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
    --policy.type=pi05_warmstart \
    --policy.device=cuda \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.repo_id=ZiangLi/pi05_warmstart_libero \
    --policy.push_to_hub=false \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --output_dir=./outputs/pi05_warmstart_libero \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.entity=Earendel \
    --wandb.project=FlashVLA \
    --job_name=pi05_warmstart_libero \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --eval_freq=1000 \
    --steps=6000 \
    --batch_size=8 \
    --save_freq=500
