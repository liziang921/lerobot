#!/bin/bash

#SBATCH --job-name=pi05_warmstart
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liziang2027@gmail.com

#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=768G
#SBATCH --time=48:00:00

#SBATCH --chdir=/gscratch/krishna/ziangli/Research/lerobot
#SBATCH --export=all
#SBATCH --output=/gscratch/krishna/ziangli/Research/lerobot/train_logs/train_output_%j.log
#SBATCH --error=/gscratch/krishna/ziangli/Research/lerobot/train_logs/train_error_%j.log

source /mmfs1/gscratch/krishna/ziangli/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export MUJOCO_GL=egl
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/mmfs1/gscratch/krishna/ziangli/wandb_cache
export WANDB_CACHE_DIR=/mmfs1/gscratch/krishna/ziangli/wandb_cache
export WANDB_DATA_DIR=/mmfs1/gscratch/krishna/ziangli/wandb_cache
mkdir -p /mmfs1/gscratch/krishna/ziangli/wandb_cache

# === RESUME ===
accelerate launch \
  --num_machines=1 \
  --num_processes=8 \
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
    --wandb.run_id=45vnw35y \
    --wandb.disable_artifact=true \
    --job_name=pi05_warmstart_libero \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_90,libero_10 \
    --eval_freq=0 \
    --steps=50000 \
    --batch_size=8 \
    --save_freq=5000 \
    --resume=true \
    --config_path=./outputs/pi05_warmstart_libero/checkpoints/030000/pretrained_model/train_config.json


# === ORIGINAL: train from scratch ===
# accelerate launch \
#   --num_machines=1 \
#   --num_processes=8 \
#   --mixed_precision=bf16 \
#   -m lerobot.scripts.lerobot_train \
#     --policy.type=pi05_warmstart \
#     --policy.device=cuda \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.repo_id=ZiangLi/pi05_warmstart_libero \
#     --policy.push_to_hub=false \
#     --dataset.repo_id=HuggingFaceVLA/libero \
#     --output_dir=./outputs/pi05_warmstart_libero \
#     --policy.dtype=bfloat16 \
#     --policy.freeze_vision_encoder=true \
#     --policy.train_expert_only=false \
#     --policy.gradient_checkpointing=true \
#     --wandb.enable=true \
#     --wandb.entity=Earendel \
#     --wandb.project=FlashVLA \
#     --wandb.disable_artifact=true \
#     --job_name=pi05_warmstart_libero \
#     --env.type=libero \
#     --env.task=libero_spatial,libero_object,libero_goal,libero_90,libero_10 \
#     --eval_freq=0 \
#     --steps=30000 \
#     --batch_size=8 \
#     --save_freq=2000
