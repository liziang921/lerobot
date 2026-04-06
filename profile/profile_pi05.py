"""
Profile original pi05 inference: VLM prefix vs action expert (all denoising steps).

Run with nsys (see profile_pi05.sh), or standalone for quick timing:
    python profile/profile_pi05.py --policy_path lerobot/pi05_libero_finetuned
"""

import argparse
import time

import torch
import torch.cuda.nvtx as nvtx

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str, default="lerobot/pi05_libero_finetuned")
parser.add_argument("--n_warmup", type=int, default=3, help="warmup runs (discarded)")
parser.add_argument("--n_runs", type=int, default=5, help="timed runs")
args = parser.parse_args()

DEVICE = "cuda"

# ── load policy ───────────────────────────────────────────────────────────────
print(f"Loading policy: {args.policy_path}")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

policy = PI05Policy.from_pretrained(args.policy_path)
policy = policy.to(DEVICE).eval()
cfg = policy.config
print(f"  chunk_size={cfg.chunk_size}  n_action_steps={cfg.n_action_steps}  "
      f"num_inference_steps={cfg.num_inference_steps}")

# ── dummy batch ───────────────────────────────────────────────────────────────
def make_dummy_batch():
    return {
        "observation.images.image": torch.randn(
            1, 3, *cfg.image_resolution, device=DEVICE  # (B, C, H, W) — no time dim
        ),
        "observation.language.tokens": torch.zeros(
            1, cfg.tokenizer_max_length, dtype=torch.long, device=DEVICE
        ),
        "observation.language.attention_mask": torch.ones(
            1, cfg.tokenizer_max_length, dtype=torch.long, device=DEVICE
        ),
    }

def sync_ms():
    torch.cuda.synchronize()
    return time.perf_counter() * 1000

# ── warmup ────────────────────────────────────────────────────────────────────
print(f"\nWarming up ({args.n_warmup} runs)...")
for _ in range(args.n_warmup):
    policy.reset()
    with torch.no_grad():
        policy.select_action(make_dummy_batch())
torch.cuda.synchronize()

# ── timed runs ────────────────────────────────────────────────────────────────
print(f"\n[PI05] ({args.n_runs} runs) — {cfg.num_inference_steps} denoising steps")
times = []
for i in range(args.n_runs):
    policy.reset()
    policy._action_queue.clear()
    batch = make_dummy_batch()

    nvtx.range_push(f"pi05_run_{i}")
    t0 = sync_ms()
    with torch.no_grad():
        policy.select_action(batch)
    t1 = sync_ms()
    nvtx.range_pop()

    times.append(t1 - t0)
    print(f"  run {i}: {times[-1]:.1f} ms")

avg = sum(times) / len(times)
print(f"  avg: {avg:.1f} ms  (~{1000/avg:.1f} Hz)")
print(f"\nNVTX ranges visible in nsys timeline:")
print(f"  vlm_prefix         — PaliGemma encoding + KV cache")
print(f"  action_expert_all  — all denoising steps combined")
print(f"  denoise_step_00    — each individual denoising step")
print(f"  pi05_run_N         — full inference")
