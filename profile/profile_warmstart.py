"""
Profile pi05_warmstart inference: VLM prefix vs per-denoising-step timing.
Records both cold-start (first call, no prior) and warm-start (subsequent calls).

Run with nsys (see profile_warmstart.sh), or standalone for quick timing:
    python profile/profile_warmstart.py --checkpoint 030000
"""

import argparse
import time

import torch
import torch.cuda.nvtx as nvtx

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="030000")
parser.add_argument("--n_warmup", type=int, default=3, help="warmup runs (discarded)")
parser.add_argument("--n_runs", type=int, default=5, help="timed runs per mode")
args = parser.parse_args()

CHECKPOINT_PATH = f"./outputs/pi05_warmstart_libero/checkpoints/{args.checkpoint}/pretrained_model"
DEVICE = "cuda"

# ── load policy ───────────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT_PATH}")
from lerobot.policies.pi05_warmstart.modeling_pi05_warmstart import PI05WarmstartPolicy

policy = PI05WarmstartPolicy.from_pretrained(CHECKPOINT_PATH)
policy = policy.to(DEVICE).eval()
cfg = policy.config
print(f"  chunk_size={cfg.chunk_size}  n_action_steps={cfg.n_action_steps}  "
      f"num_inference_steps={cfg.num_inference_steps}  execution_horizon={cfg.execution_horizon}")

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
    policy.reset()  # cold start each warmup to exercise the full path
    with torch.no_grad():
        policy.select_action(make_dummy_batch())
torch.cuda.synchronize()

# ── cold-start timing ─────────────────────────────────────────────────────────
print(f"\n[COLD START] ({args.n_runs} runs) — full {cfg.num_inference_steps} denoising steps")
cold_times = []
for i in range(args.n_runs):
    policy.reset()  # clears _prev_chunk → triggers cold start
    batch = make_dummy_batch()

    nvtx.range_push(f"cold_start_run_{i}")
    t0 = sync_ms()
    with torch.no_grad():
        policy.select_action(batch)
    t1 = sync_ms()
    nvtx.range_pop()

    cold_times.append(t1 - t0)
    print(f"  run {i}: {cold_times[-1]:.1f} ms")

avg_cold = sum(cold_times) / len(cold_times)
print(f"  avg: {avg_cold:.1f} ms  (~{1000/avg_cold:.1f} Hz)")

# ── warm-start timing ─────────────────────────────────────────────────────────
n_warmstart_steps = cfg.n_warmstart_steps if hasattr(cfg, "n_warmstart_steps") else cfg.num_inference_steps
print(f"\n[WARM START] ({args.n_runs} runs) — {n_warmstart_steps} denoising steps from shifted prior")

# Prime with one cold-start call so _prev_chunk is set
policy.reset()
with torch.no_grad():
    policy.select_action(make_dummy_batch())
torch.cuda.synchronize()

warm_times = []
for i in range(args.n_runs):
    # Clear only the action queue so select_action triggers a new model call,
    # but keep _prev_chunk so the warm-start path (shifted prior) is used.
    policy._action_queue.clear()
    batch = make_dummy_batch()

    nvtx.range_push(f"warm_start_run_{i}")
    t0 = sync_ms()
    with torch.no_grad():
        policy.select_action(batch)
    t1 = sync_ms()
    nvtx.range_pop()

    warm_times.append(t1 - t0)
    print(f"  run {i}: {warm_times[-1]:.1f} ms")

avg_warm = sum(warm_times) / len(warm_times)
print(f"  avg: {avg_warm:.1f} ms  (~{1000/avg_warm:.1f} Hz)")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Cold start avg : {avg_cold:.1f} ms")
print(f"Warm start avg : {avg_warm:.1f} ms")
print(f"Speedup        : {avg_cold/avg_warm:.2f}x")
print(f"\nNVTX ranges visible in nsys timeline:")
print(f"  vlm_prefix        — PaliGemma encoding + KV cache")
print(f"  denoise_step_00   — each action-expert denoising step")
print(f"  cold_start_run_N  — full cold inference")
print(f"  warm_start_run_N  — full warm inference")
