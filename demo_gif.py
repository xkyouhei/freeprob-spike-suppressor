import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

from stable_baselines3 import PPO
from utils import rollout_frames, evaluate_model


def add_labels_to_stacked_frame(
    stacked: np.ndarray,
    baseline_text: str,
    fp_text: str,
) -> np.ndarray:
    """
    Draw title bars on the vertically stacked frame.
    Top: baseline_text, Bottom: fp_text
    """
    img = Image.fromarray(stacked)
    draw = ImageDraw.Draw(img)

    w, h = img.size
    half_h = h // 2

    # Draw black bars at top and middle
    draw.rectangle([0, 0, w, 28], fill=(0, 0, 0))
    draw.rectangle([0, half_h, w, half_h + 28], fill=(0, 0, 0))

    draw.text((10, 6), baseline_text, fill=(255, 255, 255))
    draw.text((10, half_h + 6), fp_text, fill=(255, 255, 255))

    return np.array(img)


def main():
    os.makedirs("figs", exist_ok=True)

    baseline_path = "models/ppo_pursuit_baseline.zip"
    fp_path = "models/ppo_pursuit_fp_mp.zip"

    if not os.path.exists(baseline_path):
        raise FileNotFoundError("Baseline model not found. Run train_agents_mp.py first.")
    if not os.path.exists(fp_path):
        raise FileNotFoundError("FP-MP model not found. Run train_agents_mp.py first.")

    baseline_model = PPO.load(baseline_path)
    fp_model = PPO.load(fp_path)

    # ---- Quantitative evaluation (same environment) ----
    base_m_steps, base_sr, _ = evaluate_model(baseline_model, n_episodes=50)
    fp_m_steps, fp_sr, _ = evaluate_model(fp_model, n_episodes=50)

    # ---- Roll out from exactly the same initial state ----
    SEED = 20251122  # Any fixed value works

    print("=== Rollout Baseline PPO (same seed) ===")
    base_frames, base_steps, base_all_caught = rollout_frames(
        baseline_model, seed=SEED
    )

    print("=== Rollout FP-MP PPO (same seed) ===")
    fp_frames, fp_steps, fp_all_caught = rollout_frames(
        fp_model, seed=SEED
    )

    base_status = "ALL CAUGHT" if base_all_caught else "TIMEOUT"
    fp_status = "ALL CAUGHT" if fp_all_caught else "TIMEOUT"

    baseline_text = (
        f"Baseline PPO | {base_status} | ep_steps={base_steps} "
        f"| mean_steps={base_m_steps:.1f}, SR={base_sr:.2f}"
    )
    fp_text = (
        f"FP-MP PPO | {fp_status} | ep_steps={fp_steps} "
        f"| mean_steps={fp_m_steps:.1f}, SR={fp_sr:.2f}"
    )

    print("Baseline:", baseline_text)
    print("FP-MP   :", fp_text)

    # ---- Align frame counts and stack vertically ----
    n_frames = min(len(base_frames), len(fp_frames))
    base_frames = base_frames[:n_frames]
    fp_frames = fp_frames[:n_frames]

    stacked_frames = []
    for fb, ff in zip(base_frames, fp_frames):
        h1, w1, _ = fb.shape
        h2, w2, _ = ff.shape
        h = min(h1, h2)
        w = min(w1, w2)
        fb = fb[:h, :w, :]
        ff = ff[:h, :w, :]

        stacked = np.concatenate([fb, ff], axis=0)
        stacked_labeled = add_labels_to_stacked_frame(
            stacked, baseline_text, fp_text
        )
        stacked_frames.append(stacked_labeled)

    out_path = "figs/demo_baseline_vs_fp_mp.gif"
    imageio.mimsave(out_path, stacked_frames, fps=10)
    print(f"Saved vertical comparison GIF with labels -> {out_path}")


if __name__ == "__main__":
    main()
