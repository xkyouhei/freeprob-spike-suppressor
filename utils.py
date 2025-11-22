import os
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from pettingzoo.sisl import pursuit_v4
import supersuit as ss


# =========================================================
# Common environment configuration (shared across train / eval / demo)
# =========================================================
ENV_KWARGS = dict(
    max_cycles=300,        # Maximum steps per episode (slightly longer)
    x_size=16,             # Larger field size → slightly more challenging
    y_size=16,
    shared_reward=True,
    n_evaders=4,           # Number of evaders (prey)
    n_pursuers=6,          # Number of pursuers (predators)
    obs_range=7,           # Observation shape fixed to (7x7x3)
    n_catch=1,
    freeze_evaders=False,  # Evaders are mobile
    surround=False,
)


# =========================================================
# Environment utilities
# =========================================================

def make_vec_env(num_envs: int = 4):
    """
    Create a VecEnv for Stable-Baselines3 (used for training).
    """
    env = pursuit_v4.parallel_env(**ENV_KWARGS)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=num_envs,
        num_cpus=1,
        base_class="stable_baselines3",
    )
    return env


def make_aec_env(render: bool = False, seed: Optional[int] = None):
    """
    Create a PettingZoo AEC environment (used for evaluation and demos).

    Parameters
    ----------
    render : bool
        If True, creates environment with render_mode="rgb_array" (for demos).
    seed : Optional[int]
        Random seed passed to reset(). Using the same seed allows
        comparing Baseline / FP models from the same initial state.
    """
    if render:
        env = pursuit_v4.env(render_mode="rgb_array", **ENV_KWARGS)
    else:
        env = pursuit_v4.env(**ENV_KWARGS)

    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env


# =========================================================
# Spectral analysis
# =========================================================

def compute_cov_spectrum(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalue spectrum of the covariance matrix from feature matrix X (T, d).
    
    Returns
    -------
    evals : np.ndarray
        Sorted eigenvalues (ascending order)
    C : np.ndarray
        Covariance matrix (d, d)
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    T, d = X_centered.shape
    C = (X_centered.T @ X_centered) / T
    evals, _ = np.linalg.eigh(C)
    evals = np.sort(evals)
    return evals, C


def save_spectrum_plot(
    evals: np.ndarray,
    out_path: str,
    title: str = "Eigenvalues of covariance",
):
    """
    Save the eigenvalue sequence evals as a PNG plot.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(evals, marker="o", linestyle="none", markersize=3)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved spectrum plot -> {out_path}")


# =========================================================
# Feature extraction (from trained models)
# =========================================================

def collect_features_from_model(
    model,
    max_steps: int = 3000,
) -> np.ndarray:
    """
    Roll out a trained model in the environment and return
    the sequence of feature extractor outputs.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (T, d) where T is the number of steps
        and d is the feature dimension.
    """
    import torch

    env = make_aec_env(render=False)
    # Environment is already reset inside make_aec_env

    device = model.device
    feats_list: List[np.ndarray] = []
    step_count = 0

    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            action = None
        else:
            obs_arr = np.array(obs, dtype=np.float32)
            obs_tensor = torch.as_tensor(obs_arr, device=device).unsqueeze(0)
            with torch.no_grad():
                feats = model.policy.extract_features(obs_tensor)
                feats = feats.view(feats.shape[0], -1).cpu().numpy()[0]
            feats_list.append(feats)

            action, _ = model.predict(obs_arr, deterministic=True)

        env.step(action)
        step_count += 1
        if step_count >= max_steps:
            break

    env.close()
    X = np.stack(feats_list, axis=0)
    return X


# =========================================================
# Evaluation (success rate, mean steps)
# =========================================================

def evaluate_model(
    model,
    n_episodes: int = 50,
):
    """
    Evaluate the model for n_episodes in the common environment,
    returning mean steps, success rate, and per-episode step array.
    
    Returns
    -------
    mean_steps : float
        Average number of steps per episode
    success_rate : float
        Fraction of episodes where all evaders were caught
    steps : np.ndarray
        Array of step counts for each episode
    """
    import statistics

    steps_list: List[int] = []
    success_list: List[float] = []

    for _ in range(n_episodes):
        env = make_aec_env(render=False)
        raw_env = env.unwrapped
        step_count = 0

        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                action = None
            else:
                obs_arr = np.array(obs, dtype=np.float32)
                action, _ = model.predict(obs_arr, deterministic=True)
            env.step(action)
            step_count += 1

        all_caught = bool(getattr(raw_env.env, "is_terminal", False))
        env.close()

        steps_list.append(step_count)
        success_list.append(1.0 if all_caught else 0.0)

    mean_steps = statistics.mean(steps_list)
    success_rate = statistics.mean(success_list)
    return mean_steps, success_rate, np.array(steps_list, dtype=np.int32)


# =========================================================
# Demo rollout (for GIF generation)
# =========================================================

def rollout_frames(model, seed: Optional[int] = None):
    """
    Roll out one episode in the common environment and return
    RGB frame sequence, total step count, and all-caught flag.
    
    If seed is specified, the initial state is fixed, allowing
    comparison of Baseline / FP-MP models from the same starting state.
    
    Returns
    -------
    frames : List[np.ndarray]
        List of RGB frames (H, W, 3)
    step_count : int
        Total number of steps in the episode
    all_caught : bool
        True if all evaders were caught, False otherwise
    """
    env = make_aec_env(render=True, seed=seed)
    raw_env = env.unwrapped

    frames: List[np.ndarray] = []
    step_count = 0

    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            action = None
        else:
            obs_arr = np.array(obs, dtype=np.float32)
            action, _ = model.predict(obs_arr, deterministic=True)

        env.step(action)
        frame = env.render()
        frames.append(frame)
        step_count += 1

    all_caught = bool(getattr(raw_env.env, "is_terminal", False))
    env.close()
    return frames, step_count, all_caught
