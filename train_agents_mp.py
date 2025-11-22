"""
Training script for comparing:
- FP-MP regularized PPO
- Baseline PPO

on Pursuit Gridworld.

Prerequisites:
    - utils.py must be in the same directory, containing definitions for:
      ENV_KWARGS / make_vec_env / collect_features_from_model /
      compute_cov_spectrum / save_spectrum_plot / evaluate_model
"""

import os
import numpy as np

import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from utils import (
    make_vec_env,
    collect_features_from_model,
    compute_cov_spectrum,
    save_spectrum_plot,
    evaluate_model,
)


# =========================================================
# 1. Baseline FeatureExtractor: plain MLP
# =========================================================

class PlainFeatureExtractor(BaseFeaturesExtractor):
    """
    Simple feature extractor: flatten observation → MLP → ReLU.
    Used for both Baseline and FP-MP models to ensure fair comparison.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__(observation_space, features_dim)
        obs_shape = observation_space.shape
        self.obs_dim = int(np.prod(obs_shape))

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


# =========================================================
# 2. Free-Probability regularization Callback based on MP distribution
# =========================================================

class MPFreeProbRegularizer(BaseCallback):
    """
    Free-Probability regularization using Marchenko-Pastur (MP) distribution.
    
    Samples one minibatch of observations from the rollout buffer,
    computes the feature matrix H (B, d), and fits the MP distribution's
    bulk edge λ_+ to the eigenvalue spectrum of covariance C = H^T H / B.
    
    Parameters:
    - d: feature dimension
    - B: batch size
    - q = d / B (aspect ratio)
    - σ^2 ~ mean eigenvalue (trace / d)
    - λ_+ = σ^2 * (1 + sqrt(q))^2 (bulk edge)
    
    For spike components exceeding λ_+, defines the penalty:
    
        L_FP = fp_weight * mean( max(0, λ_i - λ_+)^2 )
    
    Performs one gradient update step on features_extractor parameters.
    This regularization encourages the feature covariance to follow the
    MP distribution, reducing overfitting and improving generalization.
    """

    def __init__(
        self,
        batch_size: int = 1024,
        fp_lr: float = 1e-4,
        fp_weight: float = 1.0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.batch_size = batch_size
        self.fp_lr = fp_lr
        self.fp_weight = fp_weight
        self.fp_optimizer = None

    def _on_training_start(self) -> None:
        # Only update features_extractor parameters (not policy/value networks)
        params = list(self.model.policy.features_extractor.parameters())
        self.fp_optimizer = torch.optim.Adam(params, lr=self.fp_lr)

    def _on_step(self) -> bool:
        """
        BaseCallback abstract method. Does nothing here.
        FP regularization is executed in _on_rollout_end.
        """
        return True

    def _on_rollout_end(self) -> bool:
        """
        Called after each rollout. Performs one FP regularization step here.
        """
        # Sample one minibatch from rollout_buffer
        gen = self.model.rollout_buffer.get(self.batch_size)

        try:
            rollout_data = next(gen)
        except StopIteration:
            # Skip if batch cannot be retrieved
            return True

        obs = rollout_data.observations.to(self.model.device)  # (B, *obs_shape)

        # Feature extraction (with gradients enabled)
        feats = self.model.policy.extract_features(obs)  # (B, d' or (B,k,h))
        feats = feats.reshape(feats.shape[0], -1)        # (B, d)
        B, d = feats.shape

        if B < 2 or d < 2:
            return True

        # Center features (subtract mean)
        feats_centered = feats - feats.mean(dim=0, keepdim=True)

        # Covariance matrix C = (1/B) H^T H  (d, d)
        C = (feats_centered.T @ feats_centered) / float(B)

        # Eigenvalues (ascending order) - only Hermitian eigenvalues
        evals = torch.linalg.eigvalsh(C)  # (d,)
        # Guard against numerical instability
        evals = torch.clamp(evals, min=0.0)

        # MP distribution parameters
        trace = torch.sum(evals)
        sigma2_hat = trace / float(d)          # Estimate of noise variance
        q = d / float(B)                       # Aspect ratio
        q_t = torch.tensor(q, device=evals.device, dtype=evals.dtype)

        lambda_plus = sigma2_hat * (1.0 + torch.sqrt(q_t)) ** 2  # Bulk edge

        # Penalty for spike components (only for λ_i > λ_+)
        spikes = torch.clamp(evals - lambda_plus, min=0.0)
        fp_loss = self.fp_weight * torch.mean(spikes ** 2)

        if not torch.isfinite(fp_loss):
            # Skip if something is wrong
            return True

        # Backpropagate and update features_extractor by one step
        self.fp_optimizer.zero_grad()
        fp_loss.backward()
        self.fp_optimizer.step()

        if self.logger is not None:
            self.logger.record("train/fp_mp_loss", fp_loss.item())
            self.logger.record("train/fp_lambda_plus", lambda_plus.item())
            num_spikes = (spikes > 0).sum().item()
            self.logger.record("train/fp_num_spikes", num_spikes)

        if self.verbose > 0:
            print(
                f"[FP-MP] loss={fp_loss.item():.3e}, "
                f"lambda_plus={lambda_plus.item():.3e}, "
                f"num_spikes={int((spikes > 0).sum())}"
            )

        return True


# =========================================================
# 3. Baseline PPO training
# =========================================================

def train_baseline(total_timesteps: int = 400_000):
    """
    Train a baseline PPO agent without Free-Probability regularization.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = make_vec_env(num_envs=4)

    policy_kwargs = dict(
        features_extractor_class=PlainFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            hidden_dim=512,
        ),
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("models/ppo_pursuit_baseline")
    env.close()
    print("Baseline training finished.")

    # ---- Quantitative evaluation (same environment) ----
    m_steps, sr, steps = evaluate_model(model, n_episodes=50)
    print(f"[Baseline] mean_steps={m_steps:.1f}, success_rate={sr:.2f}")

    np.savez(
        "results/metrics_baseline.npz",
        mean_steps=m_steps,
        success_rate=sr,
        steps=steps,
    )

    # ---- Spectral analysis ----
    X = collect_features_from_model(model, max_steps=3000)
    np.save("models/features_baseline.npy", X)
    evals, _ = compute_cov_spectrum(X)
    save_spectrum_plot(
        evals,
        out_path="figs/eigs_baseline.png",
        title="Eigenvalues (Baseline PPO features)",
    )


# =========================================================
# 4. FP-MP regularized PPO training
# =========================================================

def train_fp_mp(
    total_timesteps: int = 400_000,
    fp_lr: float = 1e-4,
    fp_weight: float = 1.0,
    fp_batch_size: int = 1024,
):
    """
    Train a PPO agent with Free-Probability (MP distribution) regularization.
    
    Parameters
    ----------
    total_timesteps : int
        Total number of PPO training steps
    fp_lr : float
        Learning rate for Adam optimizer used to update features_extractor
        in FP regularization
    fp_weight : float
        Strength of spike eigenvalue penalty
    fp_batch_size : int
        Batch size for spectrum estimation (should satisfy d << B for
        valid MP distribution approximation)
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = make_vec_env(num_envs=4)

    policy_kwargs = dict(
        features_extractor_class=PlainFeatureExtractor,  # Same base as Baseline
        features_extractor_kwargs=dict(
            features_dim=256,
            hidden_dim=512,
        ),
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
    )

    fp_callback = MPFreeProbRegularizer(
        batch_size=fp_batch_size,
        fp_lr=fp_lr,
        fp_weight=fp_weight,
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=fp_callback)
    model.save("models/ppo_pursuit_fp_mp")
    env.close()
    print("FP-MP-regularized PPO training finished.")

    # ---- Quantitative evaluation (same environment) ----
    m_steps, sr, steps = evaluate_model(model, n_episodes=50)
    print(f"[FP-MP] mean_steps={m_steps:.1f}, success_rate={sr:.2f}")

    np.savez(
        "results/metrics_fp_mp.npz",
        mean_steps=m_steps,
        success_rate=sr,
        steps=steps,
    )

    # ---- Spectral analysis ----
    X = collect_features_from_model(model, max_steps=3000)
    np.save("models/features_fp_mp.npy", X)
    evals, _ = compute_cov_spectrum(X)
    save_spectrum_plot(
        evals,
        out_path="figs/eigs_fp_mp.png",
        title="Eigenvalues (FP-MP PPO features)",
    )


# =========================================================
# 5. main (train FP-MP first, then Baseline)
# =========================================================

def main():
    os.makedirs("logs", exist_ok=True)

    print("=== Train FP-MP-regularized PPO ===")
    train_fp_mp(
        total_timesteps=400_000,
        fp_lr=1e-4,
        fp_weight=1.0,
        fp_batch_size=1024,
    )

    print("\n=== Train Baseline PPO ===")
    train_baseline(total_timesteps=400_000)


if __name__ == "__main__":
    main()  # voila!
