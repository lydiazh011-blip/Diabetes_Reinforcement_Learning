import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from rl_env import DiabetesEnv


SEED = 2025
PATIENT_NAME = "adult#006"

TOTAL_TIMESTEPS = 50000
CAP_LIST = [
    0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.10, 0.11, 0.12, 0.13, 0.14, 0.15
]

SAVE_DIR = f"sac_models/{PATIENT_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(patient_name, insulin_cap, seed):
    env = PaperEnv(
        patient_name=patient_name,
        insulin_cap=insulin_cap,
        seed=seed
    )
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99
    )
    return env


class SimpleProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, desc):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.desc = desc
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc=self.desc,
            leave=True
        )

    def _on_step(self) -> bool:
        self.pbar.update(self.model.env.num_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()


def train_single_cap(cap):
    model_path = f"{SAVE_DIR}/sac_cap_{cap:.2f}.zip"
    vecnorm_path = f"{SAVE_DIR}/sac_vecnorm_cap_{cap:.2f}.pkl"

    if os.path.exists(model_path) and os.path.exists(vecnorm_path):
        print(f"SAC cap={cap:.2f} skip")
        return

    print(f"Training SAC | patient={PATIENT_NAME} | cap={cap:.2f}")

    env = make_env(PATIENT_NAME, cap, SEED)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        seed=SEED,
        verbose=0
    )

    progress_cb = SimpleProgressCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        desc=f"SAC cap={cap:.2f}"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=progress_cb
    )

    model.save(model_path)
    env.save(vecnorm_path)
    env.close()

    print(f"✅ Finished SAC cap={cap:.2f}")


if __name__ == "__main__":

    print("=" * 70)
    print("SAC pre-training with visible progress")
    print(f"Patient: {PATIENT_NAME}")
    print(f"Total timesteps per cap: {TOTAL_TIMESTEPS}")
    print(f"Caps: {CAP_LIST}")
    print("=" * 70)

    set_global_seed(SEED)

    for cap in CAP_LIST:
        train_single_cap(cap)

    print("\n" + "=" * 70)
    print("All SAC models trained.")
    print(f"Saved in: {SAVE_DIR}")
    print("=" * 70)
