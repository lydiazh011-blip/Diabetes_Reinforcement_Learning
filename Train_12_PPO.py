import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import torch
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Magni_Env import PaperEnv

SEED = 2025
PATIENT_NAME = "adult#002"
TOTAL_TIMESTEPS = 50000

CAP_LIST = [
    0.04, 0.05, 0.06, 0.07,
    0.08, 0.09, 0.10, 0.11,
    0.12, 0.13, 0.14, 0.15
]

SAVE_DIR = "models/adult#002"
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


def save_log_csv(env, filepath, cap):
    if not hasattr(env, "log_buffer"):
        print("env 中不存在 log_buffer，未保存 CSV")
        return

    if len(env.log_buffer) == 0:
        print("log_buffer 为空，未保存 CSV")
        return

    df = pd.DataFrame(env.log_buffer)
    df["cap"] = cap
    df.to_csv(filepath, index=False)
    print(f"📄 CSV saved: {filepath}")

def train_single_cap(cap):
    model_path = f"{SAVE_DIR}/agent_cap_{cap:.2f}.zip"
    vecnorm_path = f"{SAVE_DIR}/vec_norm_cap_{cap:.2f}.pkl"
    csv_path = f"{SAVE_DIR}/log_cap_{cap:.2f}.csv"

    if os.path.exists(model_path) and os.path.exists(vecnorm_path):
        print(f"✅ cap={cap:.2f} 已存在，跳过")
        return

    print(f"Training PPO | patient={PATIENT_NAME} | cap={cap:.2f}")

    env = make_env(PATIENT_NAME, cap, SEED)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=5e-3,
        seed=SEED,
        verbose=0
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True
    )

    model.save(model_path)
    env.save(vecnorm_path)
    raw_env = env.envs[0]  # DummyVecEnv -> PaperEnv
    save_log_csv(raw_env, csv_path, cap)

    env.close()
    print(f"✅ Finished cap={cap:.2f}")


if __name__ == "__main__":

    print("=" * 70)
    print("Paper-consistent PPO pre-training (Section 3.6)")
    print(f"Patient: {PATIENT_NAME}")
    print(f"Caps: {CAP_LIST}")
    print(f"Training steps per PPO: {TOTAL_TIMESTEPS}")
    print("=" * 70)

    set_global_seed(SEED)

    for cap in CAP_LIST:
        train_single_cap(cap)

    print("\n" + "=" * 70)
    print("🎉 All 12 PPO models trained successfully.")
    print(f"Saved in: {SAVE_DIR}")
    print("Next step: Dual PPO network search (Section 3.7)")
    print("=" * 70)

