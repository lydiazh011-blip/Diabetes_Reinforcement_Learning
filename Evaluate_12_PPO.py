import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env import PaperEnv


PATIENT_NAME = "adult#007"
MODEL_DIR = f"models/{PATIENT_NAME}"

CAP_LIST = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15]

N_RUNS = 10        
MAX_STEPS = 2400    
SEED_BASE = 2025

PRINT_EVERY = 100


def run_single_episode(cap, seed, show_steps=True):
    env = PaperEnv(
        patient_name=PATIENT_NAME,
        insulin_cap=cap,
        seed=seed
    )

    def env_fn():
        return PaperEnv(
            patient_name=PATIENT_NAME,
            insulin_cap=cap,
            seed=seed
        )

    venv = DummyVecEnv([env_fn])
    venv = VecNormalize.load(
        f"{MODEL_DIR}/vec_norm_cap_{cap:.2f}.pkl",
        venv
    )
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(
        f"{MODEL_DIR}/agent_cap_{cap:.2f}.zip",
        env=venv
    )

    obs, _ = env.reset()
    cgm_trace = []
    action_trace = []

    steps = range(MAX_STEPS)
    if show_steps:
        steps = tqdm(
            steps,
            desc=f"cap={cap:.2f} | seed={seed}",
            leave=False
        )

    for step in steps:
        obs_norm = venv.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_norm, deterministic=True)

        obs, _, done, truncated, _ = env.step(action)

        cgm = obs[0]
        cgm_trace.append(cgm)
        action_trace.append(action.item())

        if PRINT_EVERY and step % PRINT_EVERY == 0:
            tqdm.write(
                f"step={step:4d} | CGM={cgm:6.1f} | action={action.item():.4f}"
            )

        if done or truncated:
            break

    env.close()

    cgm = np.array(cgm_trace)
    action_trace = np.array(action_trace)
    tir = float(np.mean((cgm >= 70) & (cgm <= 180)))

    return tir, action_trace, cgm


def eval_cap(cap):
    tirs = []

    run_iter = tqdm(
        range(N_RUNS),
        desc=f"Runs for cap={cap:.2f}",
        leave=False
    )

    for k in run_iter:
        tir, actions, cgm = run_single_episode(
            cap,
            seed=SEED_BASE + k,
            show_steps=(k == 0)
        )
        tirs.append(tir)
        run_iter.set_postfix(TIR=f"{tir:.3f}")

    return float(np.mean(tirs)), float(np.std(tirs))

def plot_cap_vs_tir(results):
    caps = [r[0] for r in results]
    mean_tirs = [r[1] for r in results]
    std_tirs = [r[2] for r in results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        caps,
        mean_tirs,
        yerr=std_tirs,
        fmt='o-',
        capsize=4
    )
    plt.xlabel("Insulin cap (U/min)")
    plt.ylabel("TIR (70–180 mg/dL)")
    plt.title(f"Single PPO Performance vs Cap ({PATIENT_NAME})")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_episode(cgm, actions, cap, seed):
    t = np.arange(len(cgm)) * 3 / 60  # hours

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(t, cgm, color="tab:blue", label="CGM (mg/dL)")
    ax1.axhline(70, color='gray', linestyle='--', linewidth=1)
    ax1.axhline(180, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("CGM (mg/dL)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(t, actions, color="tab:red", alpha=0.6, label="Insulin rate")
    ax2.set_ylabel("Insulin action", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    plt.title(f"Episode Trace | cap={cap:.2f}, seed={seed}")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=" * 70)
    print("Single PPO Evaluation + Visualization")
    print(f"Patient: {PATIENT_NAME}")
    print("=" * 70)

    results = []

    cap_iter = tqdm(CAP_LIST, desc="Evaluating caps")

    for cap in cap_iter:
        model_path = f"{MODEL_DIR}/agent_cap_{cap:.2f}.zip"
        norm_path = f"{MODEL_DIR}/vec_norm_cap_{cap:.2f}.pkl"

        if not (os.path.exists(model_path) and os.path.exists(norm_path)):
            tqdm.write(f"[SKIP] cap={cap:.2f} missing files")
            continue

        mean_tir, std_tir = eval_cap(cap)
        results.append((cap, mean_tir, std_tir))

        cap_iter.set_postfix(
            TIR=f"{mean_tir:.3f}",
            STD=f"{std_tir:.3f}"
        )

    if results:
        best = max(results, key=lambda x: x[1])

        print("\n" + "=" * 70)
        print("BEST SINGLE PPO")
        print(
            f"cap={best[0]:.2f} | "
            f"TIR={best[1]:.3f} ± {best[2]:.3f}"
        )
        print("=" * 70)

        plot_cap_vs_tir(results)
        tir, actions, cgm = run_single_episode(
            best[0],
            seed=SEED_BASE,
            show_steps=False
        )
        plot_episode(cgm, actions, best[0], SEED_BASE)
