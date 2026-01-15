import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_environment import DiabetesEnv

PATIENT_NAME = "adult#008"
MODEL_DIR = f"Visualization_models/{PATIENT_NAME}"

CAP = 0.08
SEED = 2025

STEP_MINUTES = 3
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES  
PLOT_STEPS = STEPS_PER_DAY               
LOW_DASH = 50
HIGH_DASH = 300

TIR_LOW = 70
TIR_HIGH = 180

FIG_TITLE = f"Simulation: Glucose Regulation using PPO Model ({PATIENT_NAME}, cap={CAP:.2f})"
PRINT_EVERY = 120  


CGM_PINK      = "#C75D7A"   
TIR_PINK      = "#F6D6DE"   
THRESH_BLUE   = "#2F4B7C"   
MEAL_PURPLE   = "#4B1D3F"   
GRID_GRAY     = "#DADADA"   

def _to_scalar_action(action):
    a = np.asarray(action).squeeze()
    if a.ndim == 0:
        return float(a)
    return float(a.flatten()[0])


def make_venv_for_vecnorm(cap, seed):
    def env_fn():
        return DiabetesEnv(
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
    return venv


def _get_meal_array_safe(env):
    if hasattr(env, "meal_array"):
        try:
            arr = np.asarray(env.meal_array, dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                return arr
        except Exception:
            pass
    if hasattr(env, "_env") and hasattr(env._env, "meal_array"):
        try:
            arr = np.asarray(env._env.meal_array, dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                return arr
        except Exception:
            pass
    return None


def _get_start_time_safe(env):
    if hasattr(env, "_env") and hasattr(env._env, "scenario") and hasattr(env._env.scenario, "start_time"):
        return env._env.scenario.start_time

    if hasattr(env, "scenario") and hasattr(env.scenario, "start_time"):
        return env.scenario.start_time

    import datetime as _dt
    now = _dt.datetime.now()
    return _dt.datetime(now.year, now.month, now.day, 0, 0, 0)


def rollout_one_day_with_meals_ppo(cap, seed, show_progress=True):
    env = DiabetesEnv(
        patient_name=PATIENT_NAME,
        insulin_cap=cap,
        seed=seed
    )
    venv = make_venv_for_vecnorm(cap, seed)

    model = PPO.load(
        f"{MODEL_DIR}/agent_cap_{cap:.2f}.zip",
        env=venv
    )

    obs, _ = env.reset()

    start_time = _get_start_time_safe(env)
    meal_events = []
    meal_array = _get_meal_array_safe(env)
    if meal_array is not None:
        for step_idx in range(min(PLOT_STEPS, len(meal_array))):
            grams = float(meal_array[step_idx])
            if grams > 0:
                meal_events.append((step_idx, grams))

    cgm_trace = []
    insulin_rate_trace = []

    steps = range(PLOT_STEPS)
    if show_progress:
        steps = tqdm(steps, desc=f"PPO rollout 1 day | cap={cap:.2f} | seed={seed}")

    for step in steps:
        obs_norm = venv.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_norm, deterministic=True)
        insulin_rate = _to_scalar_action(action)

        obs, _, done, truncated, _ = env.step(insulin_rate)

        cgm_trace.append(float(obs[0]))
        insulin_rate_trace.append(float(insulin_rate))

        if PRINT_EVERY and (step % PRINT_EVERY == 0):
            print(f"step={step:3d} | CGM={cgm_trace[-1]:6.1f} | insulin_rate={insulin_rate_trace[-1]:.4f}")

        if done or truncated:
            break

    env.close()

    cgm = np.asarray(cgm_trace, dtype=np.float32)
    insulin_rate = np.asarray(insulin_rate_trace, dtype=np.float32)

    times = [start_time + timedelta(minutes=STEP_MINUTES * i) for i in range(len(cgm))]
    return times, cgm, insulin_rate, meal_events, start_time

def plot_like_sac_subplots_opposite_palette(times, cgm, insulin_rate, meal_events, start_time):
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.05)

    ax_cgm = fig.add_subplot(gs[0, 0])
    ax_ins = fig.add_subplot(gs[1, 0], sharex=ax_cgm)

    ax_cgm.set_title(FIG_TITLE)
    ax_cgm.plot(
        times, cgm,
        linewidth=2.2,
        color=CGM_PINK,
        label="CGM (Sensor: Dexcom)"
    )
    ax_cgm.set_ylabel("CGM [mg/dL]")

    ax_cgm.axhspan(TIR_LOW, TIR_HIGH, color=TIR_PINK, alpha=0.55)
    ax_cgm.axhline(LOW_DASH,  color=THRESH_BLUE, linestyle="--", linewidth=1.6)
    ax_cgm.axhline(HIGH_DASH, color=THRESH_BLUE, linestyle="--", linewidth=1.6)

    y_text = max(float(np.nanmax(cgm)) + 10.0, TIR_HIGH + 30.0)

    for step_idx, grams in meal_events:
        if 0 <= step_idx < len(times):
            t_meal = times[step_idx]
            ax_cgm.axvline(t_meal, color=MEAL_PURPLE, linewidth=2.2, alpha=0.9)
            ax_cgm.text(
                t_meal, y_text,
                f"Carbohydrates: {grams:.1f}g",
                color=MEAL_PURPLE,
                fontsize=11,
                va="bottom",
                ha="left"
            )

    ax_cgm.set_ylim(0, 600)
    ax_cgm.grid(True, which="both", axis="both", color=GRID_GRAY, alpha=0.4)
    ax_cgm.legend(loc="upper right")

    width_days = (STEP_MINUTES / 60.0) / 24.0
    ax_ins.bar(
        times,
        insulin_rate,
        width=width_days,
        align="edge",
        color=INSULIN_BLUE,
        edgecolor=INSULIN_BLUE,
        alpha=0.80,
        label="Insulin (Pump: Insulet)"
    )
    ax_ins.set_ylabel("Insulin [U/min]")
    ax_ins.set_xlabel("Time (hrs)")
    ax_ins.grid(True, which="both", axis="y", color=GRID_GRAY, alpha=0.4)
    ax_ins.set_yscale("log")
    if np.any(insulin_rate > 0):
        ymin = max(float(np.min(insulin_rate[insulin_rate > 0])), 1e-6)
    else:
        ymin = 1e-6
    ax_ins.set_ylim(ymin, max(float(np.max(insulin_rate)) * 1.2, ymin * 50))

    ax_ins.legend(loc="upper right")
    ax_ins.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
    ax_ins.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_ins.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    ax_ins.annotate(
        start_time.strftime("%b %d"),
        xy=(0, 0),
        xycoords=("axes fraction", "axes fraction"),
        xytext=(-10, -22),
        textcoords="offset points",
        ha="left",
        va="top"
    )
    end_time = times[-1] + timedelta(minutes=STEP_MINUTES)
    ax_ins.annotate(
        end_time.strftime("%b %d"),
        xy=(1, 0),
        xycoords=("axes fraction", "axes fraction"),
        xytext=(10, -22),
        textcoords="offset points",
        ha="right",
        va="top"
    )

    plt.setp(ax_cgm.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_path = f"{MODEL_DIR}/agent_cap_{CAP:.2f}.zip"
    norm_path  = f"{MODEL_DIR}/vec_norm_cap_{CAP:.2f}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing vecnorm: {norm_path}")

    times, cgm, insulin_rate, meal_events, start_time = rollout_one_day_with_meals_ppo(
        cap=CAP,
        seed=SEED,
        show_progress=True
    )

    tir = float(np.mean((cgm >= 70) & (cgm <= 180)))
    print(f"\nPPO 1-day rollout | cap={CAP:.2f} | seed={SEED} | steps={len(cgm)} | TIR={tir:.3f} | meals={len(meal_events)}")

    plot_like_sac_subplots_opposite_palette(times, cgm, insulin_rate, meal_events, start_time)
