import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_env import DiabetesEnv


PATIENT_NAME = "adult#001"
MODEL_DIR = f"models/{PATIENT_NAME}"

CAP_LIST = [0.06, 0.07]
THRESHOLDS = list(range(150, 201, 10))  # 150,160,...,200

N_RUNS = 5
DAYS = 5
MAX_STEPS = 2400
SAFETY_THRESHOLD = 90.0

SEED_BASE = 2025


class DualPPOController:
    def __init__(self, cap_h: float, cap_l: float, seed: int, days: int = 5):
        self.cap_h = float(cap_h)
        self.cap_l = float(cap_l)
        self.seed = int(seed)
        self.days = int(days)

        def env_h_fn():
            return DiabetesEnv(
                patient_name=PATIENT_NAME,
                insulin_cap=self.cap_h,
                seed=self.seed,
                days=self.days,
                verbose_log=False
            )

        def env_l_fn():
            return DiabetesEnv(
                patient_name=PATIENT_NAME,
                insulin_cap=self.cap_l,
                seed=self.seed,
                days=self.days,
                verbose_log=False
            )

        self.env_h = DummyVecEnv([env_h_fn])
        self.env_l = DummyVecEnv([env_l_fn])

        vec_h_path = os.path.join(MODEL_DIR, f"vec_norm_cap_{self.cap_h:.2f}.pkl")
        vec_l_path = os.path.join(MODEL_DIR, f"vec_norm_cap_{self.cap_l:.2f}.pkl")

        if not os.path.exists(vec_h_path):
            raise FileNotFoundError(f"Missing VecNormalize file: {vec_h_path}")
        if not os.path.exists(vec_l_path):
            raise FileNotFoundError(f"Missing VecNormalize file: {vec_l_path}")

        self.env_h = VecNormalize.load(vec_h_path, self.env_h)
        self.env_l = VecNormalize.load(vec_l_path, self.env_l)

        self.env_h.training = False
        self.env_l.training = False
        self.env_h.norm_reward = False
        self.env_l.norm_reward = False

        model_h_path = os.path.join(MODEL_DIR, f"agent_cap_{self.cap_h:.2f}.zip")
        model_l_path = os.path.join(MODEL_DIR, f"agent_cap_{self.cap_l:.2f}.zip")

        if not os.path.exists(model_h_path):
            raise FileNotFoundError(f"Missing PPO model: {model_h_path}")
        if not os.path.exists(model_l_path):
            raise FileNotFoundError(f"Missing PPO model: {model_l_path}")

        self.high_policy = PPO.load(model_h_path, env=self.env_h)
        self.low_policy = PPO.load(model_l_path, env=self.env_l)

    def act(self, obs: np.ndarray, threshold: float) -> np.ndarray:
        cgm = float(obs[0])
        if cgm < SAFETY_THRESHOLD:
            return np.array([0.0], dtype=np.float32)
        obs_batch = obs.reshape(1, -1)

        if cgm >= float(threshold):
            obs_norm = self.env_h.normalize_obs(obs_batch)
            action, _ = self.high_policy.predict(obs_norm, deterministic=True)
        else:
            obs_norm = self.env_l.normalize_obs(obs_batch)
            action, _ = self.low_policy.predict(obs_norm, deterministic=True)

        return action

def run_dual_episode(cap_h: float, cap_l: float, threshold: float, seed: int) -> float:
    env = DiabetesEnv(
        patient_name=PATIENT_NAME,
        insulin_cap=float(cap_h),
        seed=int(seed),
        days=DAYS,
        verbose_log=False
    )

    controller = DualPPOController(
        cap_h=float(cap_h),
        cap_l=float(cap_l),
        seed=int(seed),
        days=DAYS
    )

    obs, _ = env.reset(seed=int(seed))
    cgm_trace = []

    steps = min(MAX_STEPS, getattr(env, "episode_steps", MAX_STEPS))

    for _ in range(steps):
        cgm_trace.append(float(obs[0]))
        action = controller.act(obs, threshold=threshold)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break

    env.close()

    cgm_arr = np.asarray(cgm_trace, dtype=np.float32)
    if cgm_arr.size == 0:
        return np.nan

    tir = float(np.mean((cgm_arr >= 70.0) & (cgm_arr <= 180.0)))
    return tir


def dual_ppo_grid_search():
    combos = [
        (h, l, th)
        for h in CAP_LIST
        for l in CAP_LIST if l < h
        for th in THRESHOLDS
    ]

    best_cfg = None
    best_mean = -1.0

    print(f"\n[INFO] Total combinations: {len(combos)}")
    print(f"[INFO] Total episodes: {len(combos) * N_RUNS}\n")
    outer_bar = tqdm(combos, desc="Dual PPO Grid Search", ncols=120)

    for idx, (cap_h, cap_l, th) in enumerate(outer_bar, start=1):
        tirs = []

        inner_bar = tqdm(
            range(N_RUNS),
            desc=f"H={cap_h:.2f} L={cap_l:.2f} TH={th}",
            leave=False,
            ncols=120
        )

        for k in inner_bar:
            seed = SEED_BASE + k
            try:
                tir = run_dual_episode(cap_h, cap_l, th, seed=seed)
            except Exception as e:
                tqdm.write(
                    f"[WARN] Failed: H={cap_h:.2f} L={cap_l:.2f} TH={th} "
                    f"seed={seed} | {type(e).__name__}: {e}"
                )
                tir = np.nan
            tirs.append(tir)

        tirs_clean = np.asarray([x for x in tirs if np.isfinite(x)], dtype=np.float32)

        if tirs_clean.size > 0:
            mean_tir = float(tirs_clean.mean())
            std_tir  = float(tirs_clean.std(ddof=0))
        else:
            mean_tir = float("nan")
            std_tir  = float("nan")

        if np.isfinite(mean_tir) and mean_tir > best_mean:
            best_mean = mean_tir
            best_cfg = (cap_h, cap_l, th, mean_tir, std_tir)

        outer_bar.set_postfix(
            last=f"{mean_tir:.3f}±{std_tir:.3f}" if np.isfinite(mean_tir) else "nan",
            best=f"{best_mean:.3f}" if best_cfg is not None else "na"
        )

        tqdm.write(
            f"[{idx:3d}/{len(combos)}] "
            f"H={cap_h:.2f} L={cap_l:.2f} TH={th} "
            f"→ TIR={mean_tir:.3f} ± {std_tir:.3f}"
            if np.isfinite(mean_tir) else
            f"[{idx:3d}/{len(combos)}] "
            f"H={cap_h:.2f} L={cap_l:.2f} TH={th} → TIR=nan"
        )

    return best_cfg

if __name__ == "__main__":
    print("=" * 70)
    print("Dual PPO Network Search (mean ± std)")
    print(f"Patient: {PATIENT_NAME}")
    print(f"Days: {DAYS} | Runs/Combo: {N_RUNS} | Safety CGM<{SAFETY_THRESHOLD}")
    print("=" * 70)

    best_cfg = dual_ppo_grid_search()

    print("\n" + "=" * 70)
    print("BEST DUAL PPO CONFIG")
    if best_cfg is None:
        print("No valid configuration found.")
    else:
        print(f"Cap_H     = {best_cfg[0]:.2f}")
        print(f"Cap_L     = {best_cfg[1]:.2f}")
        print(f"Threshold = {best_cfg[2]} mg/dL")
        print(f"TIR       = {best_cfg[3]:.3f} ± {best_cfg[4]:.3f}")
    print("=" * 70)

