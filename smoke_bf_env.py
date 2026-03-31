from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

# Ensure we stay CPU-only even if a GPU exists somewhere.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduce TF chatter if imported indirectly

import gymnasium as gym

# Import your env
from envs.bellman_ford_env import BellmanFordEnv


@dataclass(frozen=True)
class SmokeConfig:
    n_nodes: int
    n_envs: int
    n_resets: int
    n_steps_per_reset: int
    seed: int
    no_clrs: bool
    run_sb3_check_env: bool
    test_subproc: bool


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _space_shape_dtype(space: gym.Space) -> Tuple[Tuple[int, ...], np.dtype]:
    shape = tuple(getattr(space, "shape", ()) or ())
    dtype = getattr(space, "dtype", None)
    if dtype is None:
        dtype = np.int64
    return shape, np.dtype(dtype)


def _check_obs_matches_space(obs: Dict[str, Any], space: gym.spaces.Dict, where: str) -> None:
    """
    Enforce:
    - keys match exactly
    - each key's value has exact shape + dtype required by the space
    This strictness is intentional: SB3 VecEnv buffers assume exact shapes/dtypes.
    """
    _assert(isinstance(space, gym.spaces.Dict), f"{where}: observation_space must be Dict")
    _assert(isinstance(obs, dict), f"{where}: obs must be dict, got {type(obs)}")

    obs_keys = set(obs.keys())
    space_keys = set(space.spaces.keys())
    _assert(obs_keys == space_keys,
            f"{where}: obs keys {sorted(obs_keys)} != space keys {sorted(space_keys)}")

    for k, subspace in space.spaces.items():
        v = obs[k]

        # Discrete subspaces: accept scalar-ish ints
        if isinstance(subspace, gym.spaces.Discrete):
            _assert(
                np.isscalar(v) or (isinstance(v, np.ndarray) and v.shape == ()),
                f"{where}: key '{k}' expected scalar for Discrete, got {type(v)} shape={getattr(v,'shape',None)}"
            )
            iv = int(v)
            _assert(0 <= iv < subspace.n,
                    f"{where}: key '{k}' value {iv} out of range [0,{subspace.n})")
            continue

        # Boxes etc: enforce shape + dtype exact
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)

        exp_shape, exp_dtype = _space_shape_dtype(subspace)
        _assert(v.shape == exp_shape,
                f"{where}: key '{k}' shape {v.shape} != expected {exp_shape}")

        got_dtype = np.dtype(v.dtype)
        _assert(got_dtype == exp_dtype,
                f"{where}: key '{k}' dtype {got_dtype} != expected {exp_dtype}")

        # Light sanity checks
        if k == "visited":
            _assert(np.all((0.0 <= v) & (v <= 1.0)),
                    f"{where}: key '{k}' has values outside [0,1]")


def _make_env(cfg: SmokeConfig) -> BellmanFordEnv:
    # Match your training defaults; override CLRS if requested.
    return BellmanFordEnv(
        n_nodes=cfg.n_nodes,
        reward_mode="sparse",
        seed=None,
        use_clrs=not cfg.no_clrs,
        clrs_root=None,
    )


def _test_single_env_api(cfg: SmokeConfig) -> None:
    env = _make_env(cfg)

    for r in range(cfg.n_resets):
        obs, info = env.reset(seed=cfg.seed + r)
        _assert(isinstance(info, dict), "reset() must return (obs, info) with info as dict")
        _check_obs_matches_space(obs, env.observation_space, where=f"single.reset(r={r})")

        for t in range(cfg.n_steps_per_reset):
            # Critical: ensure Discrete int actions work (avoids your earlier action-shape issue)
            a = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(a)

            _assert(isinstance(info, dict), "step() must return info as dict")
            _assert(isinstance(terminated, (bool, np.bool_)), "terminated must be bool")
            _assert(isinstance(truncated, (bool, np.bool_)), "truncated must be bool")
            _assert(np.isfinite(np.array(reward, dtype=np.float32)).item(), "reward must be finite")

            _check_obs_matches_space(obs, env.observation_space, where=f"single.step(r={r},t={t})")

            if terminated or truncated:
                break

    env.close()


def _test_sb3_dummy_vecenv(cfg: SmokeConfig) -> None:
    """
    This is the exact path that previously crashed you:
      DummyVecEnv.reset() -> _save_obs() broadcast into buf_obs
    """
    try:
        from stable_baselines3.common.env_util import make_vec_env
    except Exception as e:
        print(f"[smoke] SKIP SB3 DummyVecEnv (stable-baselines3 not importable): {e!r}")
        return

    env_fn = lambda: _make_env(cfg)
    venv = make_vec_env(env_fn, n_envs=cfg.n_envs, seed=cfg.seed)  # defaults to DummyVecEnv

    obs = venv.reset()  # <-- will raise if any obs key is variable-size
    _assert(isinstance(obs, dict), "VecEnv reset obs should be dict")
    for k, v in obs.items():
        _assert(isinstance(v, np.ndarray), f"VecEnv obs[{k}] should be np.ndarray")
        _assert(v.shape[0] == cfg.n_envs, f"VecEnv obs[{k}] first dim should be n_envs")

    for _ in range(8):
        actions = np.array([venv.action_space.sample() for _ in range(cfg.n_envs)], dtype=np.int64)
        obs, rewards, dones, infos = venv.step(actions)
        _assert(np.isfinite(rewards).all(), "VecEnv rewards must be finite")

    venv.close()


def _test_sb3_check_env(cfg: SmokeConfig) -> None:
    try:
        from stable_baselines3.common.env_checker import check_env
    except Exception as e:
        print(f"[smoke] SKIP SB3 check_env (stable-baselines3 not importable): {e!r}")
        return

    env = _make_env(cfg)
    check_env(env, warn=True, skip_render_check=True)
    env.close()


def _test_sb3_ppo_dry_run(cfg):
    print("[smoke] 5) SB3 PPO dry-run (policy forward pass)", flush=True)

    try:
        import torch  # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as e:
        print(f"[smoke] SKIP 5: SB3/torch missing: {e!r}", flush=True)
        return

    # Force CPU — we want this to run without GPU scheduling/visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def _make_env():
        return BellmanFordEnv(
            n_nodes=cfg.n_nodes,
            reward_mode="sparse",            # match your typical training
            seed=None,
            use_clrs=not cfg.no_clrs,
            clrs_root=getattr(cfg, "clrs_root", None),
        )

    env = DummyVecEnv([_make_env, _make_env])

    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        device="cpu",
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=64)

    env.close()
    print("[smoke] OK 5", flush=True)


def _test_sb3_subproc_vecenv(cfg: SmokeConfig) -> None:
    """
    Optional multiprocessing path (useful later when you switch training to SubprocVecEnv).
    """
    try:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as e:
        print(f"[smoke] SKIP SB3 SubprocVecEnv (SB3 not importable): {e!r}")
        return

    env_fn = lambda: _make_env(cfg)
    venv = make_vec_env(
        env_fn,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"},
    )

    obs = venv.reset()
    _assert(isinstance(obs, dict), "SubprocVecEnv reset obs should be dict")
    for k, v in obs.items():
        _assert(isinstance(v, np.ndarray), f"SubprocVecEnv obs[{k}] should be np.ndarray")
        _assert(v.shape[0] == cfg.n_envs, f"SubprocVecEnv obs[{k}] first dim should be n_envs")

    for _ in range(8):
        actions = np.array([venv.action_space.sample() for _ in range(cfg.n_envs)], dtype=np.int64)
        obs, rewards, dones, infos = venv.step(actions)
        _assert(np.isfinite(rewards).all(), "SubprocVecEnv rewards must be finite")

    venv.close()


def parse_args() -> SmokeConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=16)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-resets", type=int, default=50)
    p.add_argument("--n-steps-per-reset", type=int, default=50)
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--no-clrs", action="store_true", help="Force non-CLRS random graphs path")
    p.add_argument("--sb3-check-env", action="store_true", help="Run stable_baselines3 check_env (strict)")
    p.add_argument("--test-subproc", action="store_true", help="Also smoke-test SubprocVecEnv")
    a = p.parse_args()

    return SmokeConfig(
        n_nodes=a.n_nodes,
        n_envs=a.n_envs,
        n_resets=a.n_resets,
        n_steps_per_reset=a.n_steps_per_reset,
        seed=a.seed,
        no_clrs=a.no_clrs,
        run_sb3_check_env=a.sb3_check_env,
        test_subproc=a.test_subproc,
    )


def main() -> int:
    cfg = parse_args()
    t0 = time.time()

    print(f"[smoke] cfg={cfg}")
    print(f"[smoke] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')!r}")

    print("[smoke] 1) single-env API + fixed obs shape/dtype")
    _test_single_env_api(cfg)
    print("[smoke] OK 1")

    print("[smoke] 2) SB3 DummyVecEnv reset/step (targets broadcast bug)")
    _test_sb3_dummy_vecenv(cfg)
    print("[smoke] OK 2 (or skipped if SB3 missing)")

    if cfg.run_sb3_check_env:
        print("[smoke] 3) SB3 check_env")
        _test_sb3_check_env(cfg)
        print("[smoke] OK 3 (or skipped if SB3 missing)")

    if cfg.test_subproc:
        print("[smoke] 4) SB3 SubprocVecEnv reset/step")
        _test_sb3_subproc_vecenv(cfg)
        print("[smoke] OK 4 (or skipped if SB3 missing)")

    _test_sb3_ppo_dry_run(cfg)


    dt = time.time() - t0
    print(f"[smoke] ALL OK in {dt:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        raise
