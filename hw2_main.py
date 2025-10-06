# HW2 – Tabular MC / SARSA / Q-learning on FrozenLake (Gymnasium) - Nitya Jagadam
# Map: ["SFFF","FHFH","FFFH","HFFG"], gamma=0.95, both slippery and non-slippery
import os
import argparse
from typing import Callable, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

GAMMA = 0.95
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}
N_ROWS, N_COLS = 4, 4

def make_env(is_slippery: bool) -> gym.Env:
    return gym.make("FrozenLake-v1", desc=MAP, is_slippery=is_slippery, render_mode=None)

def print_mdp():
    print("FrozenLake 4x4 MDP")
    print(f"|S|=16, |A|=4 (0=LEFT,1=DOWN,2=RIGHT,3=UP), γ={GAMMA}")
    for row in MAP: print(" ", row)
    print("Start=S, terminals=H/G, reward=+1 at G, else 0\n")

def eps_greedy(Q: np.ndarray, s: int, eps: float) -> int:
    if np.random.rand() < eps: return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[s]))

def eval_greedy(env_fn: Callable[[], gym.Env], Q: np.ndarray, episodes: int) -> float:
    total = 0.0
    for _ in range(episodes):
        env = env_fn()
        s, _ = env.reset()
        done = False
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += r
        env.close()
    return total / episodes

def plot_learning(ax, steps: List[int], scores: List[float], label: str):
    ax.plot(steps, scores, label=label, linewidth=1.6)
    ax.set_xlabel("Env steps"); ax.set_ylabel("Eval return"); ax.grid(True, linestyle=":")
    ax.legend(frameon=False)

def plot_policy_value(ax_pol, ax_val, Q: np.ndarray, title: str):
    policy = np.argmax(Q, axis=1)
    values = np.max(Q, axis=1).reshape(N_ROWS, N_COLS)

    # Policy arrows
    ax_pol.set_title(title + " — Policy", fontsize=10)
    ax_pol.set_xticks(range(N_COLS)); ax_pol.set_yticks(range(N_ROWS))
    ax_pol.set_xticklabels([]); ax_pol.set_yticklabels([])
    ax_pol.set_xlim(-0.5, N_COLS-0.5); ax_pol.set_ylim(N_ROWS-0.5, -0.5)
    ax_pol.grid(True, linestyle=":", linewidth=0.5)
    for s in range(N_ROWS*N_COLS):
        r, c = divmod(s, N_COLS)
        ax_pol.text(c, r, ARROWS[int(policy[s])], ha="center", va="center", fontsize=14)

    # Values heatmap
    im = ax_val.imshow(values, origin="upper", interpolation="nearest")
    ax_val.set_title(title + " — Values", fontsize=10)
    ax_val.set_xticks(range(N_COLS)); ax_val.set_yticks(range(N_ROWS))
    ax_val.set_xticklabels([]); ax_val.set_yticklabels([])
    plt.colorbar(im, ax=ax_val, fraction=0.046, pad=0.04)

def save_fig(fig: plt.Figure, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved:", path)

#  Training loops (env.step-based)
def train_mc(env_fn: Callable[[], gym.Env], nS: int, nA: int,
             episodes=30000, eps_start=1.0, eps_end=0.05,
             eval_every=2000, eval_episodes=256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = np.zeros((nS, nA))
    ret_sum = np.zeros((nS, nA)); ret_cnt = np.zeros((nS, nA))
    steps, scores, total_steps, next_eval = [0], [eval_greedy(env_fn, Q, eval_episodes)], 0, eval_every

    for ep in range(1, episodes+1):
        eps = eps_end + (eps_start - eps_end) * (episodes - ep) / episodes
        env = env_fn()
        s, _ = env.reset()
        ep_traj = []
        done = False
        while not done:
            a = eps_greedy(Q, s, eps)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_traj.append((s, a, r))
            s = s2
        env.close()

        G = 0.0; seen = set()
        for t in reversed(range(len(ep_traj))):
            s_t, a_t, r_t = ep_traj[t]
            G = GAMMA*G + r_t
            if (s_t, a_t) not in seen:
                seen.add((s_t, a_t))
                ret_sum[s_t, a_t] += G
                ret_cnt[s_t, a_t] += 1
                Q[s_t, a_t] = ret_sum[s_t, a_t] / ret_cnt[s_t, a_t]

        total_steps += len(ep_traj)
        if total_steps >= next_eval:
            steps.append(next_eval)
            scores.append(eval_greedy(env_fn, Q, eval_episodes))
            next_eval += eval_every

    return Q, np.array(steps), np.array(scores)

def train_sarsa(env_fn: Callable[[], gym.Env], nS: int, nA: int,
                episodes=30000, alpha=0.1, eps_start=1.0, eps_end=0.05,
                eval_every=2000, eval_episodes=256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = np.zeros((nS, nA))
    steps, scores, total_steps, next_eval = [0], [eval_greedy(env_fn, Q, eval_episodes)], 0, eval_every

    for ep in range(1, episodes+1):
        eps = eps_end + (eps_start - eps_end) * (episodes - ep) / episodes
        env = env_fn()
        s, _ = env.reset()
        a = eps_greedy(Q, s, eps)
        done = False
        while not done:
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            if not done:
                a2 = eps_greedy(Q, s2, eps)
                target = r + GAMMA * Q[s2, a2]
            else:
                a2 = a
                target = r
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            total_steps += 1
            if total_steps >= next_eval:
                steps.append(next_eval)
                scores.append(eval_greedy(env_fn, Q, eval_episodes))
                next_eval += eval_every
        env.close()

    return Q, np.array(steps), np.array(scores)

def train_qlearn(env_fn: Callable[[], gym.Env], nS: int, nA: int,
                 episodes=30000, alpha=0.1, eps_start=1.0, eps_end=0.05,
                 eval_every=2000, eval_episodes=256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = np.zeros((nS, nA))
    steps, scores, total_steps, next_eval = [0], [eval_greedy(env_fn, Q, eval_episodes)], 0, eval_every

    for ep in range(1, episodes+1):
        eps = eps_end + (eps_start - eps_end) * (episodes - ep) / episodes
        env = env_fn()
        s, _ = env.reset()
        done = False
        while not done:
            a = eps_greedy(Q, s, eps)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            target = r if done else r + GAMMA * np.max(Q[s2])
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            total_steps += 1
            if total_steps >= next_eval:
                steps.append(next_eval)
                scores.append(eval_greedy(env_fn, Q, eval_episodes))
                next_eval += eval_every
        env.close()

    return Q, np.array(steps), np.array(scores)

#  Run one suite (one slipperiness)
def run_suite(is_slippery: bool, episodes: int, eval_every: int, eval_eps: int,
              alpha: float, eps_start: float, eps_end: float,
              outdir: str, prefix: str):
    env0 = make_env(is_slippery)
    nS, nA = env0.observation_space.n, env0.action_space.n
    env0.close()
    env_fn = lambda: make_env(is_slippery)
    tag = "slip" if is_slippery else "noslip"

    Qmc, xmc, ymc = train_mc(env_fn, nS, nA, episodes, eps_start, eps_end, eval_every, eval_eps)
    Qsa, xsa, ysa = train_sarsa(env_fn, nS, nA, episodes, alpha, eps_start, eps_end, eval_every, eval_eps)
    Qql, xql, yql = train_qlearn(env_fn, nS, nA, episodes, alpha, eps_start, eps_end, eval_every, eval_eps)

    # Learning curve
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=140)
    plot_learning(ax, xmc, ymc, f"MC ({tag})")
    plot_learning(ax, xsa, ysa, f"SARSA ({tag})")
    plot_learning(ax, xql, yql, f"Q-learning ({tag})")
    ax.set_title(f"FrozenLake 4x4 — Eval Return (slippery={is_slippery})")
    fig.tight_layout(); save_fig(fig, outdir, f"{prefix}_learning_{tag}"); plt.close(fig)

    # Policy & values per algorithm
    for (name, Q, short) in [("MC", Qmc, "mc"), ("SARSA", Qsa, "sarsa"), ("Q-learning", Qql, "qlearn")]:
        fig, (axp, axv) = plt.subplots(1, 2, figsize=(7.0, 3.4), dpi=150)
        plot_policy_value(axp, axv, Q, f"{name} ({tag})")
        fig.tight_layout(); save_fig(fig, outdir, f"{prefix}_policy_value_{short}_{tag}"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_eps", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--outdir", type=str, default="figs")
    ap.add_argument("--prefix", type=str, default="hw2")
    args = ap.parse_args()

    print_mdp()
    print("Training (slippery=False)...")
    run_suite(False, args.episodes, args.eval_every, args.eval_eps,
              args.alpha, args.eps_start, args.eps_end, args.outdir, args.prefix)

    print("Training (slippery=True)...")
    run_suite(True, args.episodes, args.eval_every, args.eval_eps,
              args.alpha, args.eps_start, args.eps_end, args.outdir, args.prefix)

    print("\nDone.")

if __name__ == "__main__":
    main()
