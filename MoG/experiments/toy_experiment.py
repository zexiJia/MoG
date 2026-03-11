"""Toy Experiment: 螺旋流形上的 MoG vs CFG vs APG 轨迹对比

复现论文中的 Figure 2，展示:
  - CFG: 欧氏直线引导，偏离流形
  - APG: 切向投影引导，安全但缓慢
  - MOG: 黎曼自然梯度引导，快速且稳定
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from matplotlib.lines import Line2D

try:
    import seaborn as sns
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("white")
except ImportError:
    pass

# ==========================================
# 1. 实验配置
# ==========================================
COLORS = {
    "Manifold": "#606060",
    "CFG": "#D62728",
    "APG": "#FF7F0E",
    "MOG": "#1F77B4",
    "Target": "#2CA02C",
    "Start": "black",
}

PARAMS = {
    "a": 0.1, "b": 0.13,
    "s_range": (0, 7.5 * np.pi),
    "s_start": 1.5 * np.pi,
    "s_target": 5.2 * np.pi,
    "dt": 0.02,
    "steps": 300,
    "guidance_w": 18.0,
    "lambda_par": 0.1,
    "lambda_perp": 100.0,
}


# ==========================================
# 2. 螺旋流形
# ==========================================
class SpiralManifold:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def pos(self, s):
        r = self.a + self.b * s
        return np.array([r * np.cos(s), r * np.sin(s)])

    def tangent(self, s):
        r = self.a + self.b * s
        dx = self.b * np.cos(s) - r * np.sin(s)
        dy = self.b * np.sin(s) + r * np.cos(s)
        n = np.sqrt(dx ** 2 + dy ** 2)
        return np.array([dx / n, dy / n])

    def normal(self, s):
        t = self.tangent(s)
        return np.array([-t[1], t[0]])

    def get_closest_s(self, x_query):
        s_grid = np.linspace(*PARAMS["s_range"], 200)
        dists = np.linalg.norm(x_query - np.array([self.pos(s) for s in s_grid]), axis=1)
        s_init = s_grid[np.argmin(dists)]
        res = minimize_scalar(
            lambda s: np.linalg.norm(x_query - self.pos(s)),
            bounds=(max(0, s_init - 1), min(PARAMS["s_range"][1], s_init + 1)),
            method="bounded",
        )
        return res.x


manifold = SpiralManifold(PARAMS["a"], PARAMS["b"])
x_target = manifold.pos(PARAMS["s_target"])


# ==========================================
# 3. 模拟
# ==========================================
def run_simulation(method):
    x_curr = manifold.pos(PARAMS["s_start"])
    trajectory = [x_curr]
    dists = []

    dt = PARAMS["dt"]
    w = PARAMS["guidance_w"]

    def s_uncond_force(x, x_proj):
        return -(x - x_proj) * 5.0

    for _ in range(PARAMS["steps"]):
        s_proj = manifold.get_closest_s(x_curr)
        x_proj = manifold.pos(s_proj)
        t_vec = manifold.tangent(s_proj)
        n_vec = manifold.normal(s_proj)
        delta_s = x_target - x_curr

        if method == "CFG":
            guidance = delta_s
        elif method == "APG":
            d_tan = np.dot(delta_s, t_vec)
            guidance = d_tan * t_vec
        elif method == "MOG":
            d_tan = np.dot(delta_s, t_vec)
            d_norm = np.dot(delta_s, n_vec)
            scale_par = 1.0 / PARAMS["lambda_par"]
            scale_perp = 1.0 / PARAMS["lambda_perp"]
            guidance = (d_tan * scale_par * t_vec) + (d_norm * scale_perp * n_vec)

        v = s_uncond_force(x_curr, x_proj) + w * guidance * 0.05
        x_next = x_curr + v * dt

        trajectory.append(x_next)
        dists.append(np.linalg.norm(x_curr - x_proj))
        x_curr = x_next

        if np.linalg.norm(x_curr) > 30:
            break

    return np.array(trajectory), np.array(dists)


# ==========================================
# 4. 运行并绘图
# ==========================================
def main():
    print("Running simulations...")
    traj_cfg, dist_cfg = run_simulation("CFG")
    traj_apg, dist_apg = run_simulation("APG")
    traj_mog, dist_mog = run_simulation("MOG")

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.6, 1], hspace=0.35)

    ax_main = plt.subplot(gs[:, 0])
    ax_dist = plt.subplot(gs[0, 1])
    ax_eff = plt.subplot(gs[1, 1])

    # --- 主图 ---
    x_grid = np.linspace(-4, 5, 150)
    y_grid = np.linspace(-5, 4, 150)
    X, Y = np.meshgrid(x_grid, y_grid)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    s_samples = np.linspace(*PARAMS["s_range"], 800)
    manifold_points = np.array([manifold.pos(s) for s in s_samples])
    d_field = cdist(pts, manifold_points[::4]).min(axis=1).reshape(X.shape)
    Z = np.exp(-d_field ** 2 / (2 * 0.5 ** 2))

    ax_main.contourf(X, Y, Z, levels=30, cmap="Greys", alpha=0.2)
    ax_main.plot(manifold_points[:, 0], manifold_points[:, 1], c=COLORS["Manifold"], lw=1.2, ls="--", alpha=0.5)

    ax_main.plot(traj_cfg[:, 0], traj_cfg[:, 1], c=COLORS["CFG"], lw=3.0)
    ax_main.plot(traj_apg[:, 0], traj_apg[:, 1], c=COLORS["APG"], lw=3.0)
    ax_main.plot(traj_mog[:, 0], traj_mog[:, 1], c=COLORS["MOG"], lw=3.5)

    start_pt = manifold.pos(PARAMS["s_start"])
    ax_main.scatter(*start_pt, c=COLORS["Start"], s=120, marker="o", edgecolors="white", linewidth=1.5, zorder=20)
    ax_main.scatter(*x_target, c=COLORS["Target"], s=240, marker="*", edgecolors="white", linewidth=1.0, zorder=20)

    ax_main.set_title("Trajectory Comparison: Efficiency vs Stability", fontweight="bold", fontsize=12)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_aspect("equal")
    for s in ax_main.spines.values():
        s.set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=COLORS["CFG"], lw=3, label="CFG (Euclidean)"),
        Line2D([0], [0], color=COLORS["APG"], lw=3, label="APG (Projected)"),
        Line2D([0], [0], color=COLORS["MOG"], lw=3.5, label="MOG (Ours)"),
        Line2D([0], [0], color=COLORS["Manifold"], lw=1.2, ls="--", label="Data Manifold"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["Start"], markersize=9, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COLORS["Target"], markersize=13, label="Condition"),
    ]
    ax_main.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    # --- Off-Manifold Distance ---
    ax_dist.plot(np.arange(len(dist_cfg)), dist_cfg, c=COLORS["CFG"], lw=2.5)
    ax_dist.plot(np.arange(len(dist_apg)), dist_apg, c=COLORS["APG"], lw=2.5)
    ax_dist.plot(np.arange(len(dist_mog)), dist_mog, c=COLORS["MOG"], lw=2.5)
    ax_dist.set_title("Artifacts (Off-Manifold Dist)", fontsize=11, fontweight="bold")
    ax_dist.set_ylabel(r"$d_{\mathcal{M}}(x_t)$", fontsize=10)
    ax_dist.set_ylim(-0.05, 1.8)
    ax_dist.set_xlim(0, PARAMS["steps"])
    ax_dist.grid(True, alpha=0.3)

    # --- Energy Convergence ---
    def get_energy(traj):
        return 0.5 * np.linalg.norm(traj - x_target, axis=1) ** 2

    e_cfg = get_energy(traj_cfg)
    e_apg = get_energy(traj_apg)
    e_mog = get_energy(traj_mog)

    ax_eff.plot(np.arange(len(e_cfg)), e_cfg, c=COLORS["CFG"], lw=2.5, label="CFG")
    ax_eff.plot(np.arange(len(e_apg)), e_apg, c=COLORS["APG"], lw=2.5, label="APG")
    ax_eff.plot(np.arange(len(e_mog)), e_mog, c=COLORS["MOG"], lw=2.5, label="MOG")
    ax_eff.set_title("Goal Alignment (Energy)", fontsize=11, fontweight="bold")
    ax_eff.set_ylabel("Cond. Energy", fontsize=10)
    ax_eff.set_xlabel("Simulation Steps", fontsize=10)
    ax_eff.set_xlim(0, PARAMS["steps"])
    ax_eff.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("toy_experiment.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("toy_experiment.png", bbox_inches="tight", dpi=150)
    print("Saved: toy_experiment.pdf / toy_experiment.png")


if __name__ == "__main__":
    main()
