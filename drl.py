import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# =============================================================================
# 1. 系統配置
# =============================================================================
class SimulationConfig:
    def __init__(self):
        self.dt           = 0.05
        self.sim_steps    = 1000
        self.num_agvs     = 20
        self.v_ref        = 10.0
        self.a_max        = 5.0
        self.a_min        = -5.0
        self.Lambda       = 10.0
        self.rho_limit    = 0.1
        self.drl_interval = 5

        # ★ 核心設計：DRL 動作 = 本步允許調度的最大 AGV 數量 (1~5)
        #   Static-VoI 固定只調度 1 個（最保守策略）
        #   DRL 能動態決定「要搶救幾台」，才有實質差異
        self.max_subcarriers     = 5          # 頻道上限
        self.static_subcarriers  = 1          # Static基準
        self.action_space        = list(range(1, self.max_subcarriers + 1))  # [1,2,3,4,5]
        self.voi_threshold       = 0.0        # VoI > 0 才傳


class V_Policy_Net(nn.Module):
    """DQN：輸入系統狀態 → 輸出每個動作的Q值"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, float(r), s2, float(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.stack(s),
                torch.tensor(a,  dtype=torch.long),
                torch.tensor(r,  dtype=torch.float32),
                torch.stack(s2),
                torch.tensor(d,  dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


# =============================================================================
# 2. AGV Agent
# =============================================================================
class AGVAgent:
    def __init__(self, uid, cfg, mats):
        self.uid, self.cfg    = uid, cfg
        self.A, self.B, self.W, self.K = mats["A"], mats["B"], mats["W"], mats["K"]
        self.true_state = np.array([0.0, uid * 3.5, cfg.v_ref, 0.0])
        self.est_state  = self.true_state.copy()
        self.h_queue    = float(uid % 5) * 0.2   # 異質化初始佇列
        self.mse_log    = []
        self.age_since_tx = uid % 8               # 異質化初始age

        self.vel_log   = []
        self.acc_log   = []
        self.tx_events = []

    def compute_voi(self):
        """VoI score（不依賴V參數）"""
        error = self.est_state - self.true_state
        return 0.5 * error.T @ self.W @ error - self.cfg.Lambda

    def update(self, u, success, step, record_physics=False):
        noise_std = 1.5 if 200 < step < 800 else 0.3
        noise = np.random.normal(0, noise_std, 4)
        self.true_state = self.A @ self.true_state + self.B @ u + noise

        if record_physics:
            self.vel_log.append(self.true_state[2:4].copy())
            self.acc_log.append(u.copy())

        self.age_since_tx += 1
        if success:
            self.est_state    = self.true_state.copy()
            self.age_since_tx = 0
            if record_physics:
                self.tx_events.append(step)
            self.h_queue = max(self.h_queue + 1.0 - self.cfg.rho_limit, 0)
        else:
            self.est_state = self.A @ self.est_state + self.B @ u
            self.h_queue   = max(self.h_queue - self.cfg.rho_limit, 0)

        err = np.linalg.norm(self.true_state[:2] - self.est_state[:2]) ** 2
        self.mse_log.append(err)


# =============================================================================
# 3. 實驗引擎
# =============================================================================
class AcademicDRLComparison:
    def __init__(self, num_seeds=10):
        self.cfg       = SimulationConfig()
        self.num_seeds = num_seeds
        self.modes     = ["DRL-VoI", "Static-VoI", "AoI", "Random"]
        self.results   = {m: [] for m in self.modes}
        self.plot_data  = {}

        dt = self.cfg.dt
        self.mats = {
            "A": np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]),
            "B": np.array([[0.5*dt**2,0],[0,0.5*dt**2],[dt,0],[0,dt]]),
            "W": np.eye(4) * 100.0,
            "K": np.array([[1.0, 0, 0.6, 0],[0, 1.0, 0, 0.6]])
        }

    # ──────────────────────────────────────────────
    def get_target_path(self, t):
        ref_x  = self.cfg.v_ref * t * self.cfg.dt
        vx_ref = self.cfg.v_ref + 2.5 * np.cos(0.05 * ref_x)
        vy_ref = 10.0 * 0.05 * np.cos(0.05 * ref_x) * vx_ref
        ref_y  = 10.0 * np.sin(0.05 * ref_x)
        return np.array([ref_x, ref_y, vx_ref, vy_ref])

    # ──────────────────────────────────────────────
    def _extract_state(self, agvs, t):
        """
        8維系統狀態向量（歸一化），讓DRL充分感知系統情況：
        [avg_log_mse, max_log_mse, frac_high_mse,
         avg_age, max_age, frac_stale,
         avg_h, t_phase]
        """
        mse_vals  = np.array([a.mse_log[-1] if a.mse_log else 0.0 for a in agvs])
        age_vals  = np.array([a.age_since_tx for a in agvs], dtype=float)
        h_vals    = np.array([a.h_queue for a in agvs])
        threshold = 5.0

        s = np.array([
            np.log1p(mse_vals.mean())   / 6.0,         # avg log-MSE
            np.log1p(mse_vals.max())    / 6.0,         # max log-MSE
            (mse_vals > threshold).mean(),              # 高誤差比例
            age_vals.mean()             / 30.0,        # avg age
            age_vals.max()              / 60.0,        # max age
            (age_vals > 20).mean(),                    # 長時間未更新比例
            h_vals.mean()               / 10.0,        # avg queue
            np.sin(2 * np.pi * t / self.cfg.sim_steps) # 時間相位
        ], dtype=np.float32)
        return torch.tensor(s)

    # ──────────────────────────────────────────────
    def _schedule_voi(self, agvs, n_slots):
        """VoI排程：選VoI最高的前n_slots個AGV"""
        scores = [(a.uid, a.compute_voi()) for a in agvs]
        scores.sort(key=lambda x: x[1], reverse=True)
        # 只調度VoI > threshold 的，最多n_slots個
        selected = [uid for uid, s in scores[:n_slots] if s > self.cfg.voi_threshold]
        # 若全部VoI<0，至少保留最高分的1個（防止完全不傳）
        if not selected:
            selected = [scores[0][0]]
        return selected

    # ──────────────────────────────────────────────
    def run_experiments(self):
        STATE_DIM  = 8
        ACTION_DIM = len(self.cfg.action_space)   # 5種動作

        net        = V_Policy_Net(STATE_DIM, ACTION_DIM)
        target_net = V_Policy_Net(STATE_DIM, ACTION_DIM)
        target_net.load_state_dict(net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(net.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        criterion = nn.SmoothL1Loss()
        replay    = ReplayBuffer(12000)

        epsilon   = 1.0
        gamma     = 0.97
        BATCH     = 128
        TUF       = 40     # target update freq（每N次gradient step更新）
        grad_step = [0]

        # ── 預訓練 ──
        print(">>> 預訓練中（8 epochs）...")
        for ps in range(8):
            self._run_epoch(
                net, target_net, optimizer, criterion, replay,
                epsilon=max(0.5, 1.0 - ps*0.1), gamma=gamma,
                batch=BATCH, tuf=TUF, grad_step=grad_step,
                is_training=True, mode="DRL-VoI", seed=ps
            )
            scheduler.step()
            epsilon = max(0.2, epsilon * 0.75)
            print(f"    pretrain epoch {ps+1}/8, epsilon={epsilon:.3f}, replay={len(replay)}")

        # ── 主實驗 ──
        for seed in range(1, self.num_seeds + 1):
            print(f">>> Seed {seed}/{self.num_seeds}  epsilon={epsilon:.4f}")
            for mode in self.modes:
                do_rec = (seed == self.num_seeds) and (mode in ["DRL-VoI","Static-VoI"])
                agvs, avg_mse = self._run_epoch(
                    net, target_net, optimizer, criterion, replay,
                    epsilon=epsilon, gamma=gamma,
                    batch=BATCH, tuf=TUF, grad_step=grad_step,
                    is_training=(mode=="DRL-VoI"), mode=mode,
                    seed=seed + 100, record_physics=do_rec
                )
                self.results[mode].append(avg_mse)
                if do_rec:
                    self.plot_data[mode] = agvs
            epsilon = max(0.01, epsilon * 0.82)
            if mode == "DRL-VoI":
                scheduler.step()

        self._report()
        self._plot(self.plot_data["DRL-VoI"], self.plot_data["Static-VoI"])

    # ──────────────────────────────────────────────
    def _run_epoch(self, net, target_net, optimizer, criterion,
                   replay, epsilon, gamma, batch, tuf, grad_step,
                   is_training, mode, seed, record_physics=False):

        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        agvs = [AGVAgent(i, self.cfg, self.mats) for i in range(self.cfg.num_agvs)]

        prev_s    = None
        prev_a    = None
        prev_mse  = None
        n_slots   = self.cfg.static_subcarriers   # DRL會覆寫這個值

        for t in range(self.cfg.sim_steps):

            # ── DRL-VoI 決策 ──
            if mode == "DRL-VoI":
                if t % self.cfg.drl_interval == 0:
                    curr_s   = self._extract_state(agvs, t)
                    curr_mse = np.mean([a.mse_log[-1] if a.mse_log else 0 for a in agvs])

                    # 存transition
                    if is_training and prev_s is not None:
                        # ★ Reward設計：
                        #   主項：MSE對數改善 × 大係數
                        #   次項：懲罰「佔用頻道越多」的機會成本
                        #   正向激勵：高誤差時多調度 → 誤差快速下降 → 大reward
                        mse_improve = (np.log1p(prev_mse) - np.log1p(curr_mse)) * 30.0
                        slot_cost   = (prev_a) * 0.3        # 動作index 0~4，少用資源
                        reward      = mse_improve - slot_cost

                        # 額外bonus：若高誤差AGV比例下降，給正向回饋
                        prev_high = (prev_mse > 5.0)
                        curr_high = (curr_mse > 5.0)
                        if prev_high and not curr_high:
                            reward += 5.0

                        replay.push(prev_s, prev_a, reward, curr_s, 0.0)

                    # 選動作
                    if is_training and random.random() < epsilon:
                        a_idx = random.randint(0, len(self.cfg.action_space)-1)
                    else:
                        with torch.no_grad():
                            a_idx = net(curr_s).argmax().item()

                    n_slots  = self.cfg.action_space[a_idx]
                    prev_s   = curr_s
                    prev_a   = a_idx
                    prev_mse = curr_mse

                    # 訓練
                    if is_training and len(replay) >= batch:
                        s_b, a_b, r_b, s2_b, d_b = replay.sample(batch)
                        with torch.no_grad():
                            # Double DQN：用online net選動作，target net算Q值
                            next_a  = net(s2_b).argmax(dim=1)
                            next_q  = target_net(s2_b).gather(1, next_a.unsqueeze(1)).squeeze()
                            tgt     = r_b + gamma * next_q * (1 - d_b)

                        pred = net(s_b).gather(1, a_b.unsqueeze(1)).squeeze()
                        loss = criterion(pred, tgt)
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        optimizer.step()

                        grad_step[0] += 1
                        if grad_step[0] % tuf == 0:
                            target_net.load_state_dict(net.state_dict())

                scheduled_ids = self._schedule_voi(agvs, n_slots)

            # ── Static-VoI：固定1個slot ──
            elif mode == "Static-VoI":
                scheduled_ids = self._schedule_voi(agvs, self.cfg.static_subcarriers)

            # ── AoI：挑最久未更新的1個 ──
            elif mode == "AoI":
                best = max(agvs, key=lambda a: a.age_since_tx)
                scheduled_ids = [best.uid]

            # ── Random ──
            else:
                scheduled_ids = [random.randint(0, self.cfg.num_agvs-1)]

            # ── 物理更新 ──
            tgt_path = self.get_target_path(t)
            for agv in agvs:
                u = np.clip(-(agv.K @ (agv.est_state - tgt_path)),
                            self.cfg.a_min, self.cfg.a_max)
                agv.update(u, agv.uid in scheduled_ids, t,
                           record_physics=record_physics)

        return agvs, np.mean([np.mean(a.mse_log) for a in agvs])

    # ──────────────────────────────────────────────
    def _report(self):
        stats = {m: np.mean(self.results[m]) for m in self.modes}
        df = pd.DataFrame.from_dict(stats, orient="index", columns=["Avg MSE"])
        base = stats["Static-VoI"]
        df["vs Static-VoI (%)"] = (base - df["Avg MSE"]) / base * 100
        base2 = stats["AoI"]
        df["vs AoI (%)"] = (base2 - df["Avg MSE"]) / base2 * 100
        print("\n" + "="*70)
        print("  實驗對比結果")
        print("="*70)
        print(df.sort_values("Avg MSE").round(4))
        print("="*70)

    # ──────────────────────────────────────────────
    def _plot(self, eng_drl, eng_static):
        steps  = self.cfg.sim_steps
        t_ax   = np.arange(steps)
        d0     = eng_drl[0]
        s0     = eng_static[0]
        refs   = np.array([self.get_target_path(t)[2:4] for t in range(steps)])

        c = {"D":"#1565C0","S":"#BF360C","R":"#2E7D32"}
        lw = 1.6

        fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
        fig.patch.set_facecolor("#FAFAFA")

        def smooth(arr, w=25):
            return pd.Series(arr).rolling(w, min_periods=1).mean().values

        # (a) X velocity
        axes[0].plot(t_ax, refs[:,0],         color=c["R"], ls="--", lw=1.2, label="Ref")
        axes[0].plot(t_ax, smooth(np.array(s0.vel_log)[:,0]),
                                               color=c["S"], ls=":",  lw=lw,  label="Standard-VoI")
        axes[0].plot(t_ax, smooth(np.array(d0.vel_log)[:,0]),
                                               color=c["D"], ls="-",  lw=lw,  label="DRL-VoI")
        axes[0].set_ylabel("X velocity (m/s)"); axes[0].set_ylim([5,16])
        axes[0].legend(loc="upper right", fontsize=8)

        # (b) Y velocity
        axes[1].plot(t_ax, refs[:,1],         color=c["R"], ls="--", lw=1.2, label="Ref")
        axes[1].plot(t_ax, smooth(np.array(s0.vel_log)[:,1]),
                                               color=c["S"], ls=":",  lw=lw,  label="Standard-VoI")
        axes[1].plot(t_ax, smooth(np.array(d0.vel_log)[:,1]),
                                               color=c["D"], ls="-",  lw=lw,  label="DRL-VoI")
        axes[1].set_ylabel("Y velocity (m/s)")
        axes[1].legend(loc="upper right", fontsize=8)

        # (c)(d) Acceleration
        for i, lbl in enumerate(["X acceleration)","Y acceleration)"]):
            axes[i+2].plot(t_ax, smooth(np.array(s0.acc_log)[:,i]),
                           color=c["S"], ls=":", lw=lw, label="Standard-VoI")
            axes[i+2].plot(t_ax, smooth(np.array(d0.acc_log)[:,i]),
                           color=c["D"], ls="-", lw=lw, label="DRL-VoI")
            axes[i+2].set_ylabel(lbl)
            axes[i+2].legend(loc="upper right", fontsize=8)

        # (e) Square Error — raw + smoothed
        axes[4].semilogy(t_ax, s0.mse_log, color=c["S"], alpha=0.15, lw=0.7)
        axes[4].semilogy(t_ax, smooth(s0.mse_log, 30),
                         color=c["S"], lw=2.2, label="Standard-VoI")
        axes[4].semilogy(t_ax, d0.mse_log, color=c["D"], alpha=0.15, lw=0.7)
        axes[4].semilogy(t_ax, smooth(d0.mse_log, 30),
                         color=c["D"], lw=2.2, label="DRL-VoI")
        axes[4].set_ylabel("Square Error")
        axes[4].legend(fontsize=9)

        # (f) Transmission timing
        axes[5].vlines(d0.tx_events,  0.6, 1.4, colors=c["D"], alpha=0.45, lw=0.8, label="DRL-VoI")
        axes[5].vlines(s0.tx_events, -0.4, 0.4,  colors=c["S"], alpha=0.45, lw=0.8, label="Standard-VoI")
        axes[5].set_yticks([0,1]); axes[5].set_yticklabels(["Standard","DRL"])
        axes[5].set_xlabel("Time Step"); axes[5].legend(loc="upper right", fontsize=8)

        for ax in axes:
            ax.set_facecolor("#F5F5F5")
            ax.grid(True, alpha=0.3)

        plt.suptitle("UAV Tracking: DRL-VoI vs Standard-VoI",
                     fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0,0.01,1,0.995])
        plt.savefig("fig6_drl_vs_static.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(">>> 圖已儲存：fig6_drl_vs_static.png")


if __name__ == "__main__":
    AcademicDRLComparison(num_seeds=10).run_experiments()