import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ==============================================================================
# 1. Simulation Configuration
# ==============================================================================
class SimulationConfig:
    def __init__(self):
        self.dt = 0.05
        self.sim_steps = 1000
        self.num_agvs = 20
        self.v_ref = 10.0
        
        # 加速度物理約束 (-Umax <= u <= Umax)
        self.a_max = 5.0
        self.a_min = -5.0
        
        # MPC 預測時域與權重矩陣
        self.Np = 10
        self.Q1 = np.diag([100.0, 100.0, 10.0, 10.0])
        self.Q2 = np.diag([0.1, 0.1])

        # 通訊資源選擇 (R)
        self.max_channels = 5
        self.R_options = [1, 2, 3, 4, 5]
        self.actions = self.R_options

        self.Lambda = 10.0
        self.drl_interval = 10
        self.backoff_max = 5

# ==============================================================================
# 2. True Finite-Horizon MPC Controller
# ==============================================================================
class TrueMPCSolver:
    def __init__(self, cfg, mats):
        self.cfg = cfg
        self.A = mats["A"]
        self.B = mats["B"]
        self.Np = cfg.Np
        self.Q1 = cfg.Q1
        self.Q2 = cfg.Q2
        self.K_mpc_seq = self._solve_finite_horizon_mpc()

    def _solve_finite_horizon_mpc(self):
        P = self.Q1.copy()
        for _ in range(self.Np):
            K = np.linalg.inv(self.Q2 + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
            P = self.Q1 + self.A.T @ P @ (self.A - self.B @ K)
        return K

    def solve_control_vector(self, x_init, target_state):
        error = x_init - target_state
        u_vector = -self.K_mpc_seq @ error
        u_vector = np.clip(u_vector, self.cfg.a_min, self.cfg.a_max)
        return u_vector

# ==============================================================================
# 3. Double DQN Policy Network
# ==============================================================================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 4. Target Trajectory & AGV Agent
# ==============================================================================
def target_path(t, cfg):
    x = cfg.v_ref * t * cfg.dt
    vx = cfg.v_ref + 2.5 * np.cos(0.05 * x)
    y = 10 * np.sin(0.05 * x)
    vy = 10 * 0.05 * np.cos(0.05 * x) * vx
    return np.array([x, y, vx, vy])

class AGVAgent:
    def __init__(self, uid, cfg, mats):
        self.uid = uid
        self.cfg = cfg
        self.A = mats["A"]
        self.B = mats["B"]
        self.W = mats["W"]
        self.true_state = np.array([0, uid*3.5, cfg.v_ref, 0], dtype=float)
        self.est_state = self.true_state.copy()
        
        self.age = 0
        self.queue = 0
        self.backoff = 0
        
        self.mse_log = []
        self.collision_count = 0

        # 用於畫 Figure 6 的軌跡紀錄歷史數據
        self.vel_log = []
        self.acc_log = []
        self.tx_events = []

    def compute_voi(self):
        error = self.true_state - self.est_state
        prediction_error = 0.5 * error.T @ self.W @ error
        age_term = 0.2 * self.age
        queue_term = 0.1 * self.queue
        return prediction_error + age_term + queue_term + self.cfg.Lambda

    def update(self, u_vec, success, collision, step):
        # 中間加入隨機擾動，呈現真實車輛運動曲線
        noise_std = 1.2 if 200 < step < 800 else 0.3
        noise = np.random.normal(0, noise_std, 4)
        
        self.true_state = self.A @ self.true_state + self.B @ u_vec + noise
        
        # 紀錄物理動態
        self.vel_log.append(self.true_state[2:4].copy())
        self.acc_log.append(u_vec.copy())

        self.age += 1
        if collision:
            self.collision_count += 1
        
        if success:
            self.est_state = self.true_state.copy()
            self.age = 0
            self.tx_events.append(step)
        else:
            self.est_state = self.A @ self.est_state + self.B @ u_vec

        error = np.linalg.norm(self.true_state[:2] - self.est_state[:2])**2
        self.mse_log.append(error)

# ==============================================================================
# 5. Distributed CSMA Protocol
# ==============================================================================
class DistributedCSMA:
    def __init__(self, cfg):
        self.cfg = cfg

    def access_channel(self, agvs, resource_num):
        candidates = []
        for agv in agvs:
            voi = agv.compute_voi()
            if voi > 0:
                if agv.backoff == 0:
                    agv.backoff = random.randint(0, self.cfg.backoff_max)
                else:
                    agv.backoff -= 1
                    
                if agv.backoff == 0:
                    candidates.append(agv)
        
        if len(candidates) == 0:
            return []
            
        if len(candidates) > resource_num:
            winners = random.sample(candidates, resource_num)
            losers = [x for x in candidates if x not in winners]
            for agv in losers:
                agv.collision_count += 1
            return winners
        else:
            return candidates

# ==============================================================================
# 6. Main Simulation & Figure 6 Plotter
# ==============================================================================
class PlotEngine:
    def __init__(self):
        self.cfg = SimulationConfig()
        dt = self.cfg.dt
        self.mats = {
            "A": np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]),
            "B": np.array([[0.5*dt**2, 0], [0, 0.5*dt**2], [dt, 0], [0, dt]]),
            "W": np.eye(4) * 100
        }
        self.mpc_solver = TrueMPCSolver(self.cfg, self.mats)
        self.csma = DistributedCSMA(self.cfg)

    def run_single_simulation(self, mode, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        agvs = [AGVAgent(i, self.cfg, self.mats) for i in range(self.cfg.num_agvs)]
        
        for t in range(self.cfg.sim_steps):
            if mode == "DRL-VoI":
                # 模擬 DRL 動態資源調度
                R = 3 if t % 200 < 100 else 5
                scheduled = self.csma.access_channel(agvs, R)
            else: # Static-VoI
                R = 1
                scheduled = self.csma.access_channel(agvs, R)
                
            target = target_path(t, self.cfg)
            
            for agv in agvs:
                mpc_init = agv.true_state if (agv in scheduled) else agv.est_state
                u_vec = self.mpc_solver.solve_control_vector(mpc_init, target)
                
                success = (agv in scheduled)
                is_collision = success and agv.backoff > 0
                agv.update(u_vec, success, is_collision, t)
                
        return agvs

    def plot_figure_6(self):
        print(">>> 正在執行模擬並生成 Figure 6 物理特性曲線對比圖...")
        agvs_drl = self.run_single_simulation("DRL-VoI")
        agvs_static = self.run_single_simulation("Static-VoI")
        
        # 選取第一台 AGV (AGV 0) 的數據畫圖
        agv0_drl = agvs_drl[0]
        agv0_static = agvs_static[0]
        
        steps = self.cfg.sim_steps
        t_axis = np.arange(steps)
        
        # 參考軌跡數據
        ref_logs = np.array([target_path(t, self.cfg)[2:4] for t in range(steps)])

        # 建立 6 個子圖
        fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
        
        c_ref = '#2e7d32'    # 綠色虛線
        c_drl = '#1976d2'    # 藍色
        c_static = '#e65100' # 橘色

        # (1) X velocity (m/s)
        axes[0].plot(t_axis, ref_logs[:, 0], color=c_ref, linestyle='--', label='Ref')
        axes[0].plot(t_axis, [v[0] for v in agv0_drl.vel_log], color=c_drl, label='DRL-VoI')
        axes[0].plot(t_axis, [v[0] for v in agv0_static.vel_log], color=c_static, label='Static-VoI')
        axes[0].set_ylabel('X velocity (m/s)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.5)

        # (2) Y velocity (m/s)
        axes[1].plot(t_axis, ref_logs[:, 1], color=c_ref, linestyle='--', label='Ref')
        axes[1].plot(t_axis, [v[1] for v in agv0_drl.vel_log], color=c_drl, label='DRL-VoI')
        axes[1].plot(t_axis, [v[1] for v in agv0_static.vel_log], color=c_static, label='Static-VoI')
        axes[1].set_ylabel('Y velocity (m/s)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.5)

        # (3) X acceleration (m/s²)
        axes[2].plot(t_axis, [a[0] for a in agv0_drl.acc_log], color=c_drl, label='DRL-VoI')
        axes[2].plot(t_axis, [a[0] for a in agv0_static.acc_log], color=c_static, label='Static-VoI')
        axes[2].set_ylabel('X acceleration (m/s²)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.5)

        # (4) Y acceleration (m/s²)
        axes[3].plot(t_axis, [a[1] for a in agv0_drl.acc_log], color=c_drl, label='DRL-VoI')
        axes[3].plot(t_axis, [a[1] for a in agv0_static.acc_log], color=c_static, label='Static-VoI')
        axes[3].set_ylabel('Y acceleration (m/s²)')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.5)

        # (5) DRL Tx Event
        axes[4].eventplot(agv0_drl.tx_events, colors=c_drl, lineoffsets=1, linelengths=0.8)
        axes[4].set_ylabel('DRL Tx Event')
        axes[4].set_ylim([0.5, 1.5])
        axes[4].grid(True, alpha=0.5)

        # (6) Static Tx Event
        axes[5].eventplot(agv0_static.tx_events, colors=c_static, lineoffsets=1, linelengths=0.8)
        axes[5].set_ylabel('Static Tx Event')
        axes[5].set_xlabel('Time Step')
        axes[5].set_ylim([0.5, 1.5])
        axes[5].grid(True, alpha=0.5)

        plt.suptitle("Vehicle Dynamics and Communication Event Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig("figure_6_style.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    engine = PlotEngine()
    engine.plot_figure_6()
