import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

# =============================================================================
# 1. 系統配置與 DRL 網絡
# =============================================================================
class SimulationConfig:
    def __init__(self):
        self.dt = 0.05
        self.sim_steps = 1000
        self.num_agvs = 20
        self.num_subcarriers = 1
        self.v_ref = 10.0
        self.a_max, self.a_min = 5.0, -5.0
        self.Np = 10
        self.Q1_diag = [100.0, 100.0, 10.0, 10.0] 
        self.Lambda = 10.0
        self.rho_limit = 0.1
        self.v_options = [1.0, 5.0, 10.0, 35.0, 100.0]
        self.drl_interval = 10 

class V_Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(V_Policy_Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x): 
        return self.fc(x)

# =============================================================================
# 2. 物理實體 (AGV/UAV Agent)
# =============================================================================
class AGVAgent:
    def __init__(self, uid, cfg, mats):
        self.uid, self.cfg = uid, cfg
        self.A, self.B, self.W, self.K = mats["A"], mats["B"], mats["W"], mats["K"]
        # 初始化 Y 軸偏移，確保圖形清晰
        self.true_state = np.array([0.0, uid * 3.5, cfg.v_ref, 0.0])
        self.est_state = self.true_state.copy()
        self.h_queue = 0.0
        self.mse_log = []
        
        # 繪圖緩衝
        self.vel_log = []   
        self.acc_log = []   
        self.tx_events = [] 

    def compute_voi(self, v_val):
        error = self.est_state - self.true_state
        voi_s = 0.5 * error.T @ self.W @ error - self.cfg.Lambda
        return voi_s - (self.h_queue / v_val)

    def update(self, u, success, step, record_physics=False):
        # 雜訊控制：確保波形不被過度淹沒
        noise_std = 1.5 if 200 < step < 800 else 0.2
        noise = np.random.normal(0, noise_std, 4)
        
        self.true_state = self.A @ self.true_state + self.B @ u + noise
        
        if record_physics:
            self.vel_log.append(self.true_state[2:4].copy())
            self.acc_log.append(u.copy())

        if success:
            self.est_state = self.true_state.copy()
            if record_physics: self.tx_events.append(step)
            self.h_queue = max(self.h_queue + 1.0 - self.cfg.rho_limit, 0)
        else:
            self.est_state = self.A @ self.est_state + self.B @ u
            self.h_queue = max(self.h_queue - self.cfg.rho_limit, 0)
            
        err = np.linalg.norm(self.true_state[:2] - self.est_state[:2])**2
        self.mse_log.append(err)

# =============================================================================
# 3. 實驗引擎
# =============================================================================
class AcademicDRLComparison:
    def __init__(self, num_seeds=10):
        self.cfg = SimulationConfig()
        self.num_seeds = num_seeds
        self.modes = ["DRL-VoI", "Static-VoI", "AoI", "Random"]
        self.results_data = {m: {"mse": []} for m in self.modes}
        self.plot_engines = {} 
        
        dt = self.cfg.dt
        self.mats = {
            "A": np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]),
            "B": np.array([[0.5*dt**2,0],[0,0.5*dt**2],[dt,0],[0,dt]]),
            "W": np.eye(4) * 100.0,
            "K": np.array([[1.0, 0, 0.6, 0], [0, 1.0, 0, 0.6]])
        }

    def get_target_path(self, t):
        """ 強化波形顯示：將 vx_ref 震幅從 0.5 提高到 2.5 """
        ref_x = self.cfg.v_ref * t * self.cfg.dt
        vx_ref = self.cfg.v_ref + 2.5 * np.cos(0.05 * ref_x) 
        vy_ref = 10.0 * 0.05 * np.cos(0.05 * ref_x) * vx_ref
        ref_y = 10.0 * np.sin(0.05 * ref_x)
        return np.array([ref_x, ref_y, vx_ref, vy_ref])

    def run_experiments(self):
        drl_net = V_Policy_Net(2, len(self.cfg.v_options))
        optimizer = optim.Adam(drl_net.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        epsilon = 0.3 

        print(">>> 正在進行網絡預訓練...")
        for p_seed in range(3):
            self._simulate_one_epoch(drl_net, optimizer, criterion, epsilon=0.5, 
                                     is_training=True, mode="DRL-VoI", seed=p_seed)

        for seed in range(1, self.num_seeds + 1):
            print(f">>> 執行 Seed {seed}/{self.num_seeds}...")
            for mode in self.modes:
                # 關鍵修改：確保紀錄 DRL-VoI 與 Static-VoI
                do_record = (seed == self.num_seeds and mode in ["DRL-VoI", "Static-VoI"])
                
                res_agvs, avg_mse = self._simulate_one_epoch(
                    drl_net, optimizer, criterion, epsilon, 
                    is_training=(mode=="DRL-VoI"), mode=mode, seed=seed, record_physics=do_record
                )
                self.results_data[mode]["mse"].append(avg_mse)
                
                if do_record:
                    self.plot_engines[mode] = res_agvs

            epsilon = max(0.01, epsilon * 0.85)

        self.final_report()
        # 繪圖對比：DRL vs Standard (Static)
        print(">>> 生成 DRL-VoI 與 Static-VoI 的物理特性對比圖...")
        self.plot_fig6_style(self.plot_engines["DRL-VoI"], self.plot_engines["Static-VoI"])

    def _simulate_one_epoch(self, net, optimizer, criterion, epsilon, is_training, mode, seed, record_physics=False):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        agvs = [AGVAgent(i, self.cfg, self.mats) for i in range(self.cfg.num_agvs)]
        v_current = 10.0
        
        for t in range(self.cfg.sim_steps):
            if mode == "DRL-VoI":
                if t % self.cfg.drl_interval == 0:
                    curr_mse = np.mean([a.mse_log[-1] if a.mse_log else 0 for a in agvs])
                    avg_h = np.mean([a.h_queue for a in agvs])
                    state = torch.tensor([curr_mse, avg_h], dtype=torch.float32)
                    v_idx = net(state).argmax().item()
                    if is_training and random.random() < epsilon:
                        v_idx = random.randint(0, len(self.cfg.v_options)-1)
                    v_current = self.cfg.v_options[v_idx]

                    if is_training:
                        reward = -(curr_mse * 10.0 + np.log(v_current)) 
                        target_q = net(state).clone().detach()
                        target_q[v_idx] = reward
                        loss = criterion(net(state), target_q)
                        optimizer.param_groups[0]['lr'] = 0.01
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                
                scores = [(a.uid, a.compute_voi(v_current)) for a in agvs]
            elif mode == "Static-VoI":
                # Standard-VoI: 固定 V = 10.0
                scores = [(a.uid, a.compute_voi(10.0)) for a in agvs]
            elif mode == "AoI":
                agvs_sorted = sorted(agvs, key=lambda a: len(a.mse_log), reverse=True)
                scheduled_ids = [agvs_sorted[0].uid]
            else: # Random
                scheduled_ids = [random.randint(0, self.cfg.num_agvs - 1)]

            if mode in ["DRL-VoI", "Static-VoI"]:
                scores.sort(key=lambda x: x[1], reverse=True)
                scheduled_ids = [s[0] for s in scores[:self.cfg.num_subcarriers]]

            target = self.get_target_path(t)
            for agv in agvs:
                u = np.clip(-(agv.K @ (agv.est_state - target)), self.cfg.a_min, self.cfg.a_max)
                agv.update(u, (agv.uid in scheduled_ids), t, record_physics=record_physics)

        return agvs, np.mean([np.mean(a.mse_log) for a in agvs])

    def final_report(self):
        final_stats = {m: np.mean(self.results_data[m]["mse"]) for m in self.modes}
        df = pd.DataFrame.from_dict(final_stats, orient='index', columns=['Avg MSE'])
        df["Improvement %"] = ((final_stats["AoI"] - df["Avg MSE"]) / final_stats["AoI"]) * 100
        print("\n" + "="*65)
        print(" 實驗對比結果：DRL-VoI vs Standard-VoI (Static) vs AoI ")
        print("="*65)
        print(df.sort_values(by="Avg MSE").round(4))
        print("="*65)

    def plot_fig6_style(self, engine_drl, engine_static):
        steps = self.cfg.sim_steps
        t_axis = np.arange(steps)
        u0_drl, u0_static = engine_drl[0], engine_static[0]

        ref_logs = []
        for t in range(steps):
            ref_data = self.get_target_path(t)
            ref_logs.append(ref_data[2:4]) 
        ref_logs = np.array(ref_logs)

        fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
        # 設置顏色：DRL 為藍色，Static 為橙色，Ref 為綠色虛線
        c = {"DRL": "#2962FF", "Static": "#FF6D00", "Ref": "#4CAF50"}

        # (a) & (b) Velocity
        for i, title in enumerate(['X velocity (m/s)', 'Y velocity (m/s)']):
            axes[i].plot(t_axis, ref_logs[:, i], color=c["Ref"], linestyle='--', label='Ref')
            axes[i].plot(t_axis, np.array(u0_drl.vel_log)[:, i], color=c["DRL"], label='DRL-VoI')
            axes[i].plot(t_axis, np.array(u0_static.vel_log)[:, i], color=c["Static"], linestyle=':', label='Standard-VoI')
            axes[i].set_ylabel(title); axes[i].legend(loc='upper right', fontsize=8)
            # 強制鎖定 X 軸範圍 (5~15) 讓 Sine 波動清晰可見
            if i == 0: axes[i].set_ylim([5, 15])

        # (c) & (d) Acceleration
        for i, label in enumerate(['X acceleration', 'Y acceleration']):
            axes[i+2].plot(t_axis, np.array(u0_drl.acc_log)[:, i], color=c["DRL"])
            axes[i+2].plot(t_axis, np.array(u0_static.acc_log)[:, i], color=c["Static"], linestyle=':')
            axes[i+2].set_ylabel(label)

        # (e) Square Error
        axes[4].plot(t_axis, u0_drl.mse_log, color=c["DRL"], label='DRL-VoI')
        axes[4].plot(t_axis, u0_static.mse_log, color=c["Static"], linestyle=':', label='Standard-VoI')
        axes[4].set_ylabel("Square Error"); axes[4].set_yscale('log'); axes[4].legend()

        # (f) Transmission Timing
        axes[5].vlines(u0_drl.tx_events, 0.6, 1.4, colors=c["DRL"], alpha=0.6, label='DRL-VoI')
        axes[5].vlines(u0_static.tx_events, -0.4, 0.4, colors=c["Static"], alpha=0.6, label='Standard-VoI')
        axes[5].set_yticks([0, 1]); axes[5].set_yticklabels(['Standard', 'DRL']); axes[5].set_xlabel("Time Step")

        plt.suptitle("UAV Tracking Dynamics: DRL-VoI vs Standard-VoI", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

if __name__ == "__main__":
    AcademicDRLComparison(num_seeds=10).run_experiments()