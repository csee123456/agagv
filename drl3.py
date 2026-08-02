import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

# =============================================================================
# 1. 系統配置與 DRL 網絡 (Joint Optimization: R + U)
# =============================================================================
class SimulationConfig:
    def __init__(self):
        self.dt = 0.05
        self.sim_steps = 1000
        self.num_agvs = 20
        self.v_ref = 10.0
        self.a_max, self.a_min = 5.0, -5.0
        self.Np = 10
        self.Q1_diag = [100.0, 100.0, 10.0, 10.0] 
        self.Lambda = 10.0
        self.rho_limit = 0.1
        self.drl_interval = 10 
        
        # 聯合最佳化 (Joint Optimization) 變數選項
        self.R_options = [1, 2, 3, 4, 5]          # 通訊 Subcarrier 數量
        self.U_options = [0.8, 1.0, 1.2]          # 控制增益 (Control Gain Scale)
        
        # 建立 15 個離散動作空間 [(R1, U1), (R1, U2), ...]
        self.action_space = [(r, u) for u in self.U_options for r in self.R_options]
        self.action_dim = len(self.action_space)  # 15

class JointPolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(JointPolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x): 
        return self.fc(x)

# =============================================================================
# 2. 物理實體 (AGV Agent)
# =============================================================================
class AGVAgent:
    def __init__(self, uid, cfg, mats):
        self.uid, self.cfg = uid, cfg
        self.A, self.B, self.W, self.K = mats["A"], mats["B"], mats["W"], mats["K"]
        self.true_state = np.array([0.0, uid * 3.5, cfg.v_ref, 0.0])
        self.est_state = self.true_state.copy()
        self.h_queue = 0.0
        self.mse_log = []
        
        # 繪圖緩衝
        self.vel_log = []   
        self.acc_log = []   
        self.tx_events = [] 

    def compute_voi(self, r_val):
        error = self.est_state - self.true_state
        voi_s = 0.5 * error.T @ self.W @ error - self.cfg.Lambda
        return voi_s - (self.h_queue / float(r_val))

    def update(self, u, success, step, record_physics=False):
        # 雜訊控制
        noise_std = 1.5 if 200 < step < 800 else 0.2
        noise = np.random.normal(0, noise_std, 4)
        
        self.true_state = self.A @ self.true_state + self.B @ u + noise
        
        if record_physics:
            self.vel_log.append(self.true_state[2:4].copy())
            self.acc_log.append(u.copy())

        if success:
            self.est_state = self.true_state.copy()
            if record_physics: 
                self.tx_events.append(step)
            self.h_queue = max(self.h_queue + 1.0 - self.cfg.rho_limit, 0)
        else:
            self.est_state = self.A @ self.est_state + self.B @ u
            self.h_queue = max(self.h_queue - self.cfg.rho_limit, 0)
            
        err = np.linalg.norm(self.true_state[:2] - self.est_state[:2])**2
        self.mse_log.append(err)

# =============================================================================
# 3. 實驗引擎 (DRL Joint Optimization Comparison)
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
        ref_x = self.cfg.v_ref * t * self.cfg.dt
        vx_ref = self.cfg.v_ref + 2.5 * np.cos(0.05 * ref_x) 
        vy_ref = 10.0 * 0.05 * np.cos(0.05 * ref_x) * vx_ref
        ref_y = 10.0 * np.sin(0.05 * ref_x)
        return np.array([ref_x, ref_y, vx_ref, vy_ref])

    def run_experiments(self):
        drl_net = JointPolicyNet(2, self.cfg.action_dim)
        optimizer = optim.Adam(drl_net.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        epsilon = 0.3 

        print(">>> 正在進行網絡預訓練 (R+U Joint Optimization)...")
        for p_seed in range(3):
            self._simulate_one_epoch(drl_net, optimizer, criterion, epsilon=0.5, 
                                     is_training=True, mode="DRL-VoI", seed=p_seed)

        for seed in range(1, self.num_seeds + 1):
            print(f">>> 執行 Seed {seed}/{self.num_seeds}...")
            for mode in self.modes:
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
        print(">>> 生成 DRL-VoI 與 Static-VoI 的物理特性對比圖...")
        self.plot_fig6_style(self.plot_engines["DRL-VoI"], self.plot_engines["Static-VoI"])

    def _simulate_one_epoch(self, net, optimizer, criterion, epsilon, is_training, mode, seed, record_physics=False):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        agvs = [AGVAgent(i, self.cfg, self.mats) for i in range(self.cfg.num_agvs)]
        
        # 預設動作參數
        r_current = 1
        u_gain_current = 1.0
        
        for t in range(self.cfg.sim_steps):
            if mode == "DRL-VoI":
                if t % self.cfg.drl_interval == 0:
                    curr_mse = np.mean([a.mse_log[-1] if a.mse_log else 0 for a in agvs])
                    avg_h = np.mean([a.h_queue for a in agvs])
                    state = torch.tensor([curr_mse, avg_h], dtype=torch.float32)
                    
                    action_idx = net(state).argmax().item()
                    if is_training and random.random() < epsilon:
                        action_idx = random.randint(0, self.cfg.action_dim - 1)
                    
                    # 解碼成 Joint Actions (R, U)
                    r_current, u_gain_current = self.cfg.action_space[action_idx]

                    if is_training:
                        # Reward: 扣除 MSE、資源消耗 (alpha * R) 與控制偏差成本 (beta * |U - 1|)
                        alpha, beta = 0.5, 2.0
                        reward = -(curr_mse * 10.0 + alpha * r_current + beta * abs(u_gain_current - 1.0))
                        
                        target_q = net(state).clone().detach()
                        target_q[action_idx] = reward
                        loss = criterion(net(state), target_q)
                        optimizer.param_groups[0]['lr'] = 0.01
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                
                scores = [(a.uid, a.compute_voi(r_current)) for a in agvs]
                scores.sort(key=lambda x: x[1], reverse=True)
                scheduled_ids = [s[0] for s in scores[:r_current]]
                
            elif mode == "Static-VoI":
                # 基線: 固定 R=1, U=1.0
                r_current, u_gain_current = 1, 1.0
                scores = [(a.uid, a.compute_voi(r_current)) for a in agvs]
                scores.sort(key=lambda x: x[1], reverse=True)
                scheduled_ids = [s[0] for s in scores[:r_current]]
                
            elif mode == "AoI":
                u_gain_current = 1.0
                agvs_sorted = sorted(agvs, key=lambda a: len(a.mse_log), reverse=True)
                scheduled_ids = [agvs_sorted[0].uid]
            else: # Random
                u_gain_current = 1.0
                scheduled_ids = [random.randint(0, self.cfg.num_agvs - 1)]

            target = self.get_target_path(t)
            for agv in agvs:
                # 根據強化的 Gain U 調整 Base Controller 輸出
                base_u = -(agv.K @ (agv.est_state - target))
                scaled_u = np.clip(u_gain_current * base_u, self.cfg.a_min, self.cfg.a_max)
                agv.update(scaled_u, (agv.uid in scheduled_ids), t, record_physics=record_physics)

        return agvs, np.mean([np.mean(a.mse_log) for a in agvs])

    def final_report(self):
        final_stats = {m: np.mean(self.results_data[m]["mse"]) for m in self.modes}
        df = pd.DataFrame.from_dict(final_stats, orient='index', columns=['Avg MSE'])
        df["Improvement %"] = ((final_stats["AoI"] - df["Avg MSE"]) / final_stats["AoI"]) * 100
        print("\n" + "="*65)
        print(" 實驗對比結果：DRL (R+U Joint) vs Standard-VoI vs AoI ")
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
        c = {"DRL": "#2962FF", "Static": "#FF6D00", "Ref": "#4CAF50"}

        # (a) & (b) Velocity
        for i, title in enumerate(['X velocity (m/s)', 'Y velocity (m/s)']):
            axes[i].plot(t_axis, ref_logs[:, i], color=c["Ref"], linestyle='--', label='Ref')
            axes[i].plot(t_axis, [v[i] for v in u0_drl.vel_log], color=c["DRL"], label='DRL-VoI')
            axes[i].plot(t_axis, [v[i] for v in u0_static.vel_log], color=c["Static"], label='Static-VoI')
            axes[i].set_ylabel(title)
            axes[i].legend(loc='upper right')
            axes[i].grid(True)

        # (c) & (d) Acceleration
        for i, title in enumerate(['X acceleration (m/s²)', 'Y acceleration (m/s²)']):
            idx = i + 2
            axes[idx].plot(t_axis, [a[i] for a in u0_drl.acc_log], color=c["DRL"], label='DRL-VoI')
            axes[idx].plot(t_axis, [a[i] for a in u0_static.acc_log], color=c["Static"], label='Static-VoI')
            axes[idx].set_ylabel(title)
            axes[idx].legend(loc='upper right')
            axes[idx].grid(True)

        # (e) & (f) Transmission Events (Tx)
        axes[4].eventplot(u0_drl.tx_events, colors=c["DRL"], lineoffsets=1, linelengths=0.5)
        axes[4].set_ylabel('DRL Tx Event')
        axes[4].grid(True)

        axes[5].eventplot(u0_static.tx_events, colors=c["Static"], lineoffsets=1, linelengths=0.5)
        axes[5].set_ylabel('Static Tx Event')
        axes[5].set_xlabel('Time Step')
        axes[5].grid(True)

        plt.tight_layout()
        plt.show()

# =============================================================================
# 4. 程式執行入口
# =============================================================================
if __name__ == "__main__":
    sim = AcademicDRLComparison(num_seeds=5)
    sim.run_experiments()