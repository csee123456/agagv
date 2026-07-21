import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import itertools

# ==============================================================================
# 1. Simulation Configuration
# ==============================================================================
class SimulationConfig:
    def __init__(self):
        # simulation
        self.dt = 0.05
        self.sim_steps = 1000
        self.num_agvs = 20
        # channel
        self.max_channels = 5
        # velocity
        self.v_ref = 10.0
        # control
        self.a_max = 5.0
        self.a_min = -5.0
        # MPC gain options
        self.U_options = [0.8, 1.0, 1.2]
        # Resource allocation
        self.R_options = [1, 2, 3, 4, 5]

        # Joint action space
        self.actions = []
        for r in self.R_options:
            for u in self.U_options:
                self.actions.append((r, u))

        # VoI related
        self.Lambda = 10
        # DRL related
        self.drl_interval = 10
        # CSMA/CD related
        self.backoff_max = 5
        self.collision_penalty = 5

        # ==============================================================================
        # 【變數參數設定區】
        # 這裡設定想要實驗對比的參數組合（例如：不同的 Lambda 值）
        # ==============================================================================
        self.varying_params = {
            'Lambda': [5, 15]  # 比較不同 Lambda 對系統平均 MSE 的影響
        }

# ==============================================================================
# 2. Double DQN Network
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
# 3. Replay Buffer
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity=3000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        data = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*data)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(s2),
            torch.tensor(d, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# 4. AGV Agent
# ==============================================================================
class AGVAgent:
    def __init__(self, uid, cfg, mats):
        self.uid = uid
        self.cfg = cfg
        self.A = mats["A"]
        self.B = mats["B"]
        self.K = mats["K"]
        self.W = mats["W"]
        self.true_state = np.array([0, uid*3.5, cfg.v_ref, 0], dtype=float)
        self.est_state = self.true_state.copy()
        
        # communication status
        self.age = 0
        self.queue = 0
        self.backoff = 0
        
        # statistics
        self.mse_log = []
        self.energy_log = []
        self.tx_count = 0
        self.collision_count = 0
        self.tx_events = []
        self.vel_log = []
        self.acc_log = []

    def compute_voi(self):
        error = self.true_state - self.est_state
        prediction_error = 0.5 * error.T @ self.W @ error
        age_term = 0.2 * self.age
        queue_term = 0.1 * self.queue
        return prediction_error + age_term + queue_term + self.cfg.Lambda

    def update(self, u, success, collision, step):
        noise = np.random.normal(0, 0.3, 4)
        self.true_state = self.A @ self.true_state + self.B @ u + noise
        
        energy = np.linalg.norm(u)**2
        self.energy_log.append(energy)
        self.vel_log.append(self.true_state[2:4].copy())
        self.acc_log.append(u.copy())
        
        self.age += 1
        if collision:
            self.collision_count += 1
        
        if success:
            self.est_state = self.true_state.copy()
            self.age = 0
            self.tx_count += 1
            self.tx_events.append(step)
        else:
            self.est_state = self.A @ self.est_state + self.B @ u

        error = np.linalg.norm(self.true_state[:2] - self.est_state[:2])**2
        self.mse_log.append(error)

# ==============================================================================
# 5. MPC Controller
# ==============================================================================
def MPC_control(agv, target, gain, cfg):
    u = -gain * (agv.K @ (agv.est_state - target))
    u = np.clip(u, cfg.a_min, cfg.a_max)
    return u

# ==============================================================================
# 6. Distributed CSMA/CD Protocol
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
# 7. Target Trajectory
# ==============================================================================
def target_path(t, cfg):
    x = cfg.v_ref * t * cfg.dt
    vx = cfg.v_ref + 2.5 * np.cos(0.05 * x)
    y = 10 * np.sin(0.05 * x)
    vy = 10 * 0.05 * np.cos(0.05 * x) * vx
    return np.array([x, y, vx, vy])

# ==============================================================================
# 8. Joint DRL Controller
# ==============================================================================
class JointDRLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_dim = 8
        self.action_dim = len(cfg.actions)
        self.policy = PolicyNet(self.state_dim, self.action_dim)
        self.target = PolicyNet(self.state_dim, self.action_dim)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2e-4)
        self.memory = ReplayBuffer()
        self.gamma = 0.97
        self.loss_fn = nn.SmoothL1Loss()
        self.steps = 0

    def get_state(self, agvs, collision_rate, t_step):
        mse = np.array([a.mse_log[-1] if len(a.mse_log) > 0 else 0 for a in agvs])
        age = np.array([a.age for a in agvs])
        queue = np.array([a.queue for a in agvs])
        
        state = np.array([
            np.mean(np.log1p(mse))/10,
            np.max(np.log1p(mse))/10,
            np.mean(age)/50,
            np.max(age)/100,
            np.mean(queue)/10,
            collision_rate,
            np.mean(mse > 10),
            t_step / self.cfg.sim_steps
        ], dtype=np.float32)
        return torch.tensor(state)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                idx = self.policy(state).argmax().item()
        return idx

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
            
        s, a, r, s2, d = self.memory.sample(batch_size)
        
        with torch.no_grad():
            next_action = self.policy(s2).argmax(dim=1)
            next_q = self.target(s2).gather(1, next_action.unsqueeze(1)).squeeze()
            target_q = r + self.gamma * next_q * (1 - d)
            
        current = self.policy(s).gather(1, a.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.policy.state_dict())

# ==============================================================================
# 9. Joint Reward
# ==============================================================================
def calculate_reward(prev_mse, curr_mse, action, energy, collision):
    R, U = action
    improvement = np.log1p(prev_mse) - np.log1p(curr_mse)
    reward = (50 * improvement) - (0.5 * R) - (0.01 * energy) - (5 * collision)
    return reward

# ==============================================================================
# 10. Experiment Engine
# ==============================================================================
class Experiment:
    def __init__(self, seeds=3):
        self.cfg = SimulationConfig()
        self.seeds = seeds
        self.modes = ["DRL-VoI", "Static-VoI", "AoI", "Random"]
        
        dt = self.cfg.dt
        self.mats = {
            "A": np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]),
            "B": np.array([[0.5*dt**2, 0], [0, 0.5*dt**2], [dt, 0], [0, dt]]),
            "K": np.array([[1, 0, 0.6, 0], [0, 1, 0, 0.6]]),
            "W": np.eye(4) * 100
        }
        
        self.drl = JointDRLController(self.cfg)
        self.csma = DistributedCSMA(self.cfg)

    def run_episode(self, mode, seed, train=False):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        agvs = [AGVAgent(i, self.cfg, self.mats) for i in range(self.cfg.num_agvs)]
        epsilon = 0.3 if train else 0.0
        prev_mse = 0
        
        # 預設初始化，防止測試非訓練集或非決策步時 NameError
        action_idx = 0
        R, U = self.cfg.actions[action_idx]
        state = self.drl.get_state(agvs, 0, 0)
        
        for t in range(self.cfg.sim_steps):
            if mode == "DRL-VoI":
                if t % self.cfg.drl_interval == 0:
                    collision_rate = np.mean([a.collision_count for a in agvs]) / 100
                    state = self.drl.get_state(agvs, collision_rate, t)
                    action_idx = self.drl.select_action(state, epsilon)
                R, U = self.cfg.actions[action_idx]
                scheduled = self.csma.access_channel(agvs, R)
            elif mode == "Static-VoI":
                R = self.cfg.max_channels
                U = 1.0
                scheduled = self.csma.access_channel(agvs, R)
            elif mode == "AoI":
                target_agent = max(agvs, key=lambda x: x.age)
                scheduled = [target_agent]
                R = 1
                U = 1.0
            else: # Random
                scheduled = [random.choice(agvs)]
                R = 1
                U = 1.0
                
            target = target_path(t, self.cfg)
            total_energy = 0
            collision_step = 0
            
            for agv in agvs:
                u = MPC_control(agv, target, U, self.cfg)
                success = (agv in scheduled)
                is_collision = success and agv.backoff > 0
                
                agv.update(u, success, is_collision, t)
                if is_collision:
                    collision_step += 1
                total_energy += np.linalg.norm(u)**2
                
            if mode == "DRL-VoI" and train and (t % self.cfg.drl_interval == self.cfg.drl_interval - 1):
                curr_mse = np.mean([np.mean(a.mse_log) for a in agvs])
                reward = calculate_reward(prev_mse, curr_mse, (R, U), total_energy, collision_step)
                collision_rate_next = np.mean([a.collision_count for a in agvs]) / 100
                next_state = self.drl.get_state(agvs, collision_rate_next, t)
                
                self.drl.memory.push(state, action_idx, reward, next_state, 0)
                self.drl.train_step()
                prev_mse = curr_mse
        
        avg_mse = np.mean([np.mean(a.mse_log) for a in agvs])
        collision = np.sum([a.collision_count for a in agvs])
        tx = np.sum([a.tx_count for a in agvs])
        
        return {"mse": avg_mse, "collision": collision, "tx": tx}

    def run_all(self):
        params_keys = list(self.cfg.varying_params.keys())
        params_values = list(self.cfg.varying_params.values())
        param_combinations = list(itertools.product(*params_values))

        combined_results = {m: {} for m in self.modes}

        # 先讓 DRL 在預設狀態下進行基礎預訓練，避免全然隨機
        print("Pre-training DRL Agent on default environment...")
        for seed in range(self.seeds):
            _ = self.run_episode("DRL-VoI", seed, train=True)

        # 開始正式進行多變數參數對比測試
        for seed in range(self.seeds):
            print(f"Running evaluation with Seed {seed}...")
            for combo in param_combinations:
                current_params = dict(zip(params_keys, combo))
                combo_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
                
                # 備份並動態複寫環境變數
                original_values = {}
                for k, v in current_params.items():
                    original_values[k] = getattr(self.cfg, k)
                    setattr(self.cfg, k, v)
                
                for mode in self.modes:
                    # 測試階段關閉 DRL 的探索 (train=False)
                    result = self.run_episode(mode, seed, train=False)
                    
                    if combo_str not in combined_results[mode]:
                        combined_results[mode][combo_str] = []
                    combined_results[mode][combo_str].append(result)
                    
                # 還原環境變數
                for k, v in original_values.items():
                    setattr(self.cfg, k, v)
        
        self.report_all(combined_results, param_combinations, params_keys)
        self.plot_all(combined_results, param_combinations, params_keys)

    def report_all(self, combined_results, param_combinations, params_keys):
        final_table = []
        for combo in param_combinations:
            current_params = dict(zip(params_keys, combo))
            combo_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
            for mode in self.modes:
                results_list = combined_results[mode][combo_str]
                mse = np.mean([x["mse"] for x in results_list])
                collision = np.mean([x["collision"] for x in results_list])
                tx = np.mean([x["tx"] for x in results_list])
                final_table.append([combo_str, mode, mse, collision, tx])
                
        df = pd.DataFrame(final_table, columns=["Params", "Mode", "MSE", "Collision", "Transmission"])
        print("\n===== Final Combined Results =====")
        print(df.round(4))
        df.to_csv("result_all.csv", index=False)

    def plot_all(self, combined_results, param_combinations, params_keys):
        plt.figure(figsize=(10, 6))
        
        num_combos = len(param_combinations)
        bar_width = 0.6 / num_combos
        mode_indices = np.arange(len(self.modes))
        
        for i, combo in enumerate(param_combinations):
            current_params = dict(zip(params_keys, combo))
            combo_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
            
            mse_vals = []
            for mode in self.modes:
                results_list = combined_results[mode][combo_str]
                mse_vals.append(np.mean([x["mse"] for x in results_list]))
            
            pos = mode_indices + (i - (num_combos - 1) / 2) * bar_width
            plt.bar(pos, mse_vals, bar_width, label=f"({combo_str})")
            
        plt.xticks(mode_indices, self.modes)
        plt.ylabel("Average MSE")
        plt.title(f"Joint DRL-VoI Performance Comparison under Different Parameters")
        plt.legend(title="Parameter Specs")
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.savefig("mse_result_all.png", dpi=300, bbox_inches='tight')
        plt.show()

# ==============================================================================
# Main入口
# ==============================================================================
if __name__ == "__main__":
    # 為快速測試，預設 seed 降為 2，可自行調大
    exp = Experiment(seeds=2) 
    exp.run_all()
