

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from collections import deque
import random
import copy

# ==========================================
# 1. 系統與模擬參數設定 (對照論文 Table I)
# ==========================================
class Config:
    def __init__(self):
        # 基本時間與模擬設定
        self.dt = 0.05               # Sample period tp (50 ms)
        self.sim_steps = 400         # 模擬總步數
        self.num_agvs = 15           # 工廠內的 AGV 總數量
        self.num_subcarriers = 4     # 可用的網路子載波數量 (頻寬受限 N_t < num_agvs)
        
        # AGV 動力學參數
        self.v_min, self.v_max = -2.0, 5.0  # 速度限制 (工廠環境較慢 m/s)
        self.a_min, self.a_max = -3.0, 3.0  # 加速度限制 (m/s^2)
        
        # MPC 控制參數
        self.Np = 5                  # Prediction horizon N_p
        
        # 成本權重矩陣 Q1 (狀態誤差懲罰), Q2 (控制輸出懲罰)
        self.Q1_diag = [10.0, 10.0, 1.0, 1.0] # 更加注重位置 (x, y) 的誤差
        self.Q2_diag = [0.1, 0.1]
        
        # VoI 與 Lyapunov 優化權重參數
        self.V_param = 10.0          # Lyapunov Weight parameter V
        self.Lambda = 10.0           # Communication cost weight \lambda
        self.rho = 0.1               # Maximum communication frequency \rho
        
        # 環境與網路噪聲
        self.process_noise_var = 0.05 # 動力學干擾 (對應論文中的 \omega_t)
        self.p_success = 0.95        # 基本封包傳送成功率 (簡化 Eq. 3)

# ==========================================
# 2. 數學與矩陣工具 (對應 Appendix A)
# ==========================================
class MathUtils:
    @staticmethod
    def build_mpc_matrices(config):
        """
        建立狀態空間模型與 MPC 預測矩陣 (Eq. 10, 11, 12, 13)
        x_{t+1} = A x_t + B u_t
        X = C x_t + D U
        """
        dt = config.dt
        Np = config.Np
        
        # 系統矩陣 A (4x4)
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 控制矩陣 B (4x2)
        B = np.array([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])
        
        # 建立預測矩陣 C (4Np x 4)
        C = np.zeros((4 * Np, 4))
        for i in range(Np):
            C[i*4:(i+1)*4, :] = np.linalg.matrix_power(A, i+1)
            
        # 建立預測矩陣 D (4Np x 2Np)
        D = np.zeros((4 * Np, 2 * Np))
        for i in range(Np):
            for j in range(i + 1):
                D[i*4:(i+1)*4, j*2:(j+1)*2] = np.dot(np.linalg.matrix_power(A, i-j), B)
                
        # 建立權重對角矩陣 Q1_bar 與 Q2_bar
        Q1_bar = np.kron(np.eye(Np), np.diag(config.Q1_diag))
        Q2_bar = np.kron(np.eye(Np), np.diag(config.Q2_diag))
        
        # 計算 Eq. 12 中定義的矩陣 E, H
        # J = X^T Q1 X + U^T Q2 U
        # U* = - E^-1 H x_t
        E_mat = np.dot(D.T, np.dot(Q1_bar, D)) + Q2_bar
        H_mat = np.dot(D.T, np.dot(Q1_bar, C))
        
        # Eq. 14: \Delta J = 1/2 e^T W e, 其中 W = H^T E^-1 H
        E_inv = np.linalg.inv(E_mat)
        W_mat = np.dot(H_mat.T, np.dot(E_inv, H_mat))
        
        return A, B, C, D, E_mat, E_inv, H_mat, W_mat

# ==========================================
# 3. AGV 節點定義 (包含 VoI 評估邏輯)
# ==========================================
class AGV:
    def __init__(self, agv_id, start_state, config, math_utils):
        self.agv_id = agv_id
        self.config = config
        self.A, self.B, self.C, self.D, self.E_mat, self.E_inv, self.H_mat, self.W_mat = math_utils
        
        # 狀態變數: x = [pos_x, pos_y, vel_x, vel_y]^T
        self.true_state = np.array(start_state, dtype=float)
        
        # 控制器對 AGV 狀態的預估值 (由於網路延遲或排程失敗，可能會有誤差)
        self.est_state = np.array(start_state, dtype=float)
        
        # 歷史軌跡紀錄
        self.history_true_x = []
        self.history_true_y = []
        self.history_ref_x = []
        self.history_ref_y = []
        
        # 狀態誤差 \Theta (Eq. 19)
        self.Theta = np.zeros(4) 
        
        # 虛擬通訊隊列 H (Eq. 20)
        self.H_queue = 0.0
        
        # 分布式 CSMA/CA 的退避參數 (Algorithm 1)
        self.C_index = 2 # 初始 CW 指數
        self.T_back = 0
        self.packet_ready = True
        
        # AoI (Age of Information) 計數器
        self.aoi = 0

    def generate_reference_path(self, step):
        """工廠環境的參考路徑 (例如: 繞行料架的 S 型或正弦軌跡)"""
        # 以 x 軸等速前進，y 軸依照正弦波動模擬閃避障礙
        ref_x = 2.0 * (step * self.config.dt)
        ref_y = 5.0 * np.sin(0.5 * ref_x)
        # 參考速度為微分
        ref_vx = 2.0
        ref_vy = 5.0 * 0.5 * 2.0 * np.cos(0.5 * ref_x)
        return np.array([ref_x, ref_y, ref_vx, ref_vy])

    def evaluate_short_term_voi(self):
        
        # 誤差 e(t) = 控制器預估狀態 - AGV 實際狀態
        e_t = self.est_state - self.true_state
        
        # 矩陣相乘: e^T * W * e
        val = 0.5 * np.dot(e_t.T, np.dot(self.W_mat, e_t))
        voi_s = val - self.config.Lambda
        return voi_s

    def evaluate_long_term_voi(self):
        
        e_t = self.est_state - self.true_state
        
        # 針對論文 Eq.36 的簡化實現: 以誤差的加權和減去虛擬隊列(發送頻率)的懲罰
        # \xi_n 權重可直接對應到狀態誤差的影響力
        xi = np.array(self.config.Q1_diag) 
        error_term = np.sum(xi * (e_t ** 2))
        
        # 虛擬隊列懲罰: 如果近期發送太頻繁，H_queue 會很大，從而降低 VoI
        freq_penalty = self.H_queue / self.config.V_param
        
        long_term_voi = error_term - self.config.Lambda - freq_penalty
        return long_term_voi

    def physical_update(self, control_u):
        """
        AGV 實體狀態更新 (Eq. 4, 5)
        加入工廠環境特有的地面摩擦或馬達干擾 (Noise)
        """
        noise = np.random.normal(0, self.config.process_noise_var, 4)
        self.true_state = np.dot(self.A, self.true_state) + np.dot(self.B, control_u) + noise
        
        # 記錄歷史
        self.history_true_x.append(self.true_state[0])
        self.history_true_y.append(self.true_state[1])
        self.aoi += 1

    def estimator_update(self, control_u, successful_tx):
        """
        邊緣控制器端對 AGV 狀態的預估 (Eq. 5, 沒有 noise)
        如果通訊成功，預估狀態會校正為實際狀態
        """
        if successful_tx:
            # R_i,t * P_i,t = 1 (通訊成功，誤差歸零)
            self.est_state = np.copy(self.true_state)
            self.Theta = np.zeros(4)
            self.aoi = 0
        else:
            # 通訊失敗，按照物理模型猜測
            self.est_state = np.dot(self.A, self.est_state) + np.dot(self.B, control_u)
            # 狀態誤差隊列累積 (Eq. 19 簡化)
            self.Theta += np.abs(self.est_state - self.true_state)

    def queue_update(self, successful_tx):
        """
        更新 Lyapunov 虛擬隊列 H (Eq. 20)
        H_{t+1} = max(H_t + R_t - \rho, 0)
        """
        R_t = 1.0 if successful_tx else 0.0
        self.H_queue = max(self.H_queue + R_t - self.config.rho, 0.0)

    def distributed_csma_ca_update(self, voi, successful_tx):
        """
        基於 VoI 的分散式 CSMA/CA 協議 (Algorithm 1)
        根據 VoI 調整競爭窗口 (Contention Window)
        """
        # Eq. 42: 更新指數 C_t
        if voi <= 0:
            self.C_index = min(self.C_index + 1, 5) # 價值低，退避增加 (讓出頻寬)
        else:
            self.C_index = max(self.C_index - 1, 0) # 價值高，退避減少 (搶佔頻寬)
            
        # Eq. 43: 設定退避計時器 T_back = 2^{C_t} + random
        cw_size = 2 ** self.C_index
        alpha = 7 # 論文中設定的隨機範圍參數
        self.T_back = cw_size + random.randint(0, alpha)

# ==========================================
# 4. 控制器與調度器 (RSU / Edge Server)
# ==========================================
class EdgeController:
    def __init__(self, config, math_utils):
        self.config = config
        self.A, self.B, self.C, self.D, self.E_mat, self.E_inv, self.H_mat, self.W_mat = math_utils

    def calculate_mpc_control(self, agv, step):
        """
        計算最佳控制力 u_t (Eq. 13)
        注意: 控制器只能看到 agv.est_state (預估狀態)，而非真值
        """
        # 產生未來 Np 步的參考軌跡向量 X_ref
        X_ref = np.zeros(4 * self.config.Np)
        for i in range(self.config.Np):
            ref_t = agv.generate_reference_path(step + i)
            X_ref[i*4:(i+1)*4] = ref_t
            
        if step == 0:
            agv.history_ref_x.append(ref_t[0])
            agv.history_ref_y.append(ref_t[1])
            
        # 計算誤差向量 E_vec = C * x_est - X_ref
        C_x = np.dot(self.C, agv.est_state)
        E_vec = C_x - X_ref
        
        # U* = - E^-1 * D^T * Q1 * E_vec (根據 Appendix A 推導)
        # 這裡為了優化計算，直接解線性方程
        Q1_bar = np.kron(np.eye(self.config.Np), np.diag(self.config.Q1_diag))
        temp = np.dot(self.D.T, np.dot(Q1_bar, E_vec))
        U_opt = -np.dot(self.E_inv, temp)
        
        # 提取當前步驟的控制指令 [ax, ay] 並加入硬體限制
        u_t = U_opt[0:2]
        u_t[0] = np.clip(u_t[0], self.config.a_min, self.config.a_max)
        u_t[1] = np.clip(u_t[1], self.config.a_min, self.config.a_max)
        
        # 記錄當下參考點用於繪圖
        ref_current = agv.generate_reference_path(step)
        agv.history_ref_x.append(ref_current[0])
        agv.history_ref_y.append(ref_current[1])
        
        return u_t

    def centralized_voi_scheduling(self, agvs):
        """
        集中式 VoI 調度策略 (Eq. 38)
        計算所有 AGV 的 VoI 並排序，分配給 Top N_t
        """
        voi_list = []
        for i, agv in enumerate(agvs):
            voi = agv.evaluate_long_term_voi()
            voi_list.append((i, voi))
            
        # 根據 VoI 降序排列
        voi_list.sort(key=lambda x: x[1], reverse=True)
        
        # 分配子載波
        allocated_indices = [x[0] for x in voi_list[:self.config.num_subcarriers]]
        return allocated_indices

    def aoi_scheduling(self, agvs):
        """
        Baseline: 基於 AoI (Age of Information) 的調度策略
        挑選 AoI 最大的 AGV 優先傳輸
        """
        aoi_list = [(i, agv.aoi) for i, agv in enumerate(agvs)]
        aoi_list.sort(key=lambda x: x[1], reverse=True)
        allocated_indices = [x[0] for x in aoi_list[:self.config.num_subcarriers]]
        return allocated_indices

# ==========================================
# 5. 主模擬流程
# ==========================================
class Simulator:
    def __init__(self, mode="VoI_Centralized"):
        self.config = Config()
        self.math_utils = MathUtils.build_mpc_matrices(self.config)
        self.controller = EdgeController(self.config, self.math_utils)
        self.mode = mode  # "VoI_Centralized", "AoI", "VoI_Distributed"
        
        # 初始化 AGV 群 (稍微錯開初始位置，模擬不同產線上的機器人)
        self.agvs = []
        for i in range(self.config.num_agvs):
            start_y = i * 2.0  # 錯開 Y 軸位置
            self.agvs.append(AGV(i, [0.0, start_y, 2.0, 0.0], self.config, self.math_utils))

    def run(self):
        print(f"--- 啟動模擬: {self.mode} 模式 ---")
        print(f"AGV 數量: {self.config.num_agvs}, 可用頻寬數量: {self.config.num_subcarriers}")
        
        total_mse = 0
        
        for step in range(self.config.sim_steps):
            
            # 1. 調度階段 (Resource Scheduling)
            successful_tx_flags = [False] * self.config.num_agvs
            
            if self.mode == "VoI_Centralized":
                allocated_ids = self.controller.centralized_voi_scheduling(self.agvs)
            elif self.mode == "AoI":
                allocated_ids = self.controller.aoi_scheduling(self.agvs)
            elif self.mode == "VoI_Distributed":
                # 分布式 CSMA/CA 模擬: 每回合遞減 T_back，降到 0 的嘗試發送
                allocated_ids = []
                for agv in self.agvs:
                    if agv.T_back > 0:
                        agv.T_back -= 1
                    if agv.T_back <= 0:
                        allocated_ids.append(agv.agv_id)
                # 衝突處理: 如果超過子載波數量，隨機丟包 (模擬碰撞)
                if len(allocated_ids) > self.config.num_subcarriers:
                    random.shuffle(allocated_ids)
                    allocated_ids = allocated_ids[:self.config.num_subcarriers]
            
            # 決定封包是否成功傳送 (考慮 Rayleigh 衰落的簡化機率 P_it)
            for aid in allocated_ids:
                if random.random() < self.config.p_success:
                    successful_tx_flags[aid] = True
            
            # 2. 控制與更新階段
            for i, agv in enumerate(self.agvs):
                # RSU 計算控制指令
                u_t = self.controller.calculate_mpc_control(agv, step)
                
                # RSU 更新其內部的預估狀態 (Eq. 5)
                agv.estimator_update(u_t, successful_tx_flags[i])
                
                # AGV 在物理世界中執行指令並更新狀態 (Eq. 4)
                agv.physical_update(u_t)
                
                # 更新隊列與分散式參數
                agv.queue_update(successful_tx_flags[i])
                if self.mode == "VoI_Distributed":
                    voi = agv.evaluate_long_term_voi()
                    agv.distributed_csma_ca_update(voi, successful_tx_flags[i])

        # 3. 計算統計數據 (MSE 軌跡誤差)
        mse_list = []
        for agv in self.agvs:
            err_x = np.array(agv.history_true_x) - np.array(agv.history_ref_x[1:])
            err_y = np.array(agv.history_true_y) - np.array(agv.history_ref_y[1:])
            mse = np.mean(err_x**2 + err_y**2)
            mse_list.append(mse)
            
        print(f"[{self.mode}] 模擬完成. 平均軌跡誤差 (MSE): {np.mean(mse_list):.4f}")
        return self.agvs, np.mean(mse_list)

# ==========================================
# 6. 繪圖與結果對比
# ==========================================
def plot_results(agvs_voi, agvs_aoi):
    """繪製 VoI 與 AoI 策略的 AGV 軌跡對比圖 (對照論文 Fig. 4)"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 畫出第一台 AGV 的軌跡作為代表
    agv_idx = 0
    
    # 圖 1: VoI 策略
    ax1 = axes[0]
    ax1.plot(agvs_voi[agv_idx].history_ref_x[1:], agvs_voi[agv_idx].history_ref_y[1:], 
             'k--', label='Reference Path (預定產線軌跡)')
    ax1.plot(agvs_voi[agv_idx].history_true_x, agvs_voi[agv_idx].history_true_y, 
             'g-', label='Actual Path (VoI Strategy)')
    ax1.set_title('Factory AGV Trajectory Tracking - VoI Strategy')
    ax1.set_ylabel('Y Position (m)')
    ax1.legend()
    ax1.grid(True)
    
    # 圖 2: AoI 策略
    ax2 = axes[1]
    ax2.plot(agvs_aoi[agv_idx].history_ref_x[1:], agvs_aoi[agv_idx].history_ref_y[1:], 
             'k--', label='Reference Path (預定產線軌跡)')
    ax2.plot(agvs_aoi[agv_idx].history_true_x, agvs_aoi[agv_idx].history_true_y, 
             'r-', label='Actual Path (AoI Strategy)')
    ax2.set_title('Factory AGV Trajectory Tracking - AoI Strategy (Baseline)')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("agv_trajectory_comparison.png")
    print("對比圖表已儲存為 agv_trajectory_comparison.png")

# ==========================================
# 主程式執行入口
# ==========================================
if __name__ == "__main__":
    # 執行基於 VoI 的集中式網路模擬
    sim_voi = Simulator(mode="VoI_Centralized")
    agvs_voi, mse_voi = sim_voi.run()
    
    # 執行基於 VoI 的分布式 WiFi 模擬 (CSMA/CA)
    sim_dist = Simulator(mode="VoI_Distributed")
    _, mse_dist = sim_dist.run()
    
    # 執行基於 AoI (最新鮮資料優先) 的對照組模擬
    sim_aoi = Simulator(mode="AoI")
    agvs_aoi, mse_aoi = sim_aoi.run()
    
    print("\n=== 最終效能評估 (MSE 越低越好) ===")
    print(f"集中式 VoI 調度 MSE: {mse_voi:.4f} (資源分配給最危險/偏離最大的 AGV)")
    print(f"分布式 VoI 調度 MSE: {mse_dist:.4f} (AGV 自行依據 VoI 調整退避時間)")
    print(f"集中式 AoI 調度 MSE: {mse_aoi:.4f} (僅考慮資料新鮮度，不考慮物理危險性)")
    
    # 繪製圖表
    plot_results(agvs_voi, agvs_aoi)