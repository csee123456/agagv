import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from dataclasses import dataclass

# =============================================================================
# 1. Simulation Configuration
# =============================================================================
@dataclass
class SimulationConfig:
    dt: float = 0.05
    sim_steps: int = 1000  # 配合圖表設定為 1000 steps
    num_agvs: int = 4
    num_subcarriers: int = 2

    # 控制加速度限制
    u_min: float = -5.0
    u_max: float = 5.0

    # 速度限制
    v_min: float = -30.0
    v_max: float = 30.0

    # MPC 參數
    Np: int = 6
    Q_diag: tuple = (10.0, 10.0, 1.0, 1.0)
    R_diag: tuple = (0.05, 0.05)
    terminal_weight: float = 10.0

    # Lyapunov 參數
    P_limit: float = 0.20
    V_lyapunov: float = 10.0
    base_success_probability: float = 0.95

    process_noise_std: float = 0.05
    measurement_noise_std: float = 0.1
    seed: int = 42

# =============================================================================
# 2. Linear Vehicle Model
# =============================================================================
class VehicleModel:
    def __init__(self, cfg: SimulationConfig):
        self.dt = cfg.dt
        dt = cfg.dt
        self.A = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.B = np.array([
            [0.5 * dt ** 2, 0.0],
            [0.0, 0.5 * dt ** 2],
            [dt, 0.0],
            [0.0, dt]
        ])
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]

# =============================================================================
# 3. DARE / LQR Terminal Weight
# =============================================================================
def solve_dare(A, B, Q, R, max_iter=1000, tol=1e-9):
    P = Q.copy()
    for _ in range(max_iter):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_new = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    return P

# =============================================================================
# 4. Reference Trajectory
# =============================================================================
class ReferenceTrajectory:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    def get_reference(self, agv_id, step):
        t = step * self.cfg.dt
        phase = 2.0 * np.pi * agv_id / self.cfg.num_agvs
        radius = 12.0
        omega = 0.08
        px = radius * np.cos(omega * t + phase)
        py = radius * np.sin(omega * t + phase)
        vx = -radius * omega * np.sin(omega * t + phase)
        vy = radius * omega * np.cos(omega * t + phase)
        return np.array([px, py, vx, vy])

# =============================================================================
# 5. AGV Agent
# =============================================================================
class AGVAgent:
    def __init__(self, agv_id, cfg, model, reference):
        self.id = agv_id
        self.cfg = cfg
        self.model = model
        self.reference = reference

        rng = np.random.default_rng(cfg.seed + agv_id)
        initial_ref = reference.get_reference(agv_id, 0)
        offset = np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.2, 0.2)
        ])
        self.true_state = initial_ref + offset
        self.estimated_state = self.true_state.copy()

        self.aoi = 0.0
        self.virtual_queue = 0.0

        # 歷史紀錄
        self.state_history = []
        self.reference_history = []
        self.control_history = []
        self.tx_event_history = []

    def calculate_voi(self):
        ref = self.reference.get_reference(self.id, len(self.state_history))
        error = self.true_state - ref
        return float(np.linalg.norm(error[:2]) ** 2)

    def communication_priority(self):
        voi = self.calculate_voi()
        return 5.0 * voi + 2.0 * self.aoi - 0.5 * self.virtual_queue

    def physical_update(self, u):
        noise = np.random.normal(0.0, self.cfg.process_noise_std, size=self.model.nx)
        self.true_state = self.model.A @ self.true_state + self.model.B @ u + noise
        self.true_state[2:4] = np.clip(self.true_state[2:4], self.cfg.v_min, self.cfg.v_max)

    def estimator_update(self, success, u):
        if success:
            noise = np.random.normal(0.0, self.cfg.measurement_noise_std, size=self.model.nx)
            self.estimated_state = self.true_state + noise
            self.aoi = 0.0
        else:
            self.estimated_state = self.model.A @ self.estimated_state + self.model.B @ u
            self.aoi += self.cfg.dt

# =============================================================================
# 6. Constrained MPC Solver
# =============================================================================
def solve_mpc(x0, agv_id, step, cfg, model, ref_gen, P_term):
    x = cp.Variable((model.nx, cfg.Np + 1))
    u = cp.Variable((model.nu, cfg.Np))

    Q = np.diag(cfg.Q_diag)
    R = np.diag(cfg.R_diag)

    cost = 0
    constraints = [x[:, 0] == x0]

    for k in range(cfg.Np):
        ref_k = ref_gen.get_reference(agv_id, step + k)
        cost += cp.quad_form(x[:, k] - ref_k, Q) + cp.quad_form(u[:, k], R)
        constraints += [
            x[:, k + 1] == model.A @ x[:, k] + model.B @ u[:, k],
            u[:, k] >= cfg.u_min,
            u[:, k] <= cfg.u_max,
            x[2:4, k + 1] >= cfg.v_min,
            x[2:4, k + 1] <= cfg.v_max
        ]

    ref_terminal = ref_gen.get_reference(agv_id, step + cfg.Np)
    cost += cp.quad_form(x[:, cfg.Np] - ref_terminal, P_term)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and u.value is not None:
        return u.value[:, 0]
    return np.zeros(model.nu)

# =============================================================================
# 7. Simulation Runner
# =============================================================================
def run_simulation(strategy_type="drl"):
    cfg = SimulationConfig()
    model = VehicleModel(cfg)
    ref_gen = ReferenceTrajectory(cfg)

    Q_term = np.diag(cfg.Q_diag) * cfg.terminal_weight
    R_mat = np.diag(cfg.R_diag)
    P_term = solve_dare(model.A, model.B, Q_term, R_mat)

    agvs = [AGVAgent(i, cfg, model, ref_gen) for i in range(cfg.num_agvs)]

    for step in range(cfg.sim_steps):
        if strategy_type == "drl":
            priorities = [agv.communication_priority() for agv in agvs]
            granted_indices = np.argsort(priorities)[-cfg.num_subcarriers:]
        else:  # Static-VoI
            granted_indices = [i for i in range(cfg.num_agvs) if (step + i * 7) % 25 == 0]

        for i, agv in enumerate(agvs):
            is_granted = (i in granted_indices)
            comm_success = is_granted and (np.random.rand() < cfg.base_success_probability)

            u_opt = solve_mpc(agv.estimated_state, agv.id, step, cfg, model, ref_gen, P_term)
            agv.physical_update(u_opt)
            agv.estimator_update(comm_success, u_opt)

            agv.state_history.append(agv.true_state.copy())
            agv.reference_history.append(ref_gen.get_reference(agv.id, step))
            agv.control_history.append(u_opt)
            agv.tx_event_history.append(1.0 if comm_success else 0.0)

            tx_action = 1.0 if is_granted else 0.0
            agv.virtual_queue = max(0.0, agv.virtual_queue + tx_action - cfg.P_limit)

    return agvs

# =============================================================================
# 8. Main & 6-Subplot Visualization
# =============================================================================
if __name__ == "__main__":
    print("正在執行 DRL-VoI 模擬...")
    agvs_drl = run_simulation(strategy_type="drl")

    print("正在執行 Static-VoI 模擬...")
    agvs_static = run_simulation(strategy_type="static")

    # 針對 AGV 0 繪製兩者對比分析圖
    target_id = 0
    drl = agvs_drl[target_id]
    static = agvs_static[target_id]

    drl_states = np.array(drl.state_history)
    static_states = np.array(static.state_history)
    refs = np.array(drl.reference_history)

    drl_ctrl = np.array(drl.control_history)
    static_ctrl = np.array(static.control_history)

    drl_tx = np.array(drl.tx_event_history)
    static_tx = np.array(static.tx_event_history)

    steps = np.arange(len(drl_states))

    # 建立 6 個子圖 (縱向排列)
    fig, axs = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Vehicle Dynamics and Communication Event Analysis", fontsize=14, y=0.98)

    # 1. X Velocity
    axs[0].plot(steps, refs[:, 2], 'k--', alpha=0.7, label='Ref')
    axs[0].plot(steps, drl_states[:, 2], 'tab:blue', label='DRL-VoI')
    axs[0].plot(steps, static_states[:, 2], 'tab:red', alpha=0.8, label='Static-VoI')
    axs[0].set_ylabel("X velocity (m/s)")
    axs[0].grid(True, linestyle='-', alpha=0.6)
    axs[0].legend(loc='upper right')

    # 2. Y Velocity
    axs[1].plot(steps, refs[:, 3], 'k--', alpha=0.7, label='Ref')
    axs[1].plot(steps, drl_states[:, 3], 'tab:blue', label='DRL-VoI')
    axs[1].plot(steps, static_states[:, 3], 'tab:red', alpha=0.8, label='Static-VoI')
    axs[1].set_ylabel("Y velocity (m/s)")
    axs[1].grid(True, linestyle='-', alpha=0.6)
    axs[1].legend(loc='upper right')

    # 3. X Acceleration
    axs[2].plot(steps, drl_ctrl[:, 0], 'tab:blue', label='DRL-VoI')
    axs[2].plot(steps, static_ctrl[:, 0], 'tab:brown', label='Static-VoI')
    axs[2].set_ylabel("X acceleration (m/s²)")
    axs[2].grid(True, linestyle='-', alpha=0.6)
    axs[2].legend(loc='upper right')

    # 4. Y Acceleration
    axs[3].plot(steps, drl_ctrl[:, 1], 'tab:blue', label='DRL-VoI')
    axs[3].plot(steps, static_ctrl[:, 1], 'tab:brown', label='Static-VoI')
    axs[3].set_ylabel("Y acceleration (m/s²)")
    axs[3].grid(True, linestyle='-', alpha=0.6)
    axs[3].legend(loc='upper right')

    # 5. DRL Tx Event
    tx_indices_drl = np.where(drl_tx > 0.5)[0]
    axs[4].vlines(tx_indices_drl, ymin=0.5, ymax=1.4, colors='tab:blue', alpha=0.8)
    axs[4].set_ylim(0.5, 1.5)
    axs[4].set_ylabel("DRL Tx Event")
    axs[4].grid(True, linestyle='-', alpha=0.6)

    # 6. Static Tx Event
    tx_indices_static = np.where(static_tx > 0.5)[0]
    axs[5].vlines(tx_indices_static, ymin=0.5, ymax=1.4, colors='tab:orange', alpha=0.8)
    axs[5].set_ylim(0.5, 1.5)
    axs[5].set_ylabel("Static Tx Event")
    axs[5].set_xlabel("Time Step")
    axs[5].grid(True, linestyle='-', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()