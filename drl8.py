import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    dt: float = 0.05
    sim_steps: int = 1000
    num_agvs: int = 20
    num_subcarriers: int = 1

    u_min: float = -5.0
    u_max: float = 5.0
    v_min: float = -3.0
    v_max: float = 3.0

    radius: float = 12.0
    omega: float = 0.08

    # Receding-horizon/MPC-style tracking controller
    kp: float = 4.5
    kd: float = 3.0
    collision_gain: float = 0.08
    safe_distance: float = 1.5

    # Kalman filter
    process_noise_std: float = 0.01
    measurement_noise_std: float = 0.05

    # Wireless channel
    rsu_position: tuple = (0.0, 0.0)
    tx_power: float = 5.0
    noise_power: float = 0.05
    bandwidth: float = 1.0
    path_loss_exponent: float = 2.0
    min_distance: float = 1.0
    sinr_threshold: float = 0.50

    # Lyapunov-DPP
    virtual_queue_limit: float = 0.10
    V_dpp: float = 4.0
    alpha_aoi: float = 1.0
    beta_voi: float = 1.0

    seed: int = 42


class VehicleModel:
    def __init__(self, cfg):
        dt = cfg.dt
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ], dtype=float)
        self.C = np.eye(4)


class ReferenceTrajectory:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_reference(self, agv_id, step):
        t = step * self.cfg.dt
        phase = 2 * np.pi * agv_id / self.cfg.num_agvs
        theta = self.cfg.omega * t + phase
        return np.array([
            self.cfg.radius * np.cos(theta),
            self.cfg.radius * np.sin(theta),
            -self.cfg.radius * self.cfg.omega * np.sin(theta),
            self.cfg.radius * self.cfg.omega * np.cos(theta)
        ])


class KalmanFilter:
    def __init__(self, model, cfg, initial_state):
        self.A = model.A
        self.B = model.B
        self.C = model.C
        self.Q = np.eye(4) * cfg.process_noise_std**2
        self.R = np.eye(4) * cfg.measurement_noise_std**2
        self.x = initial_state.copy()
        self.P = np.eye(4) * 0.05

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement):
        innovation = measurement - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.C) @ self.P


class AGVAgent:
    def __init__(self, agv_id, cfg, model, reference, rng):
        self.id = agv_id
        self.cfg = cfg
        self.model = model
        self.reference = reference

        ref0 = reference.get_reference(agv_id, 0)
        offset = np.array([
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.15, 0.15),
            rng.uniform(-0.15, 0.15)
        ])
        self.true_state = ref0 + offset
        self.kf = KalmanFilter(model, cfg, self.true_state)

        self.aoi = 0.0
        self.virtual_queue = 0.0

        self.history_state = []
        self.history_ref = []
        self.history_control = []
        self.history_aoi = []
        self.history_voi = []
        self.history_sinr = []
        self.history_tx = []
        self.history_success = []
        self.history_queue = []

    def compute_control(self, step, all_positions):
        ref = self.reference.get_reference(self.id, step)
        estimated = self.kf.x

        pos_error = ref[:2] - estimated[:2]
        vel_error = ref[2:4] - estimated[2:4]

        # Feed-forward acceleration
        t = step * self.cfg.dt
        phase = 2 * np.pi * self.id / self.cfg.num_agvs
        theta = self.cfg.omega * t + phase
        a_ff = np.array([
            -self.cfg.radius * self.cfg.omega**2 * np.cos(theta),
            -self.cfg.radius * self.cfg.omega**2 * np.sin(theta)
        ])

        u = a_ff + self.cfg.kp * pos_error + self.cfg.kd * vel_error

        # Safety repulsion
        p = self.true_state[:2]
        repulsion = np.zeros(2)
        for other_id, other_p in all_positions:
            if other_id == self.id:
                continue
            delta = p - other_p
            d = np.linalg.norm(delta)
            if 1e-6 < d < self.cfg.safe_distance:
                repulsion += (
                    self.cfg.collision_gain *
                    (self.cfg.safe_distance - d) *
                    delta / d
                )

        return np.clip(
            u + repulsion,
            self.cfg.u_min,
            self.cfg.u_max
        )

    def physical_update(self, u, rng):
        noise = rng.normal(0.0, self.cfg.process_noise_std, size=4)
        self.true_state = (
            self.model.A @ self.true_state
            + self.model.B @ u
            + noise
        )
        self.true_state[2:4] = np.clip(
            self.true_state[2:4],
            self.cfg.v_min,
            self.cfg.v_max
        )

    def calculate_voi(self, step, u):
        ref = self.reference.get_reference(self.id, step)
        e_est = self.kf.x - ref
        e_true = self.true_state - ref

        Q = np.diag([5, 5, 1, 1])
        R = np.diag([0.15, 0.15])

        J_est = float(e_est.T @ Q @ e_est + u.T @ R @ u)
        J_true = float(e_true.T @ Q @ e_true + u.T @ R @ u)

        voi = max(0.0, J_est - J_true)
        return float(voi * (1.0 + 0.15 * self.aoi))


def channel_sinr(agent, cfg, rng):
    p = agent.true_state[:2]
    rsu = np.array(cfg.rsu_position)
    d = max(np.linalg.norm(p - rsu), cfg.min_distance)

    fading_power = rng.exponential(1.0)
    received_power = cfg.tx_power * fading_power / (d**cfg.path_loss_exponent)
    return float(received_power / cfg.noise_power)


def packet_success_probability(sinr, cfg):
    x = 5.0 * (sinr - cfg.sinr_threshold)
    x = np.clip(x, -30, 30)
    return float(1.0 / (1.0 + np.exp(-x)))


def choose_agent(agents, strategy, step, controls, cfg, rng):
    if strategy == "Random":
        return int(rng.integers(0, len(agents)))

    if strategy in ["Static-RR", "Static-VoI"]:
        return step % len(agents)

    if strategy == "AoI":
        return int(np.argmax([a.aoi for a in agents]))

    if strategy == "VoI":
        return int(np.argmax([
            a.calculate_voi(step, controls[i])
            for i, a in enumerate(agents)
        ]))

    if strategy in ["Proposed-DPP", "Proposed Lyapunov-DPP"]:
        scores = []
        for i, a in enumerate(agents):
            voi = a.calculate_voi(step, controls[i])
            aoi_norm = a.aoi / max(cfg.dt * 20.0, 1e-6)
            voi_norm = voi / 10.0
            score = (
                cfg.V_dpp * (cfg.alpha_aoi * aoi_norm + cfg.beta_voi * voi_norm)
                - a.virtual_queue
            )
            scores.append(score)
        return int(np.argmax(scores))

    raise ValueError("Unknown strategy: " + strategy)


def run_simulation(strategy, cfg):
    # 固定同一個初始種子以確保兩組策略起點一致
    rng = np.random.default_rng(cfg.seed)

    model = VehicleModel(cfg)
    reference = ReferenceTrajectory(cfg)
    agents = [
        AGVAgent(i, cfg, model, reference, rng)
        for i in range(cfg.num_agvs)
    ]

    for step in range(cfg.sim_steps):
        positions = [(a.id, a.true_state[:2].copy()) for a in agents]
        controls = [a.compute_control(step, positions) for a in agents]

        selected = choose_agent(agents, strategy, step, controls, cfg, rng)

        for i, agent in enumerate(agents):
            agent.physical_update(controls[i], rng)
            agent.kf.predict(controls[i])

        for i, agent in enumerate(agents):
            is_selected = (i == selected)

            if is_selected:
                sinr = channel_sinr(agent, cfg, rng)
                p_success = packet_success_probability(sinr, cfg)
                success = rng.random() < p_success

                if success:
                    measurement = agent.true_state + rng.normal(0.0, cfg.measurement_noise_std, size=4)
                    agent.kf.update(measurement)
                    agent.aoi = 0.0
                else:
                    agent.aoi += cfg.dt
            else:
                sinr = 0.0
                success = False
                agent.aoi += cfg.dt

            action = 1.0 if is_selected else 0.0
            agent.virtual_queue = max(0.0, agent.virtual_queue + action - cfg.virtual_queue_limit)
            voi = agent.calculate_voi(step, controls[i])

            # 記錄歷史資料
            agent.history_state.append(agent.true_state.copy())
            agent.history_ref.append(reference.get_reference(agent.id, step))
            agent.history_control.append(controls[i].copy())
            agent.history_aoi.append(agent.aoi)
            agent.history_voi.append(voi)
            agent.history_sinr.append(sinr)
            agent.history_tx.append(action)
            agent.history_success.append(success)
            agent.history_queue.append(agent.virtual_queue)

    return agents


if __name__ == "__main__":
    cfg = SimulationConfig(sim_steps=1000)

    print("Running Proposed Lyapunov-DPP...")
    agents_dpp = run_simulation("Proposed Lyapunov-DPP", cfg)

    print("Running Static-VoI...")
    agents_static = run_simulation("Static-VoI", cfg)

    # 觀察目標 AGV (ID = 0)
    target_id = 0
    dpp_agv = agents_dpp[target_id]
    static_agv = agents_static[target_id]

    steps = np.arange(cfg.sim_steps)

    # 擷取繪圖所需資料
    ref = np.array(dpp_agv.history_ref)
    state_dpp = np.array(dpp_agv.history_state)
    state_static = np.array(static_agv.history_state)
    u_dpp = np.array(dpp_agv.history_control)
    u_static = np.array(static_agv.history_control)
    tx_dpp = np.array(dpp_agv.history_tx)
    tx_static = np.array(static_agv.history_tx)

    # 建立與參考照片完全對應的 6 個 Subplots
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("AGV Dynamics & Lyapunov DPP Communication Scheduling Analysis", fontsize=13)

    # 1. X Velocity
    axes[0].plot(steps, ref[:, 2], 'k--', label='Ref', alpha=0.7)
    axes[0].plot(steps, state_dpp[:, 2], label='Proposed Lyapunov-DPP', color='#1f77b4', lw=1.2)
    axes[0].plot(steps, state_static[:, 2], label='Static-VoI', color='#7f7f7f', lw=0.9, alpha=0.8)
    axes[0].set_ylabel("X velocity (m/s)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, linestyle='-', alpha=0.6)

    # 2. Y Velocity
    axes[1].plot(steps, ref[:, 3], 'k--', label='Ref', alpha=0.7)
    axes[1].plot(steps, state_dpp[:, 3], label='Proposed Lyapunov-DPP', color='#1f77b4', lw=1.2)
    axes[1].plot(steps, state_static[:, 3], label='Static-VoI', color='#7f7f7f', lw=0.9, alpha=0.8)
    axes[1].set_ylabel("Y velocity (m/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, linestyle='-', alpha=0.6)

    # 3. X Acceleration
    axes[2].plot(steps, u_dpp[:, 0], label='Proposed Lyapunov-DPP', color='#1f77b4', lw=1.2)
    axes[2].plot(steps, u_static[:, 0], label='Static-VoI', color='#7f7f7f', lw=0.9, alpha=0.8)
    axes[2].set_ylabel("X acceleration (m/s²)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, linestyle='-', alpha=0.6)

    # 4. Y Acceleration
    axes[3].plot(steps, u_dpp[:, 1], label='Proposed Lyapunov-DPP', color='#1f77b4', lw=1.2)
    axes[3].plot(steps, u_static[:, 1], label='Static-VoI', color='#7f7f7f', lw=0.9, alpha=0.8)
    axes[3].set_ylabel("Y acceleration (m/s²)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, linestyle='-', alpha=0.6)

    # 5. Proposed Tx Event (長條/脈衝圖)
    axes[4].vlines(steps[tx_dpp > 0], 0, 1.4, colors='#0080ff', lw=1.2)
    axes[4].set_ylabel("Proposed Tx Event")
    axes[4].set_ylim(0, 1.6)
    axes[4].grid(True, linestyle='-', alpha=0.6)

    # 6. Static Tx Event (長條/脈衝圖)
    axes[5].vlines(steps[tx_static > 0], 0, 1.4, colors='#7f7f7f', lw=1.2)
    axes[5].set_ylabel("Static Tx Event")
    axes[5].set_ylim(0, 1.6)
    axes[5].set_xlabel("Time Step")
    axes[5].grid(True, linestyle='-', alpha=0.6)

    plt.tight_layout()
    plt.show()