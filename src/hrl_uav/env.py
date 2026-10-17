from collections import deque
from dataclasses import dataclass, field
from numbers import Real

import numpy as np


TRAFFIC_PROFILES = {
    "easy": {
        "hot_high": 0.27,
        "hot_low": 0.63,
        "cold_high": 0.09,
        "cold_low": 0.21,
    },
    "normal": {
        "hot_high": 0.36,
        "hot_low": 0.84,
        "cold_high": 0.12,
        "cold_low": 0.28,
    },
    "hard": {
        "hot_high": 0.45,
        "hot_low": 1.05,
        "cold_high": 0.15,
        "cold_low": 0.35,
    },
}


DEVICE_PROFILES = {
    "sensor": {
        "high_rate_scale": 0.55,
        "low_rate_scale": 1.10,
        "high_size_mbits": 0.35,
        "low_size_mbits": 0.90,
    },
    "controller": {
        "high_rate_scale": 2.20,
        "low_rate_scale": 0.25,
        "high_size_mbits": 0.55,
        "low_size_mbits": 0.35,
    },
    "vision": {
        "high_rate_scale": 0.15,
        "low_rate_scale": 0.80,
        "high_size_mbits": 1.20,
        "low_size_mbits": 1.40,
    },
}


ZONE_DEVICE_MIXES = {
    0: np.array([0.35, 0.30, 0.35], dtype=np.float32),
    1: np.array([0.60, 0.30, 0.10], dtype=np.float32),
    2: np.array([0.45, 0.15, 0.40], dtype=np.float32),
}


ZONE_TARGET_LOAD_MBPS = {
    # Match the old path-planning workload split:
    # 15 users/region with Bernoulli rates [0.8, 0.55, 0.3]
    # and mean task size 0.85 Mbits.
    0: 10.2,
    1: 7.0,
    2: 3.8,
}


@dataclass
class ParentTask:
    task_id: int
    user_id: int
    source_uav: int
    task_type: str
    priority: str
    birth_time: float
    size_mbits: float
    cpu_cycles: float
    remaining_upload_mbits: float = 0.0
    upload_finish_time: float = None
    total_chunks: int = 0
    completed_chunks: int = 0
    finish_time: float = None
    chunk_finish_times: list = field(default_factory=list)


@dataclass
class TaskChunk:
    parent_task_id: int
    chunk_id: int
    birth_time: float
    priority: str
    task_type: str
    remaining_mbits: float
    initial_mbits: float
    source_uav: int
    target_uav: int
    current_stage: str


class UAV_MEC_Env:
    """
    基于分层强化学习(HRL)的双时间尺度 UAV-MEC 系统环境 (严谨科研重构版)
    
    核心升级:
    - 真实独立地面用户(M=40)的随机游走与流量加权质心观测。
    - 异构空间流量：热点管辖区(高流量) vs 边缘管辖区(低流量)。
    - 硬件同构化：所有 UAV 均为重型计算节点 (16GHz)。
    - 双重优先级队列(URLLC/eMBB)：底层的物理清空严格遵循 100% 抢占式调度。
    """
    
    def __init__(
        self,
        num_uavs=3,
        num_users=39,
        H=50.0,
        C=1000,
        kappa=1e-28,
        p_opt=0.5,
        p_gt_power=0.1,
        p_u2u_power=1.0,
        beta_0=1e-6,
        B=1e6,
        noise_power=1e-13,
        record_trajectory=False,
        traffic_profile="normal",
        traffic_scale=1.0,
        arrival_dt=0.5,
        resource_frame_dt=0.5,
        deployment_dt=1.0,
        deployment_transient_energy_mode="none",
        uav_mass_kg=20.0 / 9.81,
        uplink_bandwidth_mhz=20.0,
        u2u_bandwidth_mhz=3.0,
        cpu_alloc_mode="priority_kkt",
        kkt_v=1.0,
        cpu_energy_prices=None,
        fixed_cpu_util=0.5,
        cpu_power_cap_w=0.0,
        load_scale=1.0,
        zone_loads=None,
        uav_initial_positions=None,
        pause_service_during_flight=True,
        arrivals_during_deployment=False,
        users_static=False,
        gm_mean_speed_mps=0.5,
        gm_mean_heading_rad=np.pi / 4.0,
        gm_speed_memory=0.5,
        gm_heading_memory=0.5,
        gm_speed_noise_std=0.1,
        gm_heading_noise_std=0.1,
        user_zone_radius=80.0,
        user_zone_scatter_std=40.0,
        min_uav_separation_m=5.0,
        u2u_link_bandwidth_matrix_mhz=None,
        gm_heading_mode="shared_mean",
        rpgm_group_speed_mps=0.35,
        rpgm_group_speed_memory=0.9,
        rpgm_group_speed_noise_std=0.04,
        rpgm_group_turn_std_rad_sqrt_s=0.12,
        rpgm_local_velocity_memory=0.9,
        rpgm_local_velocity_std_mps=0.18,
        rpgm_local_restore_rate_s2=0.0002,
        rpgm_local_speed_cap_mps=0.45,
        compute_during_deployment=False,
        wireless_model="legacy_fixed_noise",
        noise_psd_w_per_hz=1.95e-19,
        u2u_tx_budget_w=1.0,
        u2u_reference_bandwidth_hz=512820.5128205128,
        u2u_reference_power_w=1.0,
        motion_protocol="legacy_fly_hover",
        mission_routes_m=None,
        mission_duration_s=900.0,
        cluster_count=1,
        cluster_load_mode="exact_share",
    ):

        # ================== 1. 系统物理设定与参数 ==================
        self.macro_dt = 10.0   # 宏观周期 10 秒
        self.micro_dt = 0.1    # 物理积分步长 0.1 秒
        self.micro_steps_per_macro = int(self.macro_dt / self.micro_dt)
        if motion_protocol not in ("legacy_fly_hover", "mission_rpgm_v1"):
            raise ValueError("Unknown motion_protocol")
        self.motion_protocol = motion_protocol
        self.continuous_mission = motion_protocol == "mission_rpgm_v1"
        if self.continuous_mission:
            routes = np.asarray(mission_routes_m, dtype=np.float64)
            if (num_uavs != 3 or routes.shape != (3, 2, 2)
                    or not np.isfinite(routes).all()
                    or np.any((routes < 0) | (routes > 1000))):
                raise ValueError("mission_routes_m must contain three finite in-map start/end pairs")
            if not np.array_equal(routes[:, 0], [[350, 350], [350, 650], [650, 500]]):
                raise ValueError("mission route starts must match the fixed source anchors")
            self.mission_routes_m = routes.copy()
            self.mission_duration_s = self._positive_wireless_parameter(
                "mission_duration_s", mission_duration_s)
            if (deployment_dt != 0 or pause_service_during_flight is not False
                    or arrivals_during_deployment is not True
                    or deployment_transient_energy_mode != "none"
                    or gm_heading_mode != "reference_point_group" or users_static
                    or resource_frame_dt != 0.5 or arrival_dt != 0.5):
                raise ValueError("mission_rpgm_v1 requires zero deployment, continuous radio, "
                                 "arrivals, RPGM, 0.5 s frames and no deployment transient")
        elif mission_routes_m is not None:
            raise ValueError("mission_routes_m requires mission_rpgm_v1")
        self.resource_frame_dt = float(resource_frame_dt)
        self.deployment_dt = float(deployment_dt)
        self.deployment_transient_energy_mode = str(
            deployment_transient_energy_mode
        ).lower()
        if self.deployment_transient_energy_mode not in (
            "none",
            "kinetic_lower_bound",
        ):
            raise ValueError(
                "deployment_transient_energy_mode must be 'none' or "
                "'kinetic_lower_bound'."
            )
        self.uav_mass_kg = float(uav_mass_kg)
        if self.uav_mass_kg <= 0.0:
            raise ValueError("uav_mass_kg must be positive.")
        if (self.resource_frame_dt <= 0.0
                or (self.deployment_dt <= 0.0 and not self.continuous_mission)):
            raise ValueError("resource_frame_dt and deployment_dt must be positive.")
        self.resource_frame_steps = int(
            round(self.resource_frame_dt / self.micro_dt)
        )
        self.deployment_steps = int(round(self.deployment_dt / self.micro_dt))
        if not np.isclose(
            self.resource_frame_steps * self.micro_dt,
            self.resource_frame_dt,
        ):
            raise ValueError("resource_frame_dt must be an integer multiple of micro_dt.")
        if not np.isclose(
            self.deployment_steps * self.micro_dt,
            self.deployment_dt,
        ):
            raise ValueError("deployment_dt must be an integer multiple of micro_dt.")
        if self.deployment_steps % self.resource_frame_steps != 0:
            raise ValueError("deployment_dt must contain whole resource frames.")
        service_steps = self.micro_steps_per_macro - self.deployment_steps
        if service_steps <= 0 or service_steps % self.resource_frame_steps != 0:
            raise ValueError("The service phase must contain whole resource frames.")
        self.deployment_resource_frames = (
            self.deployment_steps // self.resource_frame_steps
        )
        self.service_resource_frames = service_steps // self.resource_frame_steps
        self.resource_frames_per_macro = (
            self.deployment_resource_frames + self.service_resource_frames
        )
        
        self.N = num_uavs      # 无人机数量 (默认 3)
        self.M = num_users     # 真实地面用户总数量 (默认 39)
        if self.N <= 0 or self.M <= 0:
            raise ValueError("num_uavs and num_users must be positive.")
        self.H = H             # 无人机飞行高度
        self.C = C             # CPU 周期数/bit
        self.kappa = kappa     # 能耗系数
        self.p_opt = p_opt     # 固定发射功率
        self.B = B             # 信道带宽
        self.noise_power = noise_power
        # 参考信道增益 β₀ = -60 dB = 1e-6 (TCOM 2024, Xu et al., Table II)
        self.beta_0 = float(beta_0)
        self.B_gt_total = float(uplink_bandwidth_mhz) * 1e6
        self.B_u2u_total = float(u2u_bandwidth_mhz) * 1e6
        # GT 上行 0.1 W、UAV 机间 1 W（论文功率非对称：机间链路更可靠）
        self.p_gt = float(p_gt_power)
        self.p_u2u = float(p_u2u_power)
        if not isinstance(wireless_model, str) or wireless_model not in (
                "legacy_fixed_noise", "psd_per_uav_budget"):
            raise ValueError(f"Unknown wireless_model: {wireless_model!r}")
        self.wireless_model = wireless_model
        self.noise_psd_w_per_hz = self._positive_wireless_parameter(
            "noise_psd_w_per_hz", noise_psd_w_per_hz)
        self.u2u_tx_budget_w = self._positive_wireless_parameter(
            "u2u_tx_budget_w", u2u_tx_budget_w)
        self.u2u_reference_bandwidth_hz = self._positive_wireless_parameter(
            "u2u_reference_bandwidth_hz", u2u_reference_bandwidth_hz)
        self.u2u_reference_power_w = self._positive_wireless_parameter(
            "u2u_reference_power_w", u2u_reference_power_w)
        if self.wireless_model == "psd_per_uav_budget":
            if u2u_link_bandwidth_matrix_mhz is not None:
                raise ValueError(
                    "psd_per_uav_budget does not support "
                    "u2u_link_bandwidth_matrix_mhz; use equal active-link sharing.")
            for name, value in (("uplink_bandwidth_mhz", uplink_bandwidth_mhz),
                                ("u2u_bandwidth_mhz", u2u_bandwidth_mhz),
                                ("p_gt_power", p_gt_power), ("beta_0", beta_0)):
                self._positive_wireless_parameter(name, value)
            for name, value in (("uplink bandwidth in Hz", self.B_gt_total),
                                ("U2U bandwidth in Hz", self.B_u2u_total),
                                ("reference noise power", self.noise_psd_w_per_hz * self.u2u_reference_bandwidth_hz),
                                ("uplink noise power", self.noise_psd_w_per_hz * self.B_gt_total / self.M),
                                ("U2U noise power", self.noise_psd_w_per_hz * self.B_u2u_total)):
                self._positive_wireless_parameter(name, value)
        self.min_uav_separation_m = float(min_uav_separation_m)
        if self.min_uav_separation_m < 0.0:
            raise ValueError("min_uav_separation_m must be non-negative.")
        self.dynamic_u2u_bandwidth = u2u_link_bandwidth_matrix_mhz is None
        self.u2u_link_bandwidth_matrix_hz = (
            self._normalize_u2u_link_bandwidth_matrix(
                u2u_link_bandwidth_matrix_mhz
            )
        )
        self.last_u2u_link_bandwidth_matrix_hz = (
            self.u2u_link_bandwidth_matrix_hz.copy()
        )
        if self.wireless_model == "psd_per_uav_budget":
            self.last_u2u_link_bandwidth_matrix_hz.fill(0.0)
        self.communication_model_reference = (
            "Xu et al., IEEE TCOM 2024, DOI 10.1109/TCOMM.2024.3361536"
        )
        self.u2g_association_policy = "fixed_region_uav"
        self.u2g_bandwidth_policy = "orthogonal_equal_per_ground_terminal"
        self.u2u_bandwidth_policy = (
            "work_conserving_equal_active_directed_links"
            if self.dynamic_u2u_bandwidth
            else "fixed_custom_directed_links"
        )
        self.traffic_profile = str(traffic_profile).lower()
        self.arrival_dt = max(float(arrival_dt), self.resource_frame_dt)
        # Legacy experiments pause both arrivals and service during deployment.
        # The long-term queueing protocol can instead keep exogenous arrivals
        # active while U2G/U2U/CPU service remains paused.
        self.pause_service_during_flight = bool(pause_service_during_flight)
        if not isinstance(compute_during_deployment, (bool, np.bool_)):
            raise ValueError("compute_during_deployment must be a boolean.")
        self.compute_during_deployment = bool(compute_during_deployment)
        if (self.compute_during_deployment and not self.pause_service_during_flight
                and not self.continuous_mission):
            raise ValueError(
                "compute_during_deployment requires pause_service_during_flight=True "
                "so deployment communication remains paused."
            )
        self.arrivals_during_deployment = bool(arrivals_during_deployment)
        self.arrival_interval_steps = max(
            self.resource_frame_steps,
            int(round(self.arrival_dt / self.micro_dt)),
        )
        self.arrival_dt = self.arrival_interval_steps * float(self.micro_dt)
        if self.arrival_interval_steps % self.resource_frame_steps != 0:
            raise ValueError("arrival_dt must be an integer multiple of resource_frame_dt.")
        if self.traffic_profile not in TRAFFIC_PROFILES:
            raise ValueError(
                f"Unknown traffic_profile '{traffic_profile}'. "
                f"Available: {sorted(TRAFFIC_PROFILES)}"
            )
        profile = TRAFFIC_PROFILES[self.traffic_profile]
        self.traffic_scale = float(traffic_scale)
        self.lambda_hot_high = profile["hot_high"] * self.traffic_scale
        self.lambda_hot_low = profile["hot_low"] * self.traffic_scale
        self.lambda_cold_high = profile["cold_high"] * self.traffic_scale
        self.lambda_cold_low = profile["cold_low"] * self.traffic_scale
        self.device_type_names = tuple(DEVICE_PROFILES.keys())
        self.user_device_type_ids = np.zeros(self.M, dtype=np.int32)
        self.user_lambda_high = np.zeros(self.M, dtype=np.float32)
        self.user_lambda_low = np.zeros(self.M, dtype=np.float32)
        self.user_task_size_high_mbits = np.zeros(self.M, dtype=np.float32)
        self.user_task_size_low_mbits = np.zeros(self.M, dtype=np.float32)
        self.zone_expected_offered_load_mbps = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.cluster_count = int(cluster_count)
        if self.cluster_count < 1:
            raise ValueError("cluster_count must be a positive integer.")
        if cluster_load_mode not in (
            "exact_share", "preserve_type_priority", "preserve_type_priority_capped",
        ):
            raise ValueError("Unknown cluster_load_mode")
        self.cluster_load_mode = cluster_load_mode
        self.user_cluster_ids = np.zeros(self.M, dtype=np.int32)
        self.cluster_target_shares = np.full(
            (self.N, self.cluster_count),
            1.0 / self.cluster_count,
            dtype=np.float64,
        )
        self.cluster_expected_offered_load_mbps = np.zeros(
            (self.N, self.cluster_count),
            dtype=np.float32,
        )
        
        # 同构硬件设定 (统一提供强大的计算能力)
        # 统一预设为 16GHz (重型)
        self.F_max_array = np.full(self.N, 16e9)
        
        # 频率异构化闭式解推导权重参数 (保留做物理核心分配逻辑)
        self.w_D_high = 10.0   
        self.w_D_low = 1.0     
        self.w_E = 1.0         
        self.qos_deadline_s = {
            "high": 2.0,
            "low": 5.0,
        }
        self.cpu_alloc_mode = str(cpu_alloc_mode).lower()
        self.kkt_v = float(kkt_v)
        if cpu_energy_prices is None:
            cpu_energy_prices = self.kkt_v
        cpu_energy_prices = np.asarray(cpu_energy_prices, dtype=np.float64)
        if cpu_energy_prices.ndim == 0:
            cpu_energy_prices = np.full(self.N, float(cpu_energy_prices))
        elif cpu_energy_prices.shape == (1,):
            cpu_energy_prices = np.full(self.N, float(cpu_energy_prices[0]))
        if cpu_energy_prices.shape != (self.N,):
            raise ValueError(
                "cpu_energy_prices must be a scalar or have shape "
                f"({self.N},), got {cpu_energy_prices.shape}."
            )
        if not np.all(np.isfinite(cpu_energy_prices)) or np.any(
            cpu_energy_prices < 0.0
        ):
            raise ValueError("cpu_energy_prices must be finite and non-negative.")
        self.cpu_energy_prices = cpu_energy_prices.copy()
        self.kkt_bisection_iters = 24
        self.fixed_cpu_util = float(fixed_cpu_util)
        self.cpu_power_cap_w = max(float(cpu_power_cap_w), 0.0)
        self.last_cpu_power_uncapped_w = 0.0
        self.last_cpu_power_allocated_w = 0.0
        self.last_cpu_power_cap_active = False
        self._external_cpu_frequency_override = None
        self._external_cpu_override_steps_remaining = 0
        self.load_scale = float(load_scale)
        self.zone_target_load_mbps = self._normalize_zone_loads(zone_loads)
        self.valid_cpu_alloc_modes = {
            "legacy_kkt",
            "priority_kkt",
            "fixed_full",
            "fixed_average",
            "fixed_demand",
            "priority_demand",
            "priced_maxweight",
        }
        if self.cpu_alloc_mode not in self.valid_cpu_alloc_modes:
            raise ValueError(
                f"Unknown cpu_alloc_mode '{cpu_alloc_mode}'. "
                f"Available: {sorted(self.valid_cpu_alloc_modes)}"
            )
        
        # ================== 2. 状态结构初始化 ==================
        # 锚点管辖区划分 (每台 UAV 负责一个群组中心)
        self.base_anchor_points = np.array([
            [350.0, 350.0],
            [350.0, 650.0],
            [650.0, 500.0]
        ])
        if self.N > 3:
            extra = np.random.uniform(200, 800, (self.N - 3, 2))
            self.base_anchor_points = np.vstack([self.base_anchor_points, extra])
        self.anchor_points = self.base_anchor_points.copy()

        # 无人机初始部署位置：直接部署在各自管辖区锚点
        # 这样 RL 学习的是"在服务区内如何微调站位"（兼顾上行 vs U2U），
        # 而不是"如何从远处飞回服务区"。
        if uav_initial_positions is None:
            self.uav_initial_positions = self.base_anchor_points.copy()
        else:
            initial_positions = np.asarray(
                uav_initial_positions,
                dtype=np.float64,
            )
            if initial_positions.shape != (self.N, 2):
                raise ValueError(
                    "uav_initial_positions must have shape "
                    f"({self.N}, 2), got {initial_positions.shape}."
                )
            if not np.all(np.isfinite(initial_positions)):
                raise ValueError("uav_initial_positions must be finite.")
            if np.any(initial_positions < 0.0) or np.any(initial_positions > 1000.0):
                raise ValueError("uav_initial_positions must lie in [0, 1000] m.")
            self.uav_initial_positions = initial_positions.copy()
        if self.N > 1 and self.min_uav_separation_m > 0.0:
            initial_diff = (
                self.uav_initial_positions[:, None, :]
                - self.uav_initial_positions[None, :, :]
            )
            initial_distances = np.linalg.norm(initial_diff, axis=-1)
            initial_distances[np.eye(self.N, dtype=bool)] = np.inf
            if float(np.min(initial_distances)) < (
                self.min_uav_separation_m - 1e-9
            ):
                raise ValueError(
                    "uav_initial_positions violate min_uav_separation_m."
                )
        if self.N > 3:
            # 超过 3 台时，额外的 UAV 初始位置已由 base_anchor_points 提供
            pass

        # 为每个用户预先划定归属管辖区 (确保总是精确分配到 M 个用户，兼容无法整除)
        self.user_allocations = np.repeat(np.arange(self.N), int(np.ceil(self.M / self.N)))[:self.M]
        self.user_coords = np.zeros((self.M, 2))  # 用户的真实独立坐标
        self.user_offsets = np.zeros((self.M, 2)) # 用户相对于锚点的内部偏移
        
        # 流量加权质心 (给每架飞机的观测值)
        self.weighted_centroids = np.zeros((self.N, 2))
        self.instantaneous_centroids = np.zeros((self.N, 2))
        # Preserve the historical 0.1 s EMA time constant after moving
        # exogenous updates to 0.5 s resource-frame boundaries.
        self.centroid_ema_beta = 0.99 ** self.resource_frame_steps

        self.uav_coords = np.zeros((self.N, 2))
        
        # Ground-terminal Gauss-Markov mobility. The speed and heading are
        # updated once per 0.5 s resource frame; coordinates are integrated at
        # the 0.1 s physical step.
        self.gm_mean_speed_mps = float(gm_mean_speed_mps)
        self.gm_mean_heading_rad = float(gm_mean_heading_rad)
        self.gm_heading_mode = str(gm_heading_mode).lower()
        if self.gm_heading_mode not in {
            "shared_mean",
            "independent_episode_mean",
            "reference_point_group",
        }:
            raise ValueError(
                "gm_heading_mode must be 'shared_mean', "
                "'independent_episode_mean', or "
                "'reference_point_group'."
            )
        self.gm_speed_memory = float(gm_speed_memory)
        self.gm_heading_memory = float(gm_heading_memory)
        self.gm_speed_noise_std = max(float(gm_speed_noise_std), 0.0)
        self.gm_heading_noise_std = max(float(gm_heading_noise_std), 0.0)
        for name, value in (
            ("gm_speed_memory", self.gm_speed_memory),
            ("gm_heading_memory", self.gm_heading_memory),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        self.alpha_gmmm = self.gm_speed_memory
        self.max_infantry_speed = 3.0
        self.user_speeds = np.full(self.M, self.gm_mean_speed_mps)
        self.user_mean_headings = np.full(self.M, self.gm_mean_heading_rad)
        self.user_headings = np.full(self.M, self.gm_mean_heading_rad)
        self.user_velocities = np.zeros((self.M, 2))
        self.rpgm_group_speed_mps = max(float(rpgm_group_speed_mps), 0.0)
        self.rpgm_group_speed_memory = float(rpgm_group_speed_memory)
        self.rpgm_group_speed_noise_std = max(
            float(rpgm_group_speed_noise_std), 0.0
        )
        self.rpgm_group_turn_std_rad_sqrt_s = max(
            float(rpgm_group_turn_std_rad_sqrt_s), 0.0
        )
        self.rpgm_local_velocity_memory = float(
            rpgm_local_velocity_memory
        )
        self.rpgm_local_velocity_std_mps = max(
            float(rpgm_local_velocity_std_mps), 0.0
        )
        self.rpgm_local_restore_rate_s2 = max(
            float(rpgm_local_restore_rate_s2), 0.0
        )
        self.rpgm_local_speed_cap_mps = max(
            float(rpgm_local_speed_cap_mps), 0.0
        )
        for name, value in (
            ("rpgm_group_speed_memory", self.rpgm_group_speed_memory),
            ("rpgm_local_velocity_memory", self.rpgm_local_velocity_memory),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        self.mobility_group_centers = self.base_anchor_points.copy()
        self.mobility_group_speeds = np.full(
            self.N, self.rpgm_group_speed_mps, dtype=np.float64
        )
        self.mobility_group_headings = np.zeros(self.N, dtype=np.float64)
        self.mobility_group_velocities = np.zeros((self.N, 2), dtype=np.float64)
        self.user_local_velocities = np.zeros((self.M, 2), dtype=np.float64)
        self.users_static = bool(users_static)
        self.user_zone_radius = float(user_zone_radius)
        self.user_zone_scatter_std = float(user_zone_scatter_std)
        # 用于宏观-微观时隙解耦的缓存向量
        self.current_macro_v = np.zeros(self.N)
        self.current_macro_theta = np.zeros(self.N)
        self.macro_fly_steps_remaining = 0
        self.macro_service_pause_steps_remaining = 0
        
        # 【双轨优先级流体队列】
        # _high: 存放 30% 到达的高优先紧急数据
        # _low:  存放 70% 到达的低优先宽带数据
        self.Q_loc_high = np.zeros(self.N)
        self.Q_loc_low = np.zeros(self.N)
        
        self.Q_tx_high = np.zeros((self.N, self.N))
        self.Q_tx_low = np.zeros((self.N, self.N))
        self.pending_upload_jobs = []
        self.global_micro_step = 0
        self._resource_frame_prepared = False
        self.resource_frame_prepare_count = 0
        self.deployment_arrival_prepare_count = 0
        self.mobility_update_count = 0
        self._u2u_spectral_cache_coords = None
        self._u2u_spectral_cache_result = None
        self.current_arrival_active_users = np.zeros(self.N)
        
        self.A_t_high = np.zeros(self.N)
        self.A_t_low = np.zeros(self.N)
        self._init_task_tracker()
        
        # 3. UAV 部署在区域外围的三角形初始位置（以 (500,500) 为中心）
        self.uav_coords = self.uav_initial_positions.copy()
        self.current_user_associations = self.user_allocations.copy()

        self.record_trajectory = record_trajectory
        if self.record_trajectory:
            self.uav_trajectory = []
            self.centroid_trajectory = []
            self.users_trajectory = []
            self.group_center_trajectory = []
            self.arrival_high_trajectory = []
            self.arrival_low_trajectory = []

    def record_trajectory_snapshot(self):
        if not self.record_trajectory:
            return
        self.uav_trajectory.append(self.uav_coords.copy())
        self.centroid_trajectory.append(self.weighted_centroids.copy())
        self.users_trajectory.append(self.user_coords.copy())
        self.group_center_trajectory.append(
            self.mobility_group_centers.copy()
        )

    def _current_time(self):
        return float(self.global_micro_step) * float(self.micro_dt)

    @staticmethod
    def _positive_wireless_parameter(name, value):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a finite positive numeric scalar.")
        return float(value)

    def _normalize_u2u_link_bandwidth_matrix(self, matrix_mhz):
        if matrix_mhz is None:
            matrix_hz = np.zeros((self.N, self.N), dtype=np.float64)
            directed_links = self.N * max(self.N - 1, 0)
            if directed_links > 0:
                matrix_hz[~np.eye(self.N, dtype=bool)] = (
                    self.B_u2u_total / directed_links
                )
            return matrix_hz

        matrix_hz = np.asarray(matrix_mhz, dtype=np.float64) * 1e6
        if matrix_hz.shape != (self.N, self.N):
            raise ValueError(
                "u2u_link_bandwidth_matrix_mhz must have shape "
                f"({self.N}, {self.N})."
            )
        if np.any(~np.isfinite(matrix_hz)) or np.any(matrix_hz < 0.0):
            raise ValueError(
                "u2u_link_bandwidth_matrix_mhz must be finite and non-negative."
            )
        matrix_hz = matrix_hz.copy()
        np.fill_diagonal(matrix_hz, 0.0)
        if float(np.sum(matrix_hz)) > self.B_u2u_total + 1e-6:
            raise ValueError(
                "The fixed U2U link bandwidths exceed the total U2U bandwidth."
            )
        return matrix_hz

    def get_nearest_uav_associations(self, user_coords=None, uav_coords=None):
        users = np.asarray(
            self.user_coords if user_coords is None else user_coords,
            dtype=np.float64,
        )
        uavs = np.asarray(
            self.uav_coords if uav_coords is None else uav_coords,
            dtype=np.float64,
        )
        if users.ndim != 2 or users.shape[1] != 2:
            raise ValueError("user_coords must have shape (K, 2).")
        if uavs.shape != (self.N, 2):
            raise ValueError(f"uav_coords must have shape ({self.N}, 2).")
        horizontal_sq = np.sum(
            (users[:, None, :] - uavs[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(horizontal_sq, axis=1).astype(np.int64)

    def compute_u2g_rate_bps(
        self,
        user_coords=None,
        uav_coords=None,
        associations=None,
    ):
        users = np.asarray(
            self.user_coords if user_coords is None else user_coords,
            dtype=np.float64,
        )
        uavs = np.asarray(
            self.uav_coords if uav_coords is None else uav_coords,
            dtype=np.float64,
        )
        if associations is None:
            if users.shape[0] != self.M:
                raise ValueError(
                    "associations are required when computing rates for a "
                    "ground-terminal subset."
                )
            associations = self.user_allocations.copy()
        associations = np.asarray(associations, dtype=np.int64)
        if associations.shape != (users.shape[0],):
            raise ValueError("associations must contain one UAV index per GT.")
        if np.any(associations < 0) or np.any(associations >= self.N):
            raise ValueError("associations contain an invalid UAV index.")

        horizontal = users - uavs[associations]
        distance_sq = np.sum(horizontal**2, axis=1) + float(self.H) ** 2
        channel_gain = self.beta_0 / np.maximum(distance_sq, 1e-12)
        bandwidth_per_gt = self.B_gt_total / max(self.M, 1)
        noise = self.noise_power
        if self.wireless_model == "psd_per_uav_budget":
            noise = self.noise_psd_w_per_hz * bandwidth_per_gt
        return bandwidth_per_gt * np.log2(
            1.0 + (self.p_gt * channel_gain) / noise
        )

    def allocate_u2u_link_bandwidth_hz(self, active_link_mask=None):
        if not self.dynamic_u2u_bandwidth:
            return self.u2u_link_bandwidth_matrix_hz.copy()

        if active_link_mask is None:
            active = ~np.eye(self.N, dtype=bool)
        else:
            active = np.asarray(active_link_mask, dtype=bool)
            if active.shape != (self.N, self.N):
                raise ValueError(
                    f"active_link_mask must have shape ({self.N}, {self.N})."
                )
            active = active.copy()
            np.fill_diagonal(active, False)

        bandwidth_hz = np.zeros((self.N, self.N), dtype=np.float64)
        active_count = int(np.count_nonzero(active))
        if active_count > 0:
            bandwidth_hz[active] = self.B_u2u_total / active_count
        return bandwidth_hz

    def compute_u2u_spectral_efficiency(self, uav_coords=None):
        """Legacy spectral efficiency, or fixed-reference quality in PSD mode.

        The latter is an observation feature, NOT actual service efficiency.
        """
        if self.wireless_model == "psd_per_uav_budget":
            return self.compute_u2u_reference_quality(uav_coords)
        if uav_coords is None:
            return self._get_u2u_spectral_efficiency().copy()
        coords = np.asarray(uav_coords, dtype=np.float64)
        if coords.shape != (self.N, 2):
            raise ValueError(f"uav_coords must have shape ({self.N}, 2).")
        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        channel_gain = self.beta_0 / np.maximum(dist_sq, 1e-12)
        np.fill_diagonal(channel_gain, 0.0)
        spectral_efficiency = np.log2(
            1.0 + (self.p_u2u * channel_gain) / self.noise_power
        )
        np.fill_diagonal(spectral_efficiency, 0.0)
        return spectral_efficiency

    def _u2u_channel_gain(self, uav_coords=None):
        # Only geometry enters this quantity; no actual resource result is cached.
        coords = np.asarray(self.uav_coords if uav_coords is None else uav_coords,
                            dtype=np.float64)
        if coords.shape != (self.N, 2) or not np.all(np.isfinite(coords)):
            raise ValueError(f"uav_coords must be finite with shape ({self.N}, 2).")
        dist_sq = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        gain = self.beta_0 / np.maximum(dist_sq, 1e-8 if uav_coords is None else 1e-12)
        np.fill_diagonal(gain, 0.0)
        return gain

    def compute_u2u_reference_quality(self, uav_coords=None):
        """Position-only 9D reference feature; preserve legacy API/normalization."""
        if self.wireless_model == "legacy_fixed_noise":
            return self.compute_u2u_spectral_efficiency(uav_coords)
        return np.log2(1.0 + self.u2u_reference_power_w * self._u2u_channel_gain(uav_coords)
                       / (self.noise_psd_w_per_hz * self.u2u_reference_bandwidth_hz))

    def allocate_u2u_resources(self, active_link_mask, uav_coords=None):
        """Candidate model: one immutable-for-this-step allocation, in SI units.

        Service supplies the mask from existing high+low transfer backlog.
        Callers doing hypothetical queries must supply their own active mask.
        """
        if self.wireless_model != "psd_per_uav_budget":
            raise ValueError("allocate_u2u_resources requires psd_per_uav_budget.")
        if not self.dynamic_u2u_bandwidth:
            raise ValueError("psd_per_uav_budget does not support fixed link bandwidths.")
        self._positive_wireless_parameter("B_u2u_total", self.B_u2u_total)
        active = np.asarray(active_link_mask, dtype=bool)
        if active.shape != (self.N, self.N):
            raise ValueError(f"active_link_mask must have shape ({self.N}, {self.N}).")
        active = active.copy()
        np.fill_diagonal(active, False)
        bandwidth = self.allocate_u2u_link_bandwidth_hz(active)
        model = getattr(self, "u2u_model", "psd_per_uav_budget")
        if model == "paper_eq13_fixed_noise":
            # Eq. (13) interpretation: P_M per active directed link, fixed sigma^2.
            power = np.where(active, self.u2u_tx_budget_w, 0.)
            noise = np.where(active, self.u2u_noise_w, 0.)
        elif model == "paper_eq13_per_uav_budget":
            # Eq. (13) with P_M as each UAV's total transmit budget, split
            # equally over its active outgoing links; fixed sigma^2.
            outgoing = active.sum(axis=1)
            power = np.divide(self.u2u_tx_budget_w * active, outgoing[:, None],
                              out=np.zeros_like(bandwidth), where=outgoing[:, None] > 0)
            noise = np.where(active, self.u2u_noise_w, 0.)
        elif model == "psd_per_uav_budget":
            outgoing = active.sum(axis=1)
            power = np.divide(self.u2u_tx_budget_w * active, outgoing[:, None],
                              out=np.zeros_like(bandwidth), where=outgoing[:, None] > 0)
            noise = self.noise_psd_w_per_hz * bandwidth
        else:
            raise ValueError("Unknown U2U resource model")
        gain = self._u2u_channel_gain(uav_coords)
        if not np.all(np.isfinite(noise)) or np.any(noise[active] <= 0):
            raise ValueError("Wireless parameters produce invalid noise power.")
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                snr = np.divide(power * gain, noise, out=np.zeros_like(noise), where=active)
                rate = bandwidth * np.log2(1.0 + snr)
        except FloatingPointError as error:
            raise ValueError("Wireless parameters produce non-finite U2U rates.") from error
        if not np.all(np.isfinite(rate)):
            raise ValueError("Wireless parameters produce invalid noise or rates.")
        return {"u2u_link_bandwidth_hz": bandwidth, "u2u_link_power_w": power,
                "u2u_link_noise_w": noise, "u2u_rate_bps": rate}

    def compute_u2u_rate_bps(
        self,
        uav_coords=None,
        active_link_mask=None,
        return_bandwidth=False,
    ):
        if self.wireless_model == "psd_per_uav_budget":
            # No-mask queries retain the all-directed-links query convention.
            mask = ~np.eye(self.N, dtype=bool) if active_link_mask is None else active_link_mask
            resources = self.allocate_u2u_resources(mask, uav_coords)
            if return_bandwidth:
                return resources["u2u_rate_bps"], resources["u2u_link_bandwidth_hz"]
            return resources["u2u_rate_bps"]
        spectral_efficiency = self.compute_u2u_spectral_efficiency(
            uav_coords=uav_coords
        )
        bandwidth_hz = self.allocate_u2u_link_bandwidth_hz(
            active_link_mask
        )
        rate_bps = bandwidth_hz * spectral_efficiency
        if return_bandwidth:
            return rate_bps, bandwidth_hz
        return rate_bps

    def _new_loc_task_queues(self):
        return [deque() for _ in range(self.N)]

    def _new_tx_task_queues(self):
        return [[deque() for _ in range(self.N)] for _ in range(self.N)]

    def _init_task_tracker(self):
        self.parent_tasks = {}
        self.next_task_id = 0
        self.next_chunk_id = 0
        self.current_arrival_tasks_by_uav = [[] for _ in range(self.N)]
        self.uplink_task_queues = [deque() for _ in range(self.M)]
        self.completed_upload_tasks_by_uav = [[] for _ in range(self.N)]
        self.Q_loc_high_tasks = self._new_loc_task_queues()
        self.Q_loc_low_tasks = self._new_loc_task_queues()
        self.Q_tx_high_tasks = self._new_tx_task_queues()
        self.Q_tx_low_tasks = self._new_tx_task_queues()

        self.completed_task_count = 0
        self.completed_task_delay_sum = 0.0
        self.arrived_task_count = 0
        self.arrived_priority_count = {"high": 0, "low": 0}
        self.qos_violation_count = 0
        self.qos_violation_priority_count = {"high": 0, "low": 0}
        self.completed_priority_count = {"high": 0, "low": 0}
        self.completed_priority_delay_sum = {"high": 0.0, "low": 0.0}
        self.completed_type_count = {
            name: 0 for name in self.device_type_names
        }
        self.completed_type_delay_sum = {
            name: 0.0 for name in self.device_type_names
        }
        self.arrived_source_count = np.zeros(self.N, dtype=np.int64)
        self.completed_source_count = np.zeros(self.N, dtype=np.int64)
        self.completed_source_delay_sum = np.zeros(self.N, dtype=np.float64)
        self.arrived_source_priority_count = {
            priority: np.zeros(self.N, dtype=np.int64)
            for priority in ("high", "low")
        }
        self.completed_source_priority_count = {
            priority: np.zeros(self.N, dtype=np.int64)
            for priority in ("high", "low")
        }
        self.completed_source_priority_delay_sum = {
            priority: np.zeros(self.N, dtype=np.float64)
            for priority in ("high", "low")
        }
        self.coop_flow_mbits = np.zeros((self.N, self.N), dtype=np.float64)

    def _safe_avg_ms(self, delay_sum_s, count):
        if count <= 0:
            return 0.0
        return 1000.0 * float(delay_sum_s) / float(count)

    def _register_arrived_parent_task(self, parent):
        self.arrived_task_count += 1
        self.arrived_priority_count[parent.priority] = (
            self.arrived_priority_count.get(parent.priority, 0) + 1
        )
        if 0 <= int(parent.source_uav) < self.N:
            source_uav = int(parent.source_uav)
            self.arrived_source_count[source_uav] += 1
            if parent.priority in self.arrived_source_priority_count:
                self.arrived_source_priority_count[parent.priority][source_uav] += 1

    def _unregister_arrived_parent_task(self, parent):
        self.arrived_task_count = max(int(self.arrived_task_count) - 1, 0)
        self.arrived_priority_count[parent.priority] = max(
            self.arrived_priority_count.get(parent.priority, 0) - 1,
            0,
        )
        if 0 <= int(parent.source_uav) < self.N:
            source_uav = int(parent.source_uav)
            self.arrived_source_count[source_uav] = max(
                int(self.arrived_source_count[source_uav]) - 1,
                0,
            )
            if parent.priority in self.arrived_source_priority_count:
                self.arrived_source_priority_count[parent.priority][source_uav] = max(
                    int(self.arrived_source_priority_count[parent.priority][source_uav]) - 1,
                    0,
                )

    def get_alive_task_wait_info(self):
        holding_cost = 0.0
        weighted_holding_cost = 0.0
        high_count = 0
        low_count = 0

        for parent in self.parent_tasks.values():
            if parent.finish_time is not None:
                continue
            weight = self.w_D_high if parent.priority == "high" else self.w_D_low
            holding_cost += float(self.micro_dt)
            weighted_holding_cost += float(weight) * float(self.micro_dt)
            if parent.priority == "high":
                high_count += 1
            else:
                low_count += 1

        return {
            "task_holding_cost": float(holding_cost),
            "task_holding_cost_weighted": float(weighted_holding_cost),
            # Backward-compatible alias. This is an online holding cost over
            # all alive parent tasks, including upload-stage tasks.
            "task_wait_cost": float(holding_cost),
            "task_wait_cost_weighted": float(weighted_holding_cost),
            "task_alive_count": int(high_count + low_count),
            "task_alive_high_count": int(high_count),
            "task_alive_low_count": int(low_count),
        }

    def get_task_delay_stats(self):
        high_done = self.completed_priority_count.get("high", 0)
        low_done = self.completed_priority_count.get("low", 0)
        high_vio = self.qos_violation_priority_count.get("high", 0)
        low_vio = self.qos_violation_priority_count.get("low", 0)
        high_delay_sum = self.completed_priority_delay_sum.get("high", 0.0)
        low_delay_sum = self.completed_priority_delay_sum.get("low", 0.0)
        weighted_delay_den = self.w_D_high * high_done + self.w_D_low * low_done
        weighted_delay_s = (
            (self.w_D_high * high_delay_sum + self.w_D_low * low_delay_sum)
            / max(float(weighted_delay_den), 1e-12)
        )

        stats = {
            "task_arrived_count": int(self.arrived_task_count),
            "task_completed_count": int(self.completed_task_count),
            "task_completion_rate": float(self.completed_task_count)
            / max(float(self.arrived_task_count), 1.0),
            "task_qos_violation_count": int(self.qos_violation_count),
            "task_qos_violation_rate": float(self.qos_violation_count)
            / max(float(self.completed_task_count), 1.0),
            "task_qos_violation_high_rate": float(high_vio)
            / max(float(high_done), 1.0),
            "task_qos_violation_low_rate": float(low_vio)
            / max(float(low_done), 1.0),
            "task_delay_avg_ms": self._safe_avg_ms(
                self.completed_task_delay_sum,
                self.completed_task_count,
            ),
            "task_delay_weighted_ms": float(weighted_delay_s * 1000.0),
            "task_delay_e2e_weighted_ms": float(weighted_delay_s * 1000.0),
        }
        stats.update(self.get_alive_task_wait_info())

        source_delay_ms = np.zeros(self.N, dtype=np.float32)
        for source_uav in range(self.N):
            source_delay_ms[source_uav] = self._safe_avg_ms(
                self.completed_source_delay_sum[source_uav],
                self.completed_source_count[source_uav],
            )
        stats["task_arrived_source_count"] = self.arrived_source_count.copy()
        stats["task_completed_source_count"] = self.completed_source_count.copy()
        stats["task_delay_source_uav_ms"] = source_delay_ms

        source_priority_delays = []
        source_priority_completed = []
        source_priority_arrived = []
        for priority in ("high", "low"):
            completed = self.completed_source_priority_count[priority].copy()
            arrived = self.arrived_source_priority_count[priority].copy()
            delay_sum = self.completed_source_priority_delay_sum[priority]
            delay_ms = np.divide(
                delay_sum,
                np.maximum(completed, 1),
                out=np.zeros(self.N, dtype=np.float64),
                where=completed > 0,
            ).astype(np.float32) * 1000.0

            stats[f"task_delay_source_{priority}_ms"] = delay_ms
            stats[f"task_completed_source_{priority}_count"] = completed
            stats[f"task_arrived_source_{priority}_count"] = arrived
            source_priority_delays.append(delay_ms)
            source_priority_completed.append(completed.astype(np.float32))
            source_priority_arrived.append(arrived.astype(np.float32))

        stats["task_delay_source_priority_ms"] = np.stack(
            source_priority_delays,
            axis=1,
        )
        stats["task_completed_source_priority_count"] = np.stack(
            source_priority_completed,
            axis=1,
        )
        stats["task_arrived_source_priority_count"] = np.stack(
            source_priority_arrived,
            axis=1,
        )

        for priority in ("high", "low"):
            stats[f"task_completed_{priority}_count"] = int(
                self.completed_priority_count.get(priority, 0)
            )
            stats[f"task_delay_{priority}_ms"] = self._safe_avg_ms(
                self.completed_priority_delay_sum.get(priority, 0.0),
                self.completed_priority_count.get(priority, 0),
            )

        for task_type in self.device_type_names:
            stats[f"task_completed_{task_type}_count"] = int(
                self.completed_type_count.get(task_type, 0)
            )
            stats[f"task_delay_{task_type}_ms"] = self._safe_avg_ms(
                self.completed_type_delay_sum.get(task_type, 0.0),
                self.completed_type_count.get(task_type, 0),
            )

        return stats

    def get_task_cumulative_counters(self):
        """Return a copy of cumulative task counters for windowed evaluation."""
        alive = self.get_alive_task_wait_info()
        queue_high = self.Q_loc_high + np.sum(self.Q_tx_high, axis=1)
        queue_low = self.Q_loc_low + np.sum(self.Q_tx_low, axis=1)
        return {
            "time_s": self._current_time(),
            "arrived_count": int(self.arrived_task_count),
            "arrived_high_count": int(self.arrived_priority_count.get("high", 0)),
            "arrived_low_count": int(self.arrived_priority_count.get("low", 0)),
            "completed_count": int(self.completed_task_count),
            "completed_delay_sum_s": float(self.completed_task_delay_sum),
            "completed_high_count": int(
                self.completed_priority_count.get("high", 0)
            ),
            "completed_low_count": int(
                self.completed_priority_count.get("low", 0)
            ),
            "completed_high_delay_sum_s": float(
                self.completed_priority_delay_sum.get("high", 0.0)
            ),
            "completed_low_delay_sum_s": float(
                self.completed_priority_delay_sum.get("low", 0.0)
            ),
            "qos_violation_count": int(self.qos_violation_count),
            "qos_violation_high_count": int(
                self.qos_violation_priority_count.get("high", 0)
            ),
            "qos_violation_low_count": int(
                self.qos_violation_priority_count.get("low", 0)
            ),
            "arrived_source_count": self.arrived_source_count.copy(),
            "completed_source_count": self.completed_source_count.copy(),
            "completed_source_delay_sum_s": self.completed_source_delay_sum.copy(),
            "arrived_source_high_count": self.arrived_source_priority_count[
                "high"
            ].copy(),
            "arrived_source_low_count": self.arrived_source_priority_count[
                "low"
            ].copy(),
            "completed_source_high_count": self.completed_source_priority_count[
                "high"
            ].copy(),
            "completed_source_low_count": self.completed_source_priority_count[
                "low"
            ].copy(),
            "completed_source_high_delay_sum_s": (
                self.completed_source_priority_delay_sum["high"].copy()
            ),
            "completed_source_low_delay_sum_s": (
                self.completed_source_priority_delay_sum["low"].copy()
            ),
            "alive_count": int(alive["task_alive_count"]),
            "alive_high_count": int(alive["task_alive_high_count"]),
            "alive_low_count": int(alive["task_alive_low_count"]),
            "queue_high_mbits_per_uav": queue_high.astype(np.float64).copy(),
            "queue_low_mbits_per_uav": queue_low.astype(np.float64).copy(),
            "queue_total_mbits_per_uav": (
                queue_high + queue_low
            ).astype(np.float64).copy(),
        }

    def _create_parent_task(
        self,
        user_id,
        priority,
        size_mbits,
        source_uav=None,
    ):
        task_id = self.next_task_id
        self.next_task_id += 1

        if source_uav is None:
            source_uav = int(self.user_allocations[int(user_id)])
        source_uav = int(source_uav)
        task_type = self.device_type_names[
            int(self.user_device_type_ids[user_id])
        ]
        parent = ParentTask(
            task_id=task_id,
            user_id=int(user_id),
            source_uav=source_uav,
            task_type=task_type,
            priority=priority,
            birth_time=self._current_time(),
            size_mbits=float(size_mbits),
            cpu_cycles=float(size_mbits) * 1e6 * float(self.C),
            remaining_upload_mbits=float(size_mbits),
        )
        self.parent_tasks[task_id] = parent
        self._register_arrived_parent_task(parent)
        self.current_arrival_tasks_by_uav[source_uav].append(task_id)
        return parent

    def _create_chunk(self, parent, size_mbits, target_uav, stage):
        chunk = TaskChunk(
            parent_task_id=parent.task_id,
            chunk_id=self.next_chunk_id,
            birth_time=parent.birth_time,
            priority=parent.priority,
            task_type=parent.task_type,
            remaining_mbits=float(size_mbits),
            initial_mbits=float(size_mbits),
            source_uav=parent.source_uav,
            target_uav=int(target_uav),
            current_stage=stage,
        )
        self.next_chunk_id += 1
        return chunk

    def _enqueue_compute_chunk(self, chunk):
        # [PERF] 增量维护聚合队列, 取代每步多次的全量重新求和。
        chunk.current_stage = "compute_queue"
        if chunk.priority == "high":
            self.Q_loc_high_tasks[chunk.target_uav].append(chunk)
            self.Q_loc_high[chunk.target_uav] += chunk.remaining_mbits
        else:
            self.Q_loc_low_tasks[chunk.target_uav].append(chunk)
            self.Q_loc_low[chunk.target_uav] += chunk.remaining_mbits

    def _enqueue_tx_chunk(self, chunk):
        # [PERF] 增量维护聚合队列。
        chunk.current_stage = "u2u_queue"
        if chunk.priority == "high":
            self.Q_tx_high_tasks[chunk.source_uav][chunk.target_uav].append(chunk)
            self.Q_tx_high[chunk.source_uav][chunk.target_uav] += chunk.remaining_mbits
        else:
            self.Q_tx_low_tasks[chunk.source_uav][chunk.target_uav].append(chunk)
            self.Q_tx_low[chunk.source_uav][chunk.target_uav] += chunk.remaining_mbits

    def _record_completed_parent_task(self, parent):
        delay_s = max(float(parent.finish_time - parent.birth_time), 0.0)
        self.completed_task_count += 1
        self.completed_task_delay_sum += delay_s
        deadline = float(self.qos_deadline_s.get(parent.priority, np.inf))
        if delay_s > deadline:
            self.qos_violation_count += 1
            self.qos_violation_priority_count[parent.priority] = (
                self.qos_violation_priority_count.get(parent.priority, 0) + 1
            )

        self.completed_priority_count[parent.priority] = (
            self.completed_priority_count.get(parent.priority, 0) + 1
        )
        self.completed_priority_delay_sum[parent.priority] = (
            self.completed_priority_delay_sum.get(parent.priority, 0.0)
            + delay_s
        )

        self.completed_type_count[parent.task_type] = (
            self.completed_type_count.get(parent.task_type, 0) + 1
        )
        self.completed_type_delay_sum[parent.task_type] = (
            self.completed_type_delay_sum.get(parent.task_type, 0.0)
            + delay_s
        )
        if 0 <= int(parent.source_uav) < self.N:
            source_uav = int(parent.source_uav)
            self.completed_source_count[source_uav] += 1
            self.completed_source_delay_sum[source_uav] += delay_s
            if parent.priority in self.completed_source_priority_count:
                self.completed_source_priority_count[parent.priority][source_uav] += 1
                self.completed_source_priority_delay_sum[parent.priority][source_uav] += delay_s
        self.parent_tasks.pop(parent.task_id, None)

    def _mark_chunk_completed(self, chunk, finish_time):
        parent = self.parent_tasks.get(chunk.parent_task_id)
        if parent is None or parent.finish_time is not None:
            return

        parent.completed_chunks += 1
        parent.chunk_finish_times.append(float(finish_time))
        if (
            parent.total_chunks > 0
            and parent.completed_chunks >= parent.total_chunks
        ):
            parent.finish_time = max(parent.chunk_finish_times)
            self._record_completed_parent_task(parent)

    def _admit_current_arrival_tasks_to_uplink(self):
        for task_ids in self.current_arrival_tasks_by_uav:
            for task_id in task_ids:
                parent = self.parent_tasks.get(task_id)
                if parent is None:
                    continue
                self.uplink_task_queues[parent.user_id].append(task_id)

    def _take_completed_upload_tasks(self):
        completed = self.completed_upload_tasks_by_uav
        self.completed_upload_tasks_by_uav = [[] for _ in range(self.N)]
        return completed

    def _serve_uplink_task_queues(self):
        """Serve independent GT uplinks with the paper's B_K/K allocation."""
        eps = 1e-12
        step_start = self._current_time()
        task_stage_durations = {}
        for queue in self.uplink_task_queues:
            for task_id in queue:
                if task_id in self.parent_tasks:
                    task_stage_durations[task_id] = float(self.micro_dt)

        energy_per_source = np.zeros(self.N, dtype=np.float64)
        service_per_user = np.zeros(self.M, dtype=np.float64)
        active_time_per_user = np.zeros(self.M, dtype=np.float64)
        completed_per_source = np.zeros(self.N, dtype=np.int64)

        for user_id, queue in enumerate(self.uplink_task_queues):
            elapsed = 0.0
            while queue and elapsed < self.micro_dt - eps:
                task_id = queue[0]
                parent = self.parent_tasks.get(task_id)
                if parent is None:
                    queue.popleft()
                    continue

                source_uav = int(parent.source_uav)
                rate_bps = float(
                    self.compute_u2g_rate_bps(
                        user_coords=self.user_coords[[user_id]],
                        associations=np.asarray([source_uav], dtype=np.int64),
                    )[0]
                )
                rate_mbits_per_s = rate_bps / 1e6
                if rate_mbits_per_s <= eps:
                    break

                available_time = float(self.micro_dt) - elapsed
                service_mbits = min(
                    float(parent.remaining_upload_mbits),
                    rate_mbits_per_s * available_time,
                )
                transmission_time = service_mbits / rate_mbits_per_s
                parent.remaining_upload_mbits = max(
                    float(parent.remaining_upload_mbits) - service_mbits,
                    0.0,
                )
                service_per_user[user_id] += service_mbits
                active_time_per_user[user_id] += transmission_time
                energy_per_source[source_uav] += (
                    self.p_gt * transmission_time
                )
                elapsed += transmission_time

                if parent.remaining_upload_mbits > eps:
                    break

                queue.popleft()
                parent.remaining_upload_mbits = 0.0
                parent.upload_finish_time = step_start + elapsed
                task_stage_durations[task_id] = elapsed
                self.completed_upload_tasks_by_uav[source_uav].append(task_id)
                completed_per_source[source_uav] += 1

        upload_delay_high = 0.0
        upload_delay_low = 0.0
        upload_delay_per_uav = np.zeros(self.N, dtype=np.float64)
        for task_id, stage_duration in task_stage_durations.items():
            parent = self.parent_tasks.get(task_id)
            if parent is None:
                continue
            if parent.priority == "high":
                upload_delay_high += stage_duration
                weighted_duration = 5.0 * stage_duration
            else:
                upload_delay_low += stage_duration
                weighted_duration = stage_duration
            upload_delay_per_uav[parent.source_uav] += weighted_duration

        effective_rate_bps = np.divide(
            service_per_user * 1e6,
            active_time_per_user,
            out=np.zeros(self.M, dtype=np.float64),
            where=active_time_per_user > eps,
        )
        return {
            "energy_per_uav": energy_per_source,
            "service_mbits_per_user": service_per_user,
            "active_time_s_per_user": active_time_per_user,
            "effective_rate_bps_per_user": effective_rate_bps,
            "completed_per_uav": completed_per_source,
            "upload_delay_cost_high": float(upload_delay_high),
            "upload_delay_cost_low": float(upload_delay_low),
            "upload_delay_cost_per_uav": upload_delay_per_uav,
        }

    def _discard_current_arrival_tasks(self):
        for task_ids in getattr(self, "current_arrival_tasks_by_uav", []):
            for task_id in task_ids:
                parent = self.parent_tasks.get(task_id)
                if parent is not None:
                    self._unregister_arrived_parent_task(parent)
                    self.parent_tasks.pop(task_id, None)
        self.current_arrival_tasks_by_uav = [[] for _ in range(self.N)]

    def _route_arrived_parent_tasks(
        self,
        arrived_task_ids_by_uav,
        split_ratios_high,
        split_ratios_low,
    ):
        current_time = self._current_time()
        eps = 1e-10

        for source_uav, task_ids in enumerate(arrived_task_ids_by_uav):
            for task_id in task_ids:
                parent = self.parent_tasks.get(task_id)
                if parent is None:
                    continue

                ratios = (
                    split_ratios_high[source_uav]
                    if parent.priority == "high"
                    else split_ratios_low[source_uav]
                )

                chunks = []
                for target_uav, ratio in enumerate(ratios):
                    chunk_size = parent.size_mbits * float(max(ratio, 0.0))
                    if chunk_size <= eps:
                        continue
                    stage = (
                        "compute_queue"
                        if target_uav == source_uav
                        else "u2u_queue"
                    )
                    chunks.append(
                        self._create_chunk(
                            parent,
                            chunk_size,
                            target_uav,
                            stage,
                        )
                    )

                parent.total_chunks = len(chunks)
                if parent.total_chunks <= 0:
                    parent.finish_time = current_time
                    self._record_completed_parent_task(parent)
                    continue

                for chunk in chunks:
                    self.coop_flow_mbits[source_uav, chunk.target_uav] += (
                        chunk.initial_mbits
                    )
                    if chunk.target_uav == source_uav:
                        self._enqueue_compute_chunk(chunk)
                    else:
                        self._enqueue_tx_chunk(chunk)
        # [PERF] 聚合队列已由 _enqueue_* 增量维护, 无需全量重新求和。

    def _consume_compute_queue(
        self,
        queue,
        capacity_mbits,
        finish_time,
        service_by_priority=None,
        service_by_type=None,
    ):
        remaining_capacity = float(max(capacity_mbits, 0.0))
        consumed = 0.0
        eps = 1e-10

        while remaining_capacity > eps and queue:
            chunk = queue[0]
            served = min(remaining_capacity, chunk.remaining_mbits)
            chunk.remaining_mbits -= served
            remaining_capacity -= served
            consumed += served
            if service_by_priority is not None:
                service_by_priority[chunk.priority] = (
                    service_by_priority.get(chunk.priority, 0.0) + served
                )
            if service_by_type is not None:
                service_by_type[chunk.task_type] = (
                    service_by_type.get(chunk.task_type, 0.0) + served
                )

            if chunk.remaining_mbits <= eps:
                queue.popleft()
                chunk.remaining_mbits = 0.0
                chunk.current_stage = "completed"
                self._mark_chunk_completed(chunk, finish_time)

        return consumed

    def _serve_compute_task_queues(
        self,
        capacities_mbits,
        priority,
        service_by_priority=None,
        service_by_type=None,
    ):
        consumed = np.zeros(self.N, dtype=np.float64)
        finish_time = self._current_time() + float(self.micro_dt)
        queues = (
            self.Q_loc_high_tasks
            if priority == "high"
            else self.Q_loc_low_tasks
        )

        agg = self.Q_loc_high if priority == "high" else self.Q_loc_low
        for i in range(self.N):
            consumed[i] = self._consume_compute_queue(
                queues[i],
                capacities_mbits[i],
                finish_time,
                service_by_priority=service_by_priority,
                service_by_type=service_by_type,
            )
            # [PERF] 增量维护: 计算服务只会减少本地队列; 队列清空则归零防漂移。
            if queues[i]:
                agg[i] -= consumed[i]
            else:
                agg[i] = 0.0
        return consumed

    def _consume_tx_queue(self, queue, capacity_mbits):
        remaining_capacity = float(max(capacity_mbits, 0.0))
        consumed = 0.0
        eps = 1e-10

        while remaining_capacity > eps and queue:
            chunk = queue[0]
            served = min(remaining_capacity, chunk.remaining_mbits)
            chunk.remaining_mbits -= served
            remaining_capacity -= served
            consumed += served

            if chunk.remaining_mbits <= eps:
                queue.popleft()
                chunk.remaining_mbits = chunk.initial_mbits
                self._enqueue_compute_chunk(chunk)

        return consumed

    def _serve_tx_task_queues(self, capacities_mbits, priority):
        consumed = np.zeros((self.N, self.N), dtype=np.float64)
        queues = (
            self.Q_tx_high_tasks
            if priority == "high"
            else self.Q_tx_low_tasks
        )

        agg = self.Q_tx_high if priority == "high" else self.Q_tx_low
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                consumed[i, j] = self._consume_tx_queue(
                    queues[i][j],
                    capacities_mbits[i, j],
                )
                # [PERF] 增量维护: 传输服务减少 tx 队列(完成的 chunk 已在
                # _enqueue_compute_chunk 内自动加到 Q_loc); 清空则归零防漂移。
                if queues[i][j]:
                    agg[i, j] -= consumed[i, j]
                else:
                    agg[i, j] = 0.0
        return consumed

    def _get_zone_device_mix(self, zone_idx):
        if zone_idx in ZONE_DEVICE_MIXES:
            return ZONE_DEVICE_MIXES[zone_idx]
        if zone_idx == 0:
            return ZONE_DEVICE_MIXES[0]
        return ZONE_DEVICE_MIXES[1 + ((zone_idx - 1) % 2)]

    def _normalize_zone_loads(self, zone_loads):
        if zone_loads is None or (
            isinstance(zone_loads, str) and not zone_loads.strip()
        ):
            values = [
                ZONE_TARGET_LOAD_MBPS.get(
                    idx,
                    ZONE_TARGET_LOAD_MBPS[1 + ((idx - 1) % 2)]
                    if idx > 0
                    else ZONE_TARGET_LOAD_MBPS[0],
                )
                for idx in range(self.N)
            ]
        elif isinstance(zone_loads, str):
            values = [
                float(item.strip())
                for item in zone_loads.split(",")
                if item.strip()
            ]
        else:
            values = [float(item) for item in zone_loads]

        if not values:
            raise ValueError("zone_loads must contain at least one positive value.")

        arr = np.asarray(values, dtype=np.float32)
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("zone_loads must be finite positive Mbps values.")

        if arr.size < self.N:
            arr = np.asarray(
                [arr[idx % arr.size] for idx in range(self.N)],
                dtype=np.float32,
            )
        elif arr.size > self.N:
            arr = arr[: self.N].astype(np.float32)

        return arr.astype(np.float32)

    def set_zone_target_loads(self, zone_loads):
        """Update future task-arrival rates without touching existing queues."""
        self.zone_target_load_mbps = self._normalize_zone_loads(zone_loads)
        self._configure_industrial_device_profiles()
        return self.zone_expected_offered_load_mbps.copy()

    def _assign_user_clusters(self):
        """Deterministically split each zone's users into spatial clusters.

        The split is a zero-RNG projection partition: zone users are sorted by
        their projection onto a zone-specific fixed axis and cut into
        ``cluster_count`` contiguous bands. Projection bands are spatially
        separated (unlike angular sectors of a centred cloud), so rotating the
        load shares moves the traffic-weighted centroid, and the same scene
        seed always yields the same membership without consuming any
        agent/environment RNG stream.
        """
        for zone_idx in range(self.N):
            zone_users = np.where(self.user_allocations == zone_idx)[0]
            if zone_users.size == 0:
                continue
            if zone_users.size < self.cluster_count:
                raise ValueError(
                    "cluster_count exceeds the number of users in zone "
                    f"{zone_idx} ({zone_users.size})."
                )
            axis_angle = zone_idx * np.pi / 3.0
            axis = np.array([np.cos(axis_angle), np.sin(axis_angle)])
            offsets = self.user_coords[zone_users] - self.anchor_points[zone_idx]
            order = np.argsort(offsets @ axis, kind="stable")
            for cluster_idx, members in enumerate(
                np.array_split(zone_users[order], self.cluster_count)
            ):
                self.user_cluster_ids[members] = cluster_idx

    def set_cluster_target_shares(self, shares):
        """Update intra-zone cluster load shares without touching queues.

        ``shares`` must have shape ``(N, cluster_count)``; each row is
        normalized to sum to one. In exact_share mode these are aggregate load
        targets. In preserve_type_priority mode they are spatial weights for
        each device type, conditioned on the clusters containing that type.
        Actual aggregate shares can differ; zone/type/priority arrival totals
        are preserved instead. Queues and device identities never change.
        """
        shares = np.asarray(shares, dtype=np.float64)
        if shares.shape != (self.N, self.cluster_count):
            raise ValueError(
                "cluster shares must have shape "
                f"({self.N}, {self.cluster_count}), got {shares.shape}."
            )
        if not np.all(np.isfinite(shares)) or np.any(shares < 0.0):
            raise ValueError("Cluster shares must be finite and non-negative.")
        row_sums = shares.sum(axis=1)
        if np.any(row_sums <= 1e-12):
            raise ValueError("Each zone's cluster shares must sum to a positive value.")
        self.cluster_target_shares = shares / row_sums[:, None]
        self._configure_industrial_device_profiles()
        return self.cluster_expected_offered_load_mbps.copy()

    def get_compute_capacity_audit(self):
        """Return the deterministic arrival/service envelope per macro period."""
        arrival_duration_s = (
            self.macro_dt
            if self.arrivals_during_deployment
            else self.macro_dt - self.deployment_dt
        )
        service_duration_s = self.macro_dt - self.deployment_dt
        offered_load_mbps = np.asarray(
            [self._get_zone_target_load_mbps(i) for i in range(self.N)],
            dtype=np.float64,
        )
        arrival_mbits_per_macro = offered_load_mbps * arrival_duration_s
        compute_capacity_mbits_per_macro = (
            self.F_max_array.astype(np.float64)
            * service_duration_s
            / max(float(self.C) * 1e6, 1e-12)
        )
        local_margin_mbits = (
            compute_capacity_mbits_per_macro - arrival_mbits_per_macro
        )
        global_margin_mbits = float(
            np.sum(compute_capacity_mbits_per_macro)
            - np.sum(arrival_mbits_per_macro)
        )
        return {
            "arrival_duration_s": float(arrival_duration_s),
            "service_duration_s": float(service_duration_s),
            "offered_load_mbps": offered_load_mbps.copy(),
            "arrival_mbits_per_macro": arrival_mbits_per_macro.copy(),
            "compute_capacity_mbits_per_macro": (
                compute_capacity_mbits_per_macro.copy()
            ),
            "local_margin_mbits": local_margin_mbits.copy(),
            "locally_overloaded": (local_margin_mbits < 0.0),
            "global_margin_mbits": global_margin_mbits,
            "globally_feasible": bool(global_margin_mbits >= 0.0),
        }

    def _get_zone_target_load_mbps(self, zone_idx):
        scale = float(getattr(self, "load_scale", 1.0))
        zone_loads = getattr(self, "zone_target_load_mbps", None)
        if zone_loads is not None and len(zone_loads) > 0:
            return float(zone_loads[zone_idx % len(zone_loads)]) * scale
        if zone_idx in ZONE_TARGET_LOAD_MBPS:
            return float(ZONE_TARGET_LOAD_MBPS[zone_idx]) * scale
        if zone_idx == 0:
            return float(ZONE_TARGET_LOAD_MBPS[0]) * scale
        return float(ZONE_TARGET_LOAD_MBPS[1 + ((zone_idx - 1) % 2)]) * scale

    def _configure_industrial_device_profiles(self):
        self.user_lambda_high.fill(0.0)
        self.user_lambda_low.fill(0.0)
        self.user_task_size_high_mbits.fill(0.0)
        self.user_task_size_low_mbits.fill(0.0)
        self.zone_expected_offered_load_mbps.fill(0.0)

        for zone_idx in range(self.N):
            zone_users = np.where(self.user_allocations == zone_idx)[0]
            if zone_users.size == 0:
                continue

            zone_mix = self._get_zone_device_mix(zone_idx)
            expected_counts = zone_mix * zone_users.size
            type_counts = np.floor(expected_counts).astype(np.int32)
            remainder = int(zone_users.size - np.sum(type_counts))
            if remainder > 0:
                fractional = expected_counts - type_counts
                order = np.argsort(-fractional)
                type_counts[order[:remainder]] += 1

            expanded_type_ids = []
            for type_idx, count in enumerate(type_counts):
                expanded_type_ids.extend([type_idx] * int(count))
            expanded_type_ids = np.asarray(
                expanded_type_ids[: zone_users.size],
                dtype=np.int32,
            )
            if expanded_type_ids.size < zone_users.size:
                padding = np.full(
                    zone_users.size - expanded_type_ids.size,
                    len(self.device_type_names) - 1,
                    dtype=np.int32,
                )
                expanded_type_ids = np.concatenate([expanded_type_ids, padding])

            self.user_device_type_ids[zone_users] = expanded_type_ids

            if zone_idx == 0:
                base_high = self.lambda_hot_high
                base_low = self.lambda_hot_low
            else:
                base_high = self.lambda_cold_high
                base_low = self.lambda_cold_low

            for local_idx, user_idx in enumerate(zone_users):
                profile_name = self.device_type_names[
                    self.user_device_type_ids[user_idx]
                ]
                device_profile = DEVICE_PROFILES[profile_name]
                self.user_lambda_high[user_idx] = (
                    base_high * device_profile["high_rate_scale"]
                )
                self.user_lambda_low[user_idx] = (
                    base_low * device_profile["low_rate_scale"]
                )
                self.user_task_size_high_mbits[user_idx] = device_profile[
                    "high_size_mbits"
                ]
                self.user_task_size_low_mbits[user_idx] = device_profile[
                    "low_size_mbits"
                ]

            target_zone_load = float(self._get_zone_target_load_mbps(zone_idx))
            if (self.cluster_load_mode == "preserve_type_priority_capped"
                    or (self.cluster_count > 1 and self.cluster_load_mode == "preserve_type_priority")):
                # Match the single-cluster workload before redistributing it.
                raw_zone_load = float(np.sum(
                    self.user_lambda_high[zone_users] * self.user_task_size_high_mbits[zone_users]
                    + self.user_lambda_low[zone_users] * self.user_task_size_low_mbits[zone_users]
                ))
                if raw_zone_load > 1e-8:
                    zone_scale = max(target_zone_load / raw_zone_load, 1e-6)
                    self.user_lambda_high[zone_users] *= zone_scale
                    self.user_lambda_low[zone_users] *= zone_scale
                self._redistribute_cluster_arrivals(zone_idx, zone_users)
                for cluster_idx in range(self.cluster_count):
                    users = zone_users[self.user_cluster_ids[zone_users] == cluster_idx]
                    self.cluster_expected_offered_load_mbps[zone_idx, cluster_idx] = np.sum(
                        self.user_lambda_high[users] * self.user_task_size_high_mbits[users]
                        + self.user_lambda_low[users] * self.user_task_size_low_mbits[users]
                    )
                self.zone_expected_offered_load_mbps[zone_idx] = np.sum(
                    self.cluster_expected_offered_load_mbps[zone_idx]
                )
                continue
            shares = self.cluster_target_shares[zone_idx]
            for cluster_idx in range(self.cluster_count):
                cluster_users = zone_users[
                    self.user_cluster_ids[zone_users] == cluster_idx
                ]
                if cluster_users.size == 0:
                    self.cluster_expected_offered_load_mbps[
                        zone_idx, cluster_idx
                    ] = 0.0
                    continue
                raw_cluster_load = float(
                    np.sum(
                        self.user_lambda_high[cluster_users]
                        * self.user_task_size_high_mbits[cluster_users]
                        + self.user_lambda_low[cluster_users]
                        * self.user_task_size_low_mbits[cluster_users]
                    )
                )
                target_cluster_load = target_zone_load * float(shares[cluster_idx])
                if raw_cluster_load > 1e-8:
                    cluster_rate_scale = max(
                        target_cluster_load / raw_cluster_load, 1e-6
                    )
                    self.user_lambda_high[cluster_users] *= cluster_rate_scale
                    self.user_lambda_low[cluster_users] *= cluster_rate_scale
                    raw_cluster_load = float(
                        np.sum(
                            self.user_lambda_high[cluster_users]
                            * self.user_task_size_high_mbits[cluster_users]
                            + self.user_lambda_low[cluster_users]
                            * self.user_task_size_low_mbits[cluster_users]
                        )
                    )
                self.cluster_expected_offered_load_mbps[
                    zone_idx, cluster_idx
                ] = raw_cluster_load

            self.zone_expected_offered_load_mbps[zone_idx] = float(
                np.sum(self.cluster_expected_offered_load_mbps[zone_idx])
            )

    def _redistribute_cluster_arrivals(self, zone_idx, zone_users):
        """Move arrival intensity between same-type users, preserving its mix."""
        capped = self.cluster_load_mode == "preserve_type_priority_capped"
        ceiling = self.uplink_peak_rate_mbps() if capped else None
        for type_idx in range(len(self.device_type_names)):
            users = zone_users[self.user_device_type_ids[zone_users] == type_idx]
            if users.size == 0:
                continue
            clusters = self.user_cluster_ids[users]
            counts = np.bincount(clusters, minlength=self.cluster_count)
            weights = np.where(counts > 0, self.cluster_target_shares[zone_idx], 0.0)
            total = float(weights.sum())
            if total == 0.0:
                # No requested hotspot contains this type: keep its baseline.
                if not capped:
                    continue
                fractions = np.full(users.size, 1.0 / users.size)
            else:
                weights /= total
                fractions = weights[clusters] / counts[clusters]
            if capped:
                type_load = float(np.sum(
                    self.user_lambda_high[users].astype(np.float64)
                    * self.user_task_size_high_mbits[users]
                    + self.user_lambda_low[users].astype(np.float64)
                    * self.user_task_size_low_mbits[users]
                ))
                if type_load > users.size * ceiling:
                    raise ValueError(
                        f"Uplink arrival cap infeasible: zone={zone_idx}, "
                        f"type={self.device_type_names[type_idx]}, "
                        f"load={type_load:.9g} Mbps > {users.size} * {ceiling:.9g} Mbps; "
                        "cannot preserve this workload without exceeding the cap"
                    )
                if type_load > 0.0:
                    fractions = self._capacity_limited_fractions(fractions, ceiling / type_load)
            for rates in (self.user_lambda_high, self.user_lambda_low):
                type_rate = float(np.sum(rates[users], dtype=np.float64))
                rates[users] = type_rate * fractions

    def uplink_peak_rate_mbps(self):
        """Single-terminal overhead link bound, independent of the flown path."""
        return float(self.compute_u2g_rate_bps(
            user_coords=np.zeros((1, 2)),
            uav_coords=np.zeros((self.N, 2)),
            associations=np.zeros(1, dtype=np.int64),
        )[0] / 1e6)

    @staticmethod
    def _capacity_limited_fractions(weights, upper):
        """Cap proportional shares; spill excess to other same-type terminals."""
        weights = np.asarray(weights, dtype=np.float64)
        if (weights.ndim != 1 or not weights.size or not np.all(np.isfinite(weights))
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("Arrival weights must be a nonzero finite nonnegative vector")
        if not np.isfinite(upper) or upper <= 0 or weights.size * upper < 1.0 - 1e-12:
            raise ValueError("Arrival fraction cap cannot preserve total workload")
        fractions = np.zeros_like(weights)
        active = np.ones(weights.size, dtype=bool)
        remaining = 1.0
        while np.any(active):
            indices = np.flatnonzero(active)
            preference = weights[indices]
            # Zero-weight terminals receive overflow only when positive support is full.
            if preference.sum() == 0.0:
                preference = np.ones(indices.size)
            proposed = remaining * preference / preference.sum()
            saturated = proposed > upper
            if not np.any(saturated):
                fractions[indices] = proposed
                break
            filled = indices[saturated]
            fractions[filled] = upper
            remaining -= upper * filled.size
            active[filled] = False
        return fractions

    def _compute_tcom_flying_power(self, v):
        """Compute UAV fly/hover power with the TCOM U-shaped curve."""
        U1, U2, U3, U4, U5 = 78.0, 22.0, 265.0, 0.0092, 28.0
        v_tip = 110.0
        v = np.asarray(v, dtype=np.float64)
        return (
            U1 * (1 + 3 * v**2 / v_tip**2)
            + U2 * np.sqrt(np.maximum(np.sqrt(U3 + v**4 / 4.0) - v**2 / 2.0, 1e-10))
            + U4 * v**3
            + U5
        )

    def _compute_deployment_transient_energy(self, v):
        """Lower-bound start-up energy for a fly-then-hover deployment leg.

        Every macro period starts and ends its deployment leg at hover.  The
        steady-flight power curve does not include the kinetic energy needed
        to accelerate from rest.  With no regenerative recovery during the
        subsequent stop, 0.5*m*v^2 is a conservative battery-energy lower
        bound.  Deceleration and finite-time acceleration losses are omitted,
        so this term does not overstate manoeuvre losses.
        """
        v = np.asarray(v, dtype=np.float64)
        if self.deployment_transient_energy_mode == "none":
            return np.zeros_like(v, dtype=np.float64)
        return 0.5 * self.uav_mass_kg * np.square(v)

    def reset(self, seed=None, options=None):
        
        # [解耦修复]: 自行维护 Numpy Random Generator，不再依赖 gym
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random = np.random.RandomState()
            
        # [RPGM Mobility] 初始化群组中心（锚点）
        self.anchor_points = self.base_anchor_points.copy()

        # 1. 初始化用户相对组中心的偏移
        assigned_anchors = self.anchor_points[self.user_allocations] 
        self.user_offsets = self.np_random.standard_normal((self.M, 2)) * self.user_zone_scatter_std
        
        if self.user_zone_radius > 0.0:
            zone_dist = np.linalg.norm(
                self.user_offsets,
                axis=1,
                keepdims=True,
            )
            safe_zone_dist = np.maximum(zone_dist, 1e-8)
            self.user_offsets = np.where(
                zone_dist > self.user_zone_radius,
                self.user_offsets / safe_zone_dist * self.user_zone_radius,
                self.user_offsets,
            )
        
        self.user_coords = assigned_anchors + self.user_offsets
        self.user_coords = np.clip(self.user_coords, 0.0, 1000.0)
        self._assign_user_clusters()
        
        # 2. 计算初始的各管辖区质心
        for i in range(self.N):
            zone_users = self.user_coords[self.user_allocations == i]
            if len(zone_users) > 0:
                self.weighted_centroids[i] = np.mean(zone_users, axis=0) # 初始简单平均
            else:
                self.weighted_centroids[i] = self.anchor_points[i]
                
        # 3. UAV 部署在区域外围的三角形初始位置（以 (500,500) 为中心）
        self.uav_coords = self.uav_initial_positions.copy()
        self.uav_coords = np.clip(self.uav_coords, 0.0, 1000.0)
        self.current_user_associations = self.user_allocations.copy()
        
        # 4. 初始化 GMMM 战术移动初速度
        self.instantaneous_centroids = self.weighted_centroids.copy()

        self.user_speeds = np.full(
            self.M,
            0.0 if self.users_static else self.gm_mean_speed_mps,
            dtype=np.float64,
        )
        if self.gm_heading_mode == "reference_point_group":
            self.mobility_group_centers = self.anchor_points.copy()
            self.mobility_group_speeds = np.full(
                self.N,
                self.rpgm_group_speed_mps,
                dtype=np.float64,
            )
            self.mobility_group_headings = self.np_random.uniform(
                -np.pi,
                np.pi,
                size=self.N,
            )
            self.mobility_group_velocities = np.column_stack(
                [
                    self.mobility_group_speeds
                    * np.cos(self.mobility_group_headings),
                    self.mobility_group_speeds
                    * np.sin(self.mobility_group_headings),
                ]
            )
            self.user_local_velocities = self.np_random.normal(
                0.0,
                self.rpgm_local_velocity_std_mps,
                size=(self.M, 2),
            )
            self._clip_user_local_velocities()
            if self.continuous_mission:
                self._set_mission_group_velocity(0.0)
            self.user_mean_headings = self.mobility_group_headings[
                self.user_allocations
            ].copy()
            total_velocities = (
                self.mobility_group_velocities[self.user_allocations]
                + self.user_local_velocities
            )
            self._set_user_velocity_vectors(total_velocities)
        elif self.gm_heading_mode == "independent_episode_mean":
            self.user_mean_headings = self.np_random.uniform(
                -np.pi,
                np.pi,
                size=self.M,
            )
        else:
            self.user_mean_headings = np.full(
                self.M,
                self.gm_mean_heading_rad,
                dtype=np.float64,
            )
        if self.gm_heading_mode != "reference_point_group":
            self.user_headings = self.user_mean_headings.copy()
            self._sync_user_velocity_vectors()
        
        # 5. 清空控制指令缓存与流水队列
        self.current_macro_v = np.zeros(self.N)
        self.current_macro_theta = np.zeros(self.N)
        self.macro_fly_steps_remaining = 0
        self.macro_service_pause_steps_remaining = 0

        self.Q_loc_high = np.zeros(self.N)
        self.Q_loc_low = np.zeros(self.N)
        self.Q_tx_high = np.zeros((self.N, self.N))
        self.Q_tx_low = np.zeros((self.N, self.N))
        self.pending_upload_jobs = []
        self.global_micro_step = 0
        self._resource_frame_prepared = False
        self.resource_frame_prepare_count = 0
        self.deployment_arrival_prepare_count = 0
        self.mobility_update_count = 0
        if self.continuous_mission:
            self.mission_user_boundary_hits = 0
        self._u2u_spectral_cache_coords = None
        self._u2u_spectral_cache_result = None
        self.last_u2u_link_bandwidth_matrix_hz = (
            self.u2u_link_bandwidth_matrix_hz.copy()
        )
        if self.wireless_model == "psd_per_uav_budget":
            self.last_u2u_link_bandwidth_matrix_hz.fill(0.0)
        self._external_cpu_frequency_override = None
        self._external_cpu_override_steps_remaining = 0
        self.current_arrival_active_users = np.zeros(self.N)
        self.A_t_high = np.zeros(self.N)
        self.A_t_low = np.zeros(self.N)
        self._init_task_tracker()
        self._configure_industrial_device_profiles()
        self._clear_current_arrivals()
        
        if self.record_trajectory:
            self.uav_trajectory = []
            self.centroid_trajectory = []
            self.users_trajectory = []
            self.group_center_trajectory = []
            self.arrival_high_trajectory = []
            self.arrival_low_trajectory = []
            self.record_trajectory_snapshot()
            
        return self.get_raw_state(), {}

    def get_raw_state(self):
        """
        暴露纯粹的物理底层原始数据供外接 RL Wrapper 提取封装
        """
        return {
            'uav_coords': self.uav_coords.copy(),
            'user_coords': self.user_coords.copy(),
            'weighted_centroids': self.weighted_centroids.copy(),
            'Q_loc_high': self.Q_loc_high.copy(),
            'Q_loc_low': self.Q_loc_low.copy(),
            'Q_tx_high': self.Q_tx_high.copy(),
            'Q_tx_low': self.Q_tx_low.copy(),
            'A_t_high': getattr(self, 'A_t_high', np.zeros(self.N)).copy(),
            'A_t_low': getattr(self, 'A_t_low', np.zeros(self.N)).copy(),
            'user_associations': self.current_user_associations.copy(),
            'u2u_link_bandwidth_hz': (
                self.last_u2u_link_bandwidth_matrix_hz.copy()
            ),
        }

    @staticmethod
    def _wrap_angle(angle):
        return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi

    def _sync_user_velocity_vectors(self):
        self.user_velocities = np.column_stack(
            [
                self.user_speeds * np.cos(self.user_headings),
                self.user_speeds * np.sin(self.user_headings),
            ]
        )

    def _set_user_velocity_vectors(self, velocities):
        self.user_velocities = np.asarray(velocities, dtype=np.float64).copy()
        self.user_speeds = np.linalg.norm(self.user_velocities, axis=1)
        self.user_headings = np.arctan2(
            self.user_velocities[:, 1],
            self.user_velocities[:, 0],
        )

    def _clip_user_local_velocities(self):
        cap = float(self.rpgm_local_speed_cap_mps)
        if cap <= 0.0:
            self.user_local_velocities.fill(0.0)
            return
        speeds = np.linalg.norm(self.user_local_velocities, axis=1)
        scale = np.minimum(1.0, cap / np.maximum(speeds, 1e-12))
        self.user_local_velocities *= scale[:, None]

    def _update_reference_point_group_mobility(self):
        frame_dt = float(self.resource_frame_dt)
        group_speed_noise = self.np_random.normal(
            0.0,
            self.rpgm_group_speed_noise_std,
            size=self.N,
        )
        group_turn_noise = self.np_random.normal(
            0.0,
            self.rpgm_group_turn_std_rad_sqrt_s * np.sqrt(frame_dt),
            size=self.N,
        )
        c_group = self.rpgm_group_speed_memory
        self.mobility_group_speeds = (
            c_group * self.mobility_group_speeds
            + (1.0 - c_group) * self.rpgm_group_speed_mps
            + np.sqrt(max(1.0 - c_group**2, 0.0)) * group_speed_noise
        )
        self.mobility_group_speeds = np.clip(
            self.mobility_group_speeds,
            0.0,
            self.max_infantry_speed,
        )
        self.mobility_group_headings = self._wrap_angle(
            self.mobility_group_headings + group_turn_noise
        )
        self.mobility_group_velocities = np.column_stack(
            [
                self.mobility_group_speeds
                * np.cos(self.mobility_group_headings),
                self.mobility_group_speeds
                * np.sin(self.mobility_group_headings),
            ]
        )

        if self.continuous_mission:
            # Retain legacy RNG draws before the local innovations and traffic.
            self._set_mission_group_velocity(self._current_time())

        c_local = self.rpgm_local_velocity_memory
        local_innovation = self.np_random.normal(
            0.0,
            np.sqrt(max(1.0 - c_local**2, 0.0))
            * self.rpgm_local_velocity_std_mps,
            size=(self.M, 2),
        )
        restoring_velocity = (
            -self.rpgm_local_restore_rate_s2
            * self.user_offsets
            * frame_dt
        )
        self.user_local_velocities = (
            c_local * self.user_local_velocities
            + local_innovation
            + restoring_velocity
        )
        self._clip_user_local_velocities()
        self.user_mean_headings = self.mobility_group_headings[
            self.user_allocations
        ].copy()
        total_velocities = (
            self.mobility_group_velocities[self.user_allocations]
            + self.user_local_velocities
        )
        self._set_user_velocity_vectors(total_velocities)

    def _set_mission_group_velocity(self, time_s):
        velocity = (self.mission_routes_m[:, 1] - self.mission_routes_m[:, 0]) / self.mission_duration_s
        if time_s >= self.mission_duration_s:
            velocity = np.zeros_like(velocity)
        self.mobility_group_velocities = velocity
        self.mobility_group_speeds = np.linalg.norm(velocity, axis=1)
        self.mobility_group_headings = np.arctan2(velocity[:, 1], velocity[:, 0])

    def _update_gauss_markov_mobility(self):
        if self.users_static:
            self.user_speeds.fill(0.0)
            self.user_headings[:] = self.user_mean_headings
            self._sync_user_velocity_vectors()
            return

        if self.gm_heading_mode == "reference_point_group":
            self._update_reference_point_group_mobility()
            self.mobility_update_count += 1
            return

        speed_noise = self.np_random.normal(
            0.0,
            self.gm_speed_noise_std,
            size=self.M,
        )
        heading_noise = self.np_random.normal(
            0.0,
            self.gm_heading_noise_std,
            size=self.M,
        )
        c_speed = self.gm_speed_memory
        c_heading = self.gm_heading_memory
        self.user_speeds = (
            c_speed * self.user_speeds
            + (1.0 - c_speed) * self.gm_mean_speed_mps
            + np.sqrt(max(1.0 - c_speed**2, 0.0)) * speed_noise
        )
        self.user_speeds = np.clip(
            self.user_speeds,
            0.0,
            self.max_infantry_speed,
        )
        if self.gm_heading_mode == "independent_episode_mean":
            heading_offset = self._wrap_angle(
                self.user_headings - self.user_mean_headings
            )
            self.user_headings = self._wrap_angle(
                self.user_mean_headings
                + c_heading * heading_offset
                + np.sqrt(max(1.0 - c_heading**2, 0.0)) * heading_noise
            )
        else:
            # Keep the historical shared-heading update exactly reproducible.
            self.user_headings = self._wrap_angle(
                c_heading * self.user_headings
                + (1.0 - c_heading) * self.gm_mean_heading_rad
                + np.sqrt(max(1.0 - c_heading**2, 0.0)) * heading_noise
            )
        self._sync_user_velocity_vectors()
        self.mobility_update_count += 1

    def _integrate_user_positions(self):
        assigned_anchors = self.anchor_points[self.user_allocations]
        if self.users_static:
            self.user_coords = np.clip(
                assigned_anchors + self.user_offsets,
                0.0,
                1000.0,
            )
            return

        if self.gm_heading_mode == "reference_point_group":
            proposed_centers = (
                self.mobility_group_centers
                + self.mobility_group_velocities * self.micro_dt
            )
            if self.continuous_mission:
                fraction = min((self._current_time() + self.micro_dt) / self.mission_duration_s, 1.0)
                proposed_centers = (self.mission_routes_m[:, 0]
                                    + fraction * (self.mission_routes_m[:, 1] - self.mission_routes_m[:, 0]))
            center_hits = (proposed_centers < 0.0) | (proposed_centers > 1000.0)
            if np.any(center_hits):
                self.mobility_group_velocities[center_hits] *= -1.0
                affected_groups = np.any(center_hits, axis=1)
                reflected = self.mobility_group_velocities[affected_groups]
                self.mobility_group_headings[affected_groups] = np.arctan2(
                    reflected[:, 1],
                    reflected[:, 0],
                )
                proposed_centers = np.clip(proposed_centers, 0.0, 1000.0)
            self.mobility_group_centers = proposed_centers
            self.user_offsets = (
                self.user_offsets + self.user_local_velocities * self.micro_dt
            )
            proposed_coords = (
                self.mobility_group_centers[self.user_allocations]
                + self.user_offsets
            )
            boundary_hits = (
                (proposed_coords < 0.0) | (proposed_coords > 1000.0)
            )
            if self.continuous_mission:
                self.mission_user_boundary_hits += int(np.any(boundary_hits, axis=1).sum())
            if np.any(boundary_hits):
                self.user_local_velocities[boundary_hits] *= -1.0
                proposed_coords = np.clip(proposed_coords, 0.0, 1000.0)
                self.user_offsets = (
                    proposed_coords
                    - self.mobility_group_centers[self.user_allocations]
                )
            self.user_coords = proposed_coords
            total_velocities = (
                self.mobility_group_velocities[self.user_allocations]
                + self.user_local_velocities
            )
            self._set_user_velocity_vectors(total_velocities)
            return

        if self.user_zone_radius <= 0.0:
            proposed_coords = (
                self.user_coords + self.user_velocities * self.micro_dt
            )
            boundary_hits = (
                (proposed_coords < 0.0) | (proposed_coords > 1000.0)
            )
            if np.any(boundary_hits):
                self.user_velocities[boundary_hits] *= -1.0
                affected_users = np.any(boundary_hits, axis=1)
                reflected = self.user_velocities[affected_users]
                self.user_speeds[affected_users] = np.linalg.norm(
                    reflected,
                    axis=1,
                )
                self.user_headings[affected_users] = np.arctan2(
                    reflected[:, 1],
                    reflected[:, 0],
                )
            self.user_coords = np.clip(proposed_coords, 0.0, 1000.0)
            self.user_offsets = self.user_coords - assigned_anchors
            return

        proposed_offsets = self.user_offsets + self.user_velocities * self.micro_dt
        distances = np.linalg.norm(proposed_offsets, axis=1)
        outside = distances > self.user_zone_radius
        if np.any(outside):
            normals = proposed_offsets[outside] / np.maximum(
                distances[outside, None],
                1e-12,
            )
            velocities = self.user_velocities[outside]
            outward_speed = np.sum(velocities * normals, axis=1, keepdims=True)
            reflected = velocities - 2.0 * np.maximum(outward_speed, 0.0) * normals
            self.user_velocities[outside] = reflected
            self.user_speeds[outside] = np.linalg.norm(reflected, axis=1)
            self.user_headings[outside] = np.arctan2(
                reflected[:, 1],
                reflected[:, 0],
            )
            proposed_offsets[outside] = normals * self.user_zone_radius

        self.user_offsets = proposed_offsets
        self.user_coords = np.clip(
            assigned_anchors + self.user_offsets,
            0.0,
            1000.0,
        )

    def prepare_service_resource_frame(self):
        """Apply exogenous events before a 0.5 s service-frame action."""
        if self.macro_service_pause_steps_remaining > 0:
            raise RuntimeError("Cannot prepare a service frame during deployment.")
        if self.global_micro_step % self.resource_frame_steps != 0:
            raise RuntimeError("Service frames must start on resource-frame boundaries.")
        if self._resource_frame_prepared:
            return self.get_raw_state()

        self._update_gauss_markov_mobility()
        self._generate_traffic(
            append_to_ground_buffer=self.arrivals_during_deployment
        )
        self._resource_frame_prepared = True
        self.resource_frame_prepare_count += 1
        return self.get_raw_state()

    def _generate_traffic(self, append_to_ground_buffer=False):
        if not append_to_ground_buffer:
            self._discard_current_arrival_tasks()

        counts_high = np.zeros(self.M, dtype=np.float32)
        counts_low = np.zeros(self.M, dtype=np.float32)

        if self.global_micro_step % self.arrival_interval_steps == 0:
            counts_high = self.np_random.poisson(
                self.user_lambda_high * self.arrival_dt
            ).astype(np.float32)
            counts_low = self.np_random.poisson(
                self.user_lambda_low * self.arrival_dt
            ).astype(np.float32)
        if self.record_trajectory:
            self.arrival_high_trajectory.append(counts_high.copy())
            self.arrival_low_trajectory.append(counts_low.copy())

        self.current_user_associations = self.user_allocations.copy()
        high_bits = counts_high * self.user_task_size_high_mbits
        low_bits = counts_low * self.user_task_size_low_mbits
        A_t_users = high_bits + low_bits

        for user_idx in range(self.M):
            high_count = int(counts_high[user_idx])
            low_count = int(counts_low[user_idx])
            for _ in range(high_count):
                self._create_parent_task(
                    user_idx,
                    "high",
                    self.user_task_size_high_mbits[user_idx],
                    source_uav=self.current_user_associations[user_idx],
                )
            for _ in range(low_count):
                self._create_parent_task(
                    user_idx,
                    "low",
                    self.user_task_size_low_mbits[user_idx],
                    source_uav=self.current_user_associations[user_idx],
                )
        
        instantaneous_centroids = np.zeros((self.N, 2))
        for i in range(self.N):
            zone_idx = (self.user_allocations == i)
            zone_traffic = A_t_users[zone_idx]
            zone_coords = self.user_coords[zone_idx]
            
            total_traffic = np.sum(zone_traffic)
            if total_traffic > 1e-6:
                instantaneous_centroids[i] = np.sum(zone_coords * zone_traffic[:, None], axis=0) / total_traffic
            else:
                instantaneous_centroids[i] = np.mean(zone_coords, axis=0) if len(zone_coords) > 0 else self.anchor_points[i]

        self.instantaneous_centroids = instantaneous_centroids
        self.weighted_centroids = (
            self.centroid_ema_beta * self.weighted_centroids
            + (1.0 - self.centroid_ema_beta) * self.instantaneous_centroids
        )
                 
        A_t_total_high = np.zeros(self.N)
        A_t_total_low = np.zeros(self.N)
        active_users_total = np.zeros(self.N)
        for i in range(self.N):
            source_mask = self.current_user_associations == i
            A_t_total_high[i] = np.sum(high_bits[source_mask])
            A_t_total_low[i] = np.sum(low_bits[source_mask])
            active_users_total[i] = np.sum(
                (counts_high[source_mask] + counts_low[source_mask]) > 0
            )
            
        if append_to_ground_buffer:
            self.current_arrival_active_users += active_users_total
            self.A_t_high += A_t_total_high
            self.A_t_low += A_t_total_low
        else:
            self.current_arrival_active_users = active_users_total
            self.A_t_high = A_t_total_high
            self.A_t_low = A_t_total_low

        return {
            "high_per_uav": A_t_total_high.copy(),
            "low_per_uav": A_t_total_low.copy(),
            "active_users_per_uav": active_users_total.copy(),
        }

    def _clear_current_arrivals(self, discard_tasks=True):
        if discard_tasks:
            self._discard_current_arrival_tasks()
        else:
            self.current_arrival_tasks_by_uav = [[] for _ in range(self.N)]
        self.current_arrival_active_users = np.zeros(self.N)
        self.A_t_high = np.zeros(self.N)
        self.A_t_low = np.zeros(self.N)

    def _build_paused_service_info(self, flight_phase, arrival_event=None, compute_step=None):
        zeros_n = np.zeros(self.N, dtype=np.float32)
        arrival_event = arrival_event or {}
        arrival_high = np.asarray(
            arrival_event.get("high_per_uav", zeros_n),
            dtype=np.float32,
        )
        arrival_low = np.asarray(
            arrival_event.get("low_per_uav", zeros_n),
            dtype=np.float32,
        )
        arrival_active_users = np.asarray(
            arrival_event.get("active_users_per_uav", zeros_n),
            dtype=np.float32,
        )

        q_high_tot = np.sum(self.Q_loc_high) + np.sum(self.Q_tx_high)
        q_low_tot = np.sum(self.Q_loc_low) + np.sum(self.Q_tx_low)

        info = {
            'service_paused': True,
            'flight_phase': bool(flight_phase),
            'macro_transition_phase': True,
            'queue_length': q_high_tot + q_low_tot,
            'q_high_tot': q_high_tot,
            'q_low_tot': q_low_tot,
            'total_energy': 0.0,
            'energy_comp': 0.0,
            'energy_uplink': 0.0,
            'energy_trans': 0.0,
            'energy_comp_per_uav': zeros_n.copy(),
            'energy_uplink_per_uav': zeros_n.copy(),
            'energy_trans_per_uav': zeros_n.copy(),
            'energy_total_per_uav': zeros_n.copy(),
            'uav_battery_energy_per_uav': zeros_n.copy(),
            'uav_battery_service_energy': 0.0,
            'ground_uplink_energy': 0.0,
            'upload_delay_cost': 0.0,
            'upload_delay_cost_high': 0.0,
            'upload_delay_cost_low': 0.0,
            'upload_delay_cost_per_uav': zeros_n.copy(),
            'cpu_utilization_mean': 0.0,
            'cpu_utilization_per_uav': zeros_n.copy(),
            'cpu_high_utilization_per_uav': zeros_n.copy(),
            'cpu_low_utilization_per_uav': zeros_n.copy(),
            'cpu_saturation_ratio': 0.0,
            'cpu_alloc_mode': self.cpu_alloc_mode,
            'cpu_energy_model': 'macro_transition_paused',
            'kkt_v': float(self.kkt_v),
            'cpu_energy_price_per_uav': self.cpu_energy_prices.copy(),
            'f_high_per_uav': zeros_n.copy(),
            'f_low_per_uav': zeros_n.copy(),
            'f_total_per_uav': zeros_n.copy(),
            'cpu_power_cap_w': float(self.cpu_power_cap_w),
            'cpu_power_uncapped_w': 0.0,
            'cpu_power_allocated_w': 0.0,
            'cpu_power_cap_active': False,
            'compute_service_high_mbits': 0.0,
            'compute_service_low_mbits': 0.0,
            'compute_service_sensor_mbits': 0.0,
            'compute_service_controller_mbits': 0.0,
            'compute_service_vision_mbits': 0.0,
            'compute_service_high_cycles': 0.0,
            'compute_service_low_cycles': 0.0,
            'compute_service_sensor_cycles': 0.0,
            'compute_service_controller_cycles': 0.0,
            'compute_service_vision_cycles': 0.0,
            'arrival_high_sum': float(np.sum(arrival_high)),
            'arrival_low_sum': float(np.sum(arrival_low)),
            'arrival_high_per_uav': arrival_high.copy(),
            'arrival_low_per_uav': arrival_low.copy(),
            'arrival_total_per_uav': (arrival_high + arrival_low).copy(),
            'arrival_active_users_per_uav': arrival_active_users.copy(),
            'Q_loc_high_sum': np.sum(self.Q_loc_high),
            'Q_loc_low_sum': np.sum(self.Q_loc_low),
            'Q_tx_high_sum': np.sum(self.Q_tx_high),
            'Q_tx_low_sum': np.sum(self.Q_tx_low),
            'Q_loc_total': np.sum(self.Q_loc_high) + np.sum(self.Q_loc_low),
            'Q_tx_total': np.sum(self.Q_tx_high) + np.sum(self.Q_tx_low),
        }
        if compute_step is not None:
            (f_high, f_low, f_total, energy_model, served_high, served_low,
             energy, by_priority, by_type) = compute_step
            # service_paused retains its communication/decision-pause meaning.
            info.update({
                'compute_service_paused': False,
                'total_energy': float(np.sum(energy)),
                'energy_comp': float(np.sum(energy)),
                'energy_comp_per_uav': energy.copy(),
                'energy_total_per_uav': energy.copy(),
                'uav_battery_energy_per_uav': energy.copy(),
                'uav_battery_service_energy': float(np.sum(energy)),
                'cpu_utilization_mean': np.mean(f_total / np.maximum(self.F_max_array, 1e-8)),
                'cpu_utilization_per_uav': (f_total / np.maximum(self.F_max_array, 1e-8)).copy(),
                'cpu_high_utilization_per_uav': (f_high / np.maximum(self.F_max_array, 1e-8)).copy(),
                'cpu_low_utilization_per_uav': (f_low / np.maximum(self.F_max_array, 1e-8)).copy(),
                'cpu_saturation_ratio': float(np.mean(f_total >= 0.999 * np.maximum(self.F_max_array, 1e-8))),
                'cpu_energy_model': energy_model,
                'cpu_power_per_uav': (self.kappa * np.power(f_total, 3.0)).copy(),
                'f_high_per_uav': f_high.copy(),
                'f_low_per_uav': f_low.copy(),
                'f_total_per_uav': f_total.copy(),
                'cpu_power_uncapped_w': float(self.last_cpu_power_uncapped_w),
                'cpu_power_allocated_w': float(self.last_cpu_power_allocated_w),
                'cpu_power_cap_active': bool(self.last_cpu_power_cap_active),
                'compute_service_per_uav_mbits': (served_high + served_low).copy(),
                'tx_service_matrix_mbits': np.zeros((self.N, self.N)),
                'uplink_service_mbits_per_user': np.zeros(self.M),
            })
            for name, amount in {**by_priority, **by_type}.items():
                info[f'compute_service_{name}_mbits'] = amount
                info[f'compute_service_{name}_cycles'] = amount * 1e6 * self.C
        if self.wireless_model == "psd_per_uav_budget":
            # Query helpers do not imply service while deployment pauses radio.
            for name in ("u2u_link_bandwidth_hz", "u2u_link_power_w", "u2u_link_noise_w",
                         "u2u_rate_bps", "u2u_active_time_s", "u2u_link_energy_j",
                         "tx_service_matrix_mbits"):
                info[name] = np.zeros((self.N, self.N), dtype=np.float64)
            info["wireless_model"] = self.wireless_model
            info["u2u_quality_semantics"] = "fixed_reference_not_actual_efficiency"
            self.last_u2u_link_bandwidth_matrix_hz.fill(0.0)
        info.update(self.get_task_delay_stats())
        return info

    def _advance_macro_transition_step(self):
        arrival_event = None
        if (
            self.arrivals_during_deployment
            and self.global_micro_step % self.resource_frame_steps == 0
        ):
            self._update_gauss_markov_mobility()
            arrival_event = self._generate_traffic(
                append_to_ground_buffer=True
            )
            self.deployment_arrival_prepare_count += 1

        flight_phase = self.macro_fly_steps_remaining > 0
        if flight_phase:
            delta_x_uav = (
                self.current_macro_v
                * np.cos(self.current_macro_theta)
                * self.micro_dt
            )
            delta_y_uav = (
                self.current_macro_v
                * np.sin(self.current_macro_theta)
                * self.micro_dt
            )
            self.uav_coords[:, 0] += delta_x_uav
            self.uav_coords[:, 1] += delta_y_uav
            self.uav_coords = np.clip(self.uav_coords, 0.0, 1000.0)
            self.macro_fly_steps_remaining -= 1

        if self.arrivals_during_deployment:
            self._integrate_user_positions()

        compute_step = self._serve_compute_step() if self.compute_during_deployment else None
        self.macro_service_pause_steps_remaining -= 1
        paused_info = self._build_paused_service_info(
            flight_phase,
            arrival_event=arrival_event,
            compute_step=compute_step,
        )
        self.global_micro_step += 1

        self.record_trajectory_snapshot()
        return self.get_raw_state(), paused_info

    def _queue_to_frequency_need(self, queue_mbits):
        return np.maximum(queue_mbits, 0.0) * self.C * 1e6 / max(self.micro_dt, 1e-12)

    def _priority_kkt_frequency_pair(self, q_high, q_low, f_max):
        """Priority-aware KKT allocation for virtual high/low CPU slices."""
        eps = 1e-30
        v = max(float(self.kkt_v), 1e-12)
        coeff = 3.0 * v * max(float(self.kappa), eps)

        gains = np.asarray(
            [
                self.w_D_high * max(float(q_high), 0.0) / (self.C * 1e6),
                self.w_D_low * max(float(q_low), 0.0) / (self.C * 1e6),
            ],
            dtype=np.float64,
        )
        demand_caps = np.asarray(
            [
                self._queue_to_frequency_need(q_high),
                self._queue_to_frequency_need(q_low),
            ],
            dtype=np.float64,
        )
        demand_caps = np.minimum(np.maximum(demand_caps, 0.0), max(float(f_max), 0.0))

        def candidate(lam):
            values = np.sqrt(np.maximum(gains - lam, 0.0) / coeff)
            return np.minimum(values, demand_caps)

        freqs = candidate(0.0)
        if float(np.sum(freqs)) <= float(f_max):
            return float(freqs[0]), float(freqs[1])

        lo = 0.0
        hi = float(np.max(gains))
        for _ in range(self.kkt_bisection_iters):
            mid = 0.5 * (lo + hi)
            freqs = candidate(mid)
            if float(np.sum(freqs)) > float(f_max):
                lo = mid
            else:
                hi = mid

        freqs = candidate(hi)
        total = float(np.sum(freqs))
        if total > float(f_max) and total > 1e-12:
            freqs *= float(f_max) / total
        return float(freqs[0]), float(freqs[1])

    def set_cpu_energy_prices(self, prices):
        """Set non-negative per-UAV prices used by priced MaxWeight DVFS."""
        values = np.asarray(prices, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(self.N, float(values))
        elif values.shape == (1,):
            values = np.full(self.N, float(values[0]))
        if values.shape != (self.N,):
            raise ValueError(
                "CPU energy prices must be scalar or have shape "
                f"({self.N},), got {values.shape}."
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("CPU energy prices must be finite and non-negative.")
        self.cpu_energy_prices = values.copy()

    def set_external_cpu_frequencies(
        self,
        f_high_hz,
        f_low_hz,
        duration_steps=None,
    ):
        """Install a finite-duration CPU allocation from a reference solver.

        Learned policies do not call this method.  It exists so a baseline
        that jointly optimizes routing and CPU allocation can be executed
        without pretending that the environment's deterministic CPU rule is
        the CPU variable in the reference paper.
        """

        f_high = np.asarray(f_high_hz, dtype=np.float64).reshape(-1)
        f_low = np.asarray(f_low_hz, dtype=np.float64).reshape(-1)
        if f_high.shape != (self.N,) or f_low.shape != (self.N,):
            raise ValueError(
                "External CPU frequencies must each have shape "
                f"({self.N},)."
            )
        if (
            not np.all(np.isfinite(f_high))
            or not np.all(np.isfinite(f_low))
            or np.any(f_high < 0.0)
            or np.any(f_low < 0.0)
        ):
            raise ValueError(
                "External CPU frequencies must be finite and non-negative."
            )
        total = f_high + f_low
        scale = np.minimum(
            1.0,
            np.divide(
                self.F_max_array,
                np.maximum(total, 1e-12),
            ),
        )
        self._external_cpu_frequency_override = (
            f_high * scale,
            f_low * scale,
        )
        if duration_steps is None:
            duration_steps = self.resource_frame_steps
        self._external_cpu_override_steps_remaining = max(
            int(duration_steps),
            1,
        )

    def clear_external_cpu_frequencies(self):
        self._external_cpu_frequency_override = None
        self._external_cpu_override_steps_remaining = 0

    def _priced_maxweight_frequency_pair(
        self,
        q_high,
        q_low,
        f_max,
        energy_price,
    ):
        """Solve the unified-cubic, queue-weighted DVFS subproblem exactly.

        For one UAV the per-frame problem is

            max sum_c w_c (Q_c / Q_ref) (f_c / F_max)
                - (lambda / P_budget) * kappa * (sum_c f_c)^3,

        with ``Q_ref`` equal to one macro service window of compute capacity.
        ``energy_price`` is therefore the normalized dual price
        ``lambda / P_budget`` supplied by the wrapper, not an undocumented
        instantaneous CPU cap. For a fixed total frequency, assigning
        frequency in descending marginal queue-pressure order is optimal.
        The remaining scalar objective is concave and piecewise smooth, so
        its stationary point and segment boundaries are sufficient
        candidates.
        """
        f_max = max(float(f_max), 0.0)
        price = max(float(energy_price), 0.0)
        service_duration_s = max(
            float(self.macro_dt - self.deployment_dt),
            float(self.micro_dt),
        )
        queue_reference_mbits = max(
            f_max * service_duration_s / (self.C * 1e6),
            1e-12,
        )
        frequency_reference_hz = max(f_max, 1e-12)
        gains = np.asarray(
            [
                self.w_D_high
                * max(float(q_high), 0.0)
                / (queue_reference_mbits * frequency_reference_hz),
                self.w_D_low
                * max(float(q_low), 0.0)
                / (queue_reference_mbits * frequency_reference_hz),
            ],
            dtype=np.float64,
        )
        demand_caps = np.minimum(
            np.asarray(
                [
                    self._queue_to_frequency_need(q_high),
                    self._queue_to_frequency_need(q_low),
                ],
                dtype=np.float64,
            ),
            f_max,
        )
        demand_caps = np.maximum(demand_caps, 0.0)
        max_total = min(float(np.sum(demand_caps)), f_max)
        if max_total <= 1e-12 or float(np.max(gains)) <= 0.0:
            return 0.0, 0.0

        order = np.argsort(-gains, kind="stable")

        def allocate(total_frequency):
            remaining = float(np.clip(total_frequency, 0.0, max_total))
            allocation = np.zeros(2, dtype=np.float64)
            for class_index in order:
                allocated = min(float(demand_caps[class_index]), remaining)
                allocation[class_index] = allocated
                remaining -= allocated
                if remaining <= 1e-12:
                    break
            return allocation

        candidates = {0.0, max_total}
        segment_lower = 0.0
        denominator = 3.0 * price * max(float(self.kappa), 1e-30)
        for class_index in order:
            segment_upper = min(
                segment_lower + float(demand_caps[class_index]),
                max_total,
            )
            candidates.add(segment_lower)
            candidates.add(segment_upper)
            if segment_upper > segment_lower and denominator > 0.0:
                stationary = np.sqrt(
                    max(float(gains[class_index]), 0.0) / denominator
                )
                candidates.add(
                    float(np.clip(stationary, segment_lower, segment_upper))
                )
            segment_lower = segment_upper
            if segment_lower >= max_total - 1e-12:
                break

        best_allocation = np.zeros(2, dtype=np.float64)
        best_value = 0.0
        best_total = 0.0
        for candidate_total in sorted(candidates):
            allocation = allocate(candidate_total)
            total = float(np.sum(allocation))
            value = float(
                np.dot(gains, allocation)
                - price * self.kappa * total**3
            )
            if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12 and total < best_total
            ):
                best_value = value
                best_total = total
                best_allocation = allocation

        return float(best_allocation[0]), float(best_allocation[1])

    def _allocate_cpu_frequencies(self):
        f_high = np.zeros(self.N, dtype=np.float64)
        f_low = np.zeros(self.N, dtype=np.float64)
        energy_model = "total_cubic"

        if (
            self._external_cpu_frequency_override is not None
            and self._external_cpu_override_steps_remaining > 0
        ):
            f_high, f_low = (
                np.asarray(values, dtype=np.float64).copy()
                for values in self._external_cpu_frequency_override
            )
            self._external_cpu_override_steps_remaining -= 1
            if self._external_cpu_override_steps_remaining <= 0:
                self._external_cpu_frequency_override = None
            f_total = f_high + f_low
            cpu_power = float(
                self.kappa * np.sum(np.power(f_total, 3.0))
            )
            self.last_cpu_power_uncapped_w = cpu_power
            self.last_cpu_power_allocated_w = cpu_power
            self.last_cpu_power_cap_active = False
            return f_high, f_low, f_total, energy_model

        if self.cpu_alloc_mode == "priced_maxweight":
            for i in range(self.N):
                f_high[i], f_low[i] = self._priced_maxweight_frequency_pair(
                    self.Q_loc_high[i],
                    self.Q_loc_low[i],
                    self.F_max_array[i],
                    self.cpu_energy_prices[i],
                )
            f_total = f_high + f_low
            cpu_power = float(self.kappa * np.sum(np.power(f_total, 3.0)))
            self.last_cpu_power_uncapped_w = cpu_power
            self.last_cpu_power_allocated_w = cpu_power
            self.last_cpu_power_cap_active = False
            return f_high, f_low, f_total, energy_model

        if self.cpu_alloc_mode == "legacy_kkt":
            weighted_Q = (
                self.w_D_high * self.Q_loc_high
                + self.w_D_low * self.Q_loc_low
            )
            base_val = (
                weighted_Q * self.micro_dt
                / (3 * self.w_E * self.kappa * self.C)
            )
            f_total = np.clip(
                np.sqrt(np.maximum(base_val, 0.0)),
                0.0,
                self.F_max_array,
            )
            f_high = 0.8 * f_total
            f_low = 0.2 * f_total
            return f_high, f_low, f_total, energy_model

        if self.cpu_alloc_mode == "fixed_demand":
            need_high = self._queue_to_frequency_need(self.Q_loc_high)
            need_low = self._queue_to_frequency_need(self.Q_loc_low)
            need_total = need_high + need_low
            f_total = np.minimum(
                need_total,
                self.F_max_array.astype(np.float64),
            )
            scale = np.divide(
                f_total,
                np.maximum(need_total, 1e-12),
                out=np.zeros_like(f_total),
                where=need_total > 1e-12,
            )
            f_high = need_high * scale
            f_low = need_low * scale
            return f_high, f_low, f_total, energy_model

        if self.cpu_alloc_mode == "priority_demand":
            need_high = self._queue_to_frequency_need(self.Q_loc_high)
            need_low = self._queue_to_frequency_need(self.Q_loc_low)
            cap = self.F_max_array.astype(np.float64)
            f_high = np.minimum(need_high, cap)
            remaining = np.maximum(cap - f_high, 0.0)
            f_low = np.minimum(need_low, remaining)
            f_high, f_low = self._project_priority_cpu_power(
                f_high,
                f_low,
            )
            f_total = f_high + f_low
            return f_high, f_low, f_total, energy_model

        if self.cpu_alloc_mode in ("fixed_full", "fixed_average"):
            util = 1.0 if self.cpu_alloc_mode == "fixed_full" else self.fixed_cpu_util
            util = float(np.clip(util, 0.0, 1.0))
            f_total = util * self.F_max_array.astype(np.float64)
            f_total = np.minimum(
                f_total,
                self._queue_to_frequency_need(self.Q_loc_high + self.Q_loc_low),
            )
            f_high = 0.8 * f_total
            f_low = 0.2 * f_total
            return f_high, f_low, f_total, energy_model

        energy_model = "separable_cubic"
        for i in range(self.N):
            f_high[i], f_low[i] = self._priority_kkt_frequency_pair(
                self.Q_loc_high[i],
                self.Q_loc_low[i],
                self.F_max_array[i],
            )
        f_total = f_high + f_low
        return f_high, f_low, f_total, energy_model

    def _project_priority_cpu_power(self, f_high, f_low):
        """Project priority-demand frequencies onto a system CPU-power cap.

        Low-priority frequency is reduced first. If the high-priority demand
        alone exceeds the cap, all high-priority frequencies are scaled by
        the common cubic-power factor. This is a deterministic feasibility
        projection shared by every compared routing/trajectory method.
        """
        f_high = np.maximum(
            np.asarray(f_high, dtype=np.float64),
            0.0,
        ).copy()
        f_low = np.maximum(
            np.asarray(f_low, dtype=np.float64),
            0.0,
        ).copy()
        uncapped_power = float(
            self.kappa * np.sum(np.power(f_high + f_low, 3.0))
        )
        self.last_cpu_power_uncapped_w = uncapped_power
        self.last_cpu_power_cap_active = False

        cap_w = float(self.cpu_power_cap_w)
        if cap_w <= 0.0 or uncapped_power <= cap_w + 1e-9:
            self.last_cpu_power_allocated_w = uncapped_power
            return f_high, f_low

        high_power = float(self.kappa * np.sum(np.power(f_high, 3.0)))
        if high_power >= cap_w:
            scale = (cap_w / max(high_power, 1e-30)) ** (1.0 / 3.0)
            f_high *= float(np.clip(scale, 0.0, 1.0))
            f_low.fill(0.0)
        else:
            lo = 0.0
            hi = 1.0
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                power = float(
                    self.kappa
                    * np.sum(np.power(f_high + mid * f_low, 3.0))
                )
                if power > cap_w:
                    hi = mid
                else:
                    lo = mid
            f_low *= lo

        allocated_power = float(
            self.kappa * np.sum(np.power(f_high + f_low, 3.0))
        )
        self.last_cpu_power_allocated_w = allocated_power
        self.last_cpu_power_cap_active = True
        return f_high, f_low

    def _get_u2u_spectral_efficiency(self):
        if self.wireless_model == "psd_per_uav_budget":
            return self.compute_u2u_reference_quality()
        if (
            self._u2u_spectral_cache_coords is None
            or not np.array_equal(self.uav_coords, self._u2u_spectral_cache_coords)
        ):
            diff = self.uav_coords[:, None, :] - self.uav_coords[None, :, :]
            dist_2d_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dist_2d_sq, np.inf)

            channel_gain = self.beta_0 / np.maximum(dist_2d_sq, 1e-8)
            np.fill_diagonal(channel_gain, 0.0)

            spectral_efficiency = np.log2(
                1.0 + (self.p_u2u * channel_gain) / self.noise_power
            )
            np.fill_diagonal(spectral_efficiency, 0.0)

            self._u2u_spectral_cache_coords = self.uav_coords.copy()
            self._u2u_spectral_cache_result = spectral_efficiency

        return self._u2u_spectral_cache_result

    def _serve_compute_step(self):
        """Serve already-admitted compute chunks with the existing CPU rules."""
        f_high, f_low, f_total, energy_model = self._allocate_cpu_frequencies()
        by_priority = {"high": 0.0, "low": 0.0}
        by_type = {name: 0.0 for name in self.device_type_names}
        served_high = self._serve_compute_task_queues(
            (f_high * self.micro_dt) / (self.C * 1e6), "high",
            service_by_priority=by_priority, service_by_type=by_type,
        )
        served_low = self._serve_compute_task_queues(
            (f_low * self.micro_dt) / (self.C * 1e6), "low",
            service_by_priority=by_priority, service_by_type=by_type,
        )
        if energy_model == "separable_cubic":
            energy = self.kappa * (f_high**3 + f_low**3) * self.micro_dt
        else:
            energy = self.kappa * f_total**3 * self.micro_dt
        return (f_high, f_low, f_total, energy_model, served_high, served_low,
                energy, by_priority, by_type)

    def micro_step(self, split_ratios_high, split_ratios_low):
        """
        Discrete arrivals + explicit upload completion events + fluid UAV-side queues.
        """
        transition_phase = self.macro_service_pause_steps_remaining > 0
        if transition_phase and self.pause_service_during_flight:
            return self._advance_macro_transition_step()
        flight_phase = self.macro_fly_steps_remaining > 0

        if (
            self.global_micro_step % self.resource_frame_steps == 0
            and not self._resource_frame_prepared
        ):
            self.prepare_service_resource_frame()

        arrival_high_this_step = self.A_t_high.copy()
        arrival_low_this_step = self.A_t_low.copy()
        arrival_active_users_this_step = self.current_arrival_active_users.copy()

        arrived_task_ids_by_uav = self._take_completed_upload_tasks()
        self._admit_current_arrival_tasks_to_uplink()
        uplink_metrics = self._serve_uplink_task_queues()
        self._clear_current_arrivals(discard_tasks=False)

        if not self.continuous_mission:
            self._integrate_user_positions()

            if self.macro_fly_steps_remaining > 0:
                delta_x_uav = self.current_macro_v * np.cos(self.current_macro_theta) * self.micro_dt
                delta_y_uav = self.current_macro_v * np.sin(self.current_macro_theta) * self.micro_dt
                self.uav_coords[:, 0] += delta_x_uav
                self.uav_coords[:, 1] += delta_y_uav
                self.macro_fly_steps_remaining -= 1
            self.uav_coords = np.clip(self.uav_coords, 0.0, 1000.0)

        self._route_arrived_parent_tasks(
            arrived_task_ids_by_uav,
            split_ratios_high,
            split_ratios_low,
        )

        (f_high_array, f_low_array, f_opt_array, cpu_energy_model,
         actual_exe_high, actual_exe_low, energy_comp_per_uav,
         compute_service_priority_mbits, compute_service_type_mbits) = self._serve_compute_step()
        total_E_comp = np.sum(energy_comp_per_uav)

        tx_demand_mask = (
            np.asarray(self.Q_tx_high, dtype=np.float64)
            + np.asarray(self.Q_tx_low, dtype=np.float64)
        ) > 1e-12
        np.fill_diagonal(tx_demand_mask, False)
        if self.wireless_model == "psd_per_uav_budget":
            resources = self.allocate_u2u_resources(tx_demand_mask)
            rate_bps = resources["u2u_rate_bps"]
            u2u_link_bandwidth_hz = resources["u2u_link_bandwidth_hz"]
        else:
            rate_bps, u2u_link_bandwidth_hz = self.compute_u2u_rate_bps(
                active_link_mask=tx_demand_mask,
                return_bandwidth=True,
            )
        self.last_u2u_link_bandwidth_matrix_hz = (
            u2u_link_bandwidth_hz.copy()
        )
        mu_trans_mbits = (rate_bps * self.micro_dt) / 1e6

        actual_trans_high = self._serve_tx_task_queues(
            mu_trans_mbits,
            "high",
        )

        remaining_mu_trans = mu_trans_mbits - actual_trans_high
        actual_trans_low = self._serve_tx_task_queues(
            remaining_mu_trans,
            "low",
        )

        upload_delay_cost_per_uav = np.asarray(
            uplink_metrics["upload_delay_cost_per_uav"],
            dtype=np.float32,
        )
        energy_uplink_per_uav = np.asarray(
            uplink_metrics["energy_per_uav"],
            dtype=np.float32,
        )
        upload_delay_cost_high = float(
            uplink_metrics["upload_delay_cost_high"]
        )
        upload_delay_cost_low = float(
            uplink_metrics["upload_delay_cost_low"]
        )
        total_E_uplink = float(np.sum(energy_uplink_per_uav))
        upload_delay_cost = 5.0 * upload_delay_cost_high + upload_delay_cost_low

        actual_trans_total = actual_trans_high + actual_trans_low
        u2u_active_time_s = np.divide(
            actual_trans_total * 1e6,
            rate_bps,
            out=np.zeros_like(actual_trans_total, dtype=np.float64),
            where=rate_bps > (0.0 if self.wireless_model == "psd_per_uav_budget" else 1e-12),
        )
        if self.wireless_model == "psd_per_uav_budget":
            if (np.any((actual_trans_total > 0) & (rate_bps <= 0))
                    or np.any(u2u_active_time_s > self.micro_dt + 1e-9)
                    or np.any(u2u_active_time_s < 0)):
                raise AssertionError("U2U service exceeds the allocated physical-step capacity.")
            link_energy = resources["u2u_link_power_w"] * u2u_active_time_s
            energy_trans_per_uav = link_energy.sum(axis=1)
        else:
            energy_trans_per_uav = np.sum(
                u2u_active_time_s * self.p_u2u,
                axis=1,
            ).astype(np.float32)
        total_E_trans = float(np.sum(energy_trans_per_uav))
        if self.continuous_mission:
            # All services used step-start geometry. Newly uploaded work was
            # excluded by the ready-list snapshot; U2U completion follows CPU.
            self._integrate_user_positions()
            if self.macro_fly_steps_remaining > 0:
                self.uav_coords += self.current_macro_velocity_xy * self.micro_dt
                self.macro_fly_steps_remaining -= 1
                if self.macro_fly_steps_remaining == 0:
                    # Remove integration roundoff, not an additional motion.
                    if not np.allclose(self.uav_coords, self.current_macro_target, atol=1e-8, rtol=0):
                        raise AssertionError("Continuous displacement integration mismatch")
                    self.uav_coords = self.current_macro_target.copy()
        uav_battery_energy_per_uav = (
            energy_comp_per_uav + energy_trans_per_uav
        )

        q_high_tot = np.sum(self.Q_loc_high) + np.sum(self.Q_tx_high)
        q_low_tot = np.sum(self.Q_loc_low) + np.sum(self.Q_tx_low)
        total_energy = total_E_comp + total_E_uplink + total_E_trans
        compute_service_per_uav_mbits = actual_exe_high + actual_exe_low
        tx_service_matrix_mbits = actual_trans_high + actual_trans_low
        coop_total_flow = float(np.sum(self.coop_flow_mbits))
        coop_cross_flow = float(
            np.sum(self.coop_flow_mbits) - np.trace(self.coop_flow_mbits)
        )
        coop_ratio = coop_cross_flow / max(coop_total_flow, 1e-12)
        compute_service_priority_cycles = {
            key: value * 1e6 * self.C
            for key, value in compute_service_priority_mbits.items()
        }
        compute_service_type_cycles = {
            key: value * 1e6 * self.C
            for key, value in compute_service_type_mbits.items()
        }
        q_tx_high_out = np.sum(self.Q_tx_high, axis=1)
        q_tx_low_out = np.sum(self.Q_tx_low, axis=1)
        q_tx_high_in = np.sum(self.Q_tx_high, axis=0)
        q_tx_low_in = np.sum(self.Q_tx_low, axis=0)
        pressure_load = (
            self.w_D_high * (self.Q_loc_high + q_tx_high_out + q_tx_high_in)
            + self.w_D_low * (self.Q_loc_low + q_tx_low_out + q_tx_low_in)
        )
        service_capacity_mbits = (
            self.F_max_array.astype(np.float64) * self.micro_dt
        ) / max(self.C * 1e6, 1e-12)
        pressure_per_uav = pressure_load / np.maximum(
            service_capacity_mbits,
            1e-12,
        )
        pressure_imbalance = float(
            np.std(pressure_per_uav)
            / max(float(np.mean(pressure_per_uav)), 1e-12)
        )

        info = {
            'service_paused': False,
            'flight_phase': bool(flight_phase),
            'macro_transition_phase': bool(transition_phase),
            'queue_length': q_high_tot + q_low_tot,
            'q_high_tot': q_high_tot,
            'q_low_tot': q_low_tot,
            'total_energy': total_energy,
            'energy_comp': total_E_comp,
            'energy_uplink': total_E_uplink,
            'energy_trans': total_E_trans,
            'energy_comp_per_uav': energy_comp_per_uav.copy(),
            'energy_uplink_per_uav': energy_uplink_per_uav.copy(),
            'energy_trans_per_uav': energy_trans_per_uav.copy(),
            'energy_total_per_uav': (
                energy_comp_per_uav
                + energy_uplink_per_uav
                + energy_trans_per_uav
            ).copy(),
            # Ground-user uplink transmit energy is reported separately and
            # is not charged to an airborne battery constraint.
            'uav_battery_energy_per_uav': (
                uav_battery_energy_per_uav.copy()
            ),
            'uav_battery_service_energy': float(
                np.sum(uav_battery_energy_per_uav)
            ),
            'ground_uplink_energy': float(total_E_uplink),
            'upload_delay_cost': upload_delay_cost,
            'upload_delay_cost_high': upload_delay_cost_high,
            'upload_delay_cost_low': upload_delay_cost_low,
            'upload_delay_cost_per_uav': upload_delay_cost_per_uav.copy(),
            'uplink_service_mbits_per_user': np.asarray(
                uplink_metrics["service_mbits_per_user"],
                dtype=np.float32,
            ),
            'uplink_active_time_s_per_user': np.asarray(
                uplink_metrics["active_time_s_per_user"],
                dtype=np.float32,
            ),
            'uplink_rate_bps_per_user': np.asarray(
                uplink_metrics["effective_rate_bps_per_user"],
                dtype=np.float32,
            ),
            'uplink_completed_tasks_per_uav': np.asarray(
                uplink_metrics["completed_per_uav"],
                dtype=np.int64,
            ),
            'user_associations': self.current_user_associations.copy(),
            'u2u_rate_bps': rate_bps.astype(np.float64).copy(),
            'u2u_link_bandwidth_hz': (
                u2u_link_bandwidth_hz.copy()
            ),
            'u2u_active_time_s': u2u_active_time_s.copy(),
            'cpu_utilization_mean': np.mean(f_opt_array / np.maximum(self.F_max_array, 1e-8)),
            'cpu_utilization_per_uav': (f_opt_array / np.maximum(self.F_max_array, 1e-8)).copy(),
            'cpu_high_utilization_per_uav': (
                f_high_array / np.maximum(self.F_max_array, 1e-8)
            ).copy(),
            'cpu_low_utilization_per_uav': (
                f_low_array / np.maximum(self.F_max_array, 1e-8)
            ).copy(),
            'cpu_saturation_ratio': float(
                np.mean(f_opt_array >= 0.999 * np.maximum(self.F_max_array, 1e-8))
            ),
            'cpu_alloc_mode': self.cpu_alloc_mode,
            'cpu_energy_model': cpu_energy_model,
            'kkt_v': float(self.kkt_v),
            'cpu_energy_price_per_uav': self.cpu_energy_prices.copy(),
            'cpu_power_per_uav': (
                self.kappa * np.power(f_opt_array, 3.0)
            ).copy(),
            'f_high_per_uav': f_high_array.copy(),
            'f_low_per_uav': f_low_array.copy(),
            'f_total_per_uav': f_opt_array.copy(),
            'cpu_power_cap_w': float(self.cpu_power_cap_w),
            'cpu_power_uncapped_w': float(
                self.last_cpu_power_uncapped_w
            ),
            'cpu_power_allocated_w': float(
                self.last_cpu_power_allocated_w
            ),
            'cpu_power_cap_active': bool(
                self.last_cpu_power_cap_active
            ),
            'compute_service_high_mbits': compute_service_priority_mbits.get('high', 0.0),
            'compute_service_low_mbits': compute_service_priority_mbits.get('low', 0.0),
            'compute_service_high_cycles': compute_service_priority_cycles.get('high', 0.0),
            'compute_service_low_cycles': compute_service_priority_cycles.get('low', 0.0),
            'compute_service_per_uav_mbits': compute_service_per_uav_mbits.copy(),
            'tx_service_matrix_mbits': tx_service_matrix_mbits.copy(),
            'coop_flow_mbits': self.coop_flow_mbits.copy(),
            'coop_total_flow_mbits': coop_total_flow,
            'coop_cross_flow_mbits': coop_cross_flow,
            'coop_ratio': coop_ratio,
            'split_ratios_high': np.asarray(split_ratios_high, dtype=np.float32).copy(),
            'split_ratios_low': np.asarray(split_ratios_low, dtype=np.float32).copy(),
            'pressure_per_uav': pressure_per_uav.astype(np.float32),
            'pressure_imbalance': pressure_imbalance,
            'arrival_high_sum': np.sum(arrival_high_this_step),
            'arrival_low_sum': np.sum(arrival_low_this_step),
            'arrival_high_per_uav': arrival_high_this_step.copy(),
            'arrival_low_per_uav': arrival_low_this_step.copy(),
            'arrival_total_per_uav': (
                arrival_high_this_step + arrival_low_this_step
            ).copy(),
            'arrival_active_users_per_uav': arrival_active_users_this_step.copy(),
            'Q_loc_high_sum': np.sum(self.Q_loc_high),
            'Q_loc_low_sum': np.sum(self.Q_loc_low),
            'Q_tx_high_sum': np.sum(self.Q_tx_high),
            'Q_tx_low_sum': np.sum(self.Q_tx_low),
            'Q_loc_total': np.sum(self.Q_loc_high) + np.sum(self.Q_loc_low),
            'Q_tx_total': np.sum(self.Q_tx_high) + np.sum(self.Q_tx_low)
        }
        if self.wireless_model == "psd_per_uav_budget":
            info.update({"wireless_model": self.wireless_model,
                         "u2u_quality_semantics": "fixed_reference_not_actual_efficiency",
                         "u2u_link_power_w": resources["u2u_link_power_w"].copy(),
                         "u2u_link_noise_w": resources["u2u_link_noise_w"].copy(),
                         "u2u_link_energy_j": link_energy.copy()})
        for task_type in self.device_type_names:
            info[f'compute_service_{task_type}_mbits'] = (
                compute_service_type_mbits.get(task_type, 0.0)
            )
            info[f'compute_service_{task_type}_cycles'] = (
                compute_service_type_cycles.get(task_type, 0.0)
            )
        info.update(self.get_task_delay_stats())

        self.global_micro_step += 1
        if self.global_micro_step % self.resource_frame_steps == 0:
            self._resource_frame_prepared = False

        self.record_trajectory_snapshot()

        return self.get_raw_state(), info

    def _minimum_separation_along_joint_motion(self, start, delta):
        if self.N < 2:
            return np.inf
        minimum = np.inf
        for i in range(self.N):
            for j in range(i + 1, self.N):
                relative_start = start[i] - start[j]
                relative_delta = delta[i] - delta[j]
                denominator = float(np.dot(relative_delta, relative_delta))
                if denominator <= 1e-14:
                    fraction = 0.0
                else:
                    fraction = float(
                        np.clip(
                            -np.dot(relative_start, relative_delta)
                            / denominator,
                            0.0,
                            1.0,
                        )
                    )
                distance = np.linalg.norm(
                    relative_start + fraction * relative_delta
                )
                minimum = min(minimum, float(distance))
        return minimum

    def _project_joint_motion_to_safe_separation(self, start, delta):
        required = float(self.min_uav_separation_m)
        if self.N < 2 or required <= 0.0:
            return delta.copy(), 1.0
        if (
            self._minimum_separation_along_joint_motion(start, delta)
            >= required - 1e-9
        ):
            return delta.copy(), 1.0

        safe_scale = 1.0
        required_sq = required**2
        for i in range(self.N):
            for j in range(i + 1, self.N):
                relative_start = start[i] - start[j]
                relative_delta = delta[i] - delta[j]
                a = float(np.dot(relative_delta, relative_delta))
                b = 2.0 * float(np.dot(relative_start, relative_delta))
                c = float(np.dot(relative_start, relative_start)) - required_sq
                if a <= 1e-14:
                    continue
                discriminant = b * b - 4.0 * a * c
                if discriminant < 0.0:
                    continue
                root_term = np.sqrt(max(discriminant, 0.0))
                roots = (
                    (-b - root_term) / (2.0 * a),
                    (-b + root_term) / (2.0 * a),
                )
                entering_roots = [
                    root
                    for root in roots
                    if -1e-12 <= root <= 1.0 + 1e-12
                ]
                if entering_roots:
                    safe_scale = min(
                        safe_scale,
                        max(min(entering_roots), 0.0),
                    )

        safe_delta = delta * safe_scale
        if (
            self._minimum_separation_along_joint_motion(start, safe_delta)
            < required - 1e-7
        ):
            return np.zeros_like(delta), 0.0
        return safe_delta, float(safe_scale)

    def macro_step(self, macro_actions):
        """
        Fly-Hover protocol for one 10s macro slot.
        The action is interpreted as a desired 2D displacement [dx, dy] in
        meters. UAVs fly for 1s and then hover for 9s.
        """
        if self.continuous_mission:
            return self._mission_macro_step(macro_actions)
        macro_actions = np.asarray(macro_actions, dtype=np.float64)
        if macro_actions.shape != (self.N, 2):
            raise ValueError(
                f"macro_actions must have shape ({self.N}, 2)."
            )
        delta_xy = np.clip(macro_actions, -20.0, 20.0)
        t_fly = self.deployment_dt
        t_hover = self.macro_dt - self.deployment_dt

        current_coords = self.uav_coords.copy()
        requested_coords = np.clip(
            current_coords + delta_xy,
            0.0,
            1000.0,
        )
        requested_delta_xy = requested_coords - current_coords
        actual_delta_xy, separation_scale = (
            self._project_joint_motion_to_safe_separation(
                current_coords,
                requested_delta_xy,
            )
        )
        projected_coords = current_coords + actual_delta_xy

        # The joint displacement is scaled only when the paper's pairwise
        # safety-distance constraint would be violated along the flight leg.
        fly_distances = np.linalg.norm(actual_delta_xy, axis=1)
        fly_theta = np.where(
            fly_distances > 1e-8,
            np.arctan2(actual_delta_xy[:, 1], actual_delta_xy[:, 0]),
            self.current_macro_theta
        )
        fly_v = fly_distances / max(t_fly, 1e-8)

        centroid_distances = np.linalg.norm(projected_coords - self.weighted_centroids, axis=1)
        u2g_penalty = - np.sum(centroid_distances) / 1000.0
        anchor_distance_values = np.linalg.norm(projected_coords - self.anchor_points, axis=1)
        anchor_penalty = - np.sum(anchor_distance_values) / 300.0

        self.current_macro_v = fly_v
        self.current_macro_theta = fly_theta
        transition_steps = self.deployment_steps
        self.macro_fly_steps_remaining = (
            transition_steps if float(np.max(fly_distances)) > 1e-8 else 0
        )
        self.macro_service_pause_steps_remaining = (
            transition_steps if self.pause_service_during_flight else 0
        )
        self._resource_frame_prepared = False
        E_steady_fly = self._compute_tcom_flying_power(fly_v) * t_fly
        E_deployment_transient = self._compute_deployment_transient_energy(
            fly_v
        )
        E_fly = E_steady_fly + E_deployment_transient
        E_hover = self._compute_tcom_flying_power(np.zeros(self.N)) * t_hover
        E_macro_per_uav = E_fly + E_hover
        max_macro_flying_E = 3000.0
        h_flying_penalty = - (np.sum(E_fly) / max_macro_flying_E) + u2g_penalty + anchor_penalty
        E_macro_total = np.sum(E_macro_per_uav)

        return self.get_raw_state(), h_flying_penalty, {
            'E_flying': np.sum(E_fly),
            'E_steady_flying': np.sum(E_steady_fly),
            'E_deployment_transient': np.sum(E_deployment_transient),
            'E_hover': np.sum(E_hover),
            'E_macro_total': E_macro_total,
            'E_flying_per_uav': E_fly.copy(),
            'E_steady_flying_per_uav': E_steady_fly.copy(),
            'E_deployment_transient_per_uav': (
                E_deployment_transient.copy()
            ),
            'E_hover_per_uav': E_hover.copy(),
            'E_macro_per_uav': E_macro_per_uav.copy(),
            'u2g_penalty': u2g_penalty,
            'anchor_penalty': anchor_penalty,
            'macro_penalty': h_flying_penalty,
            'u2g_distances': centroid_distances.copy(),
            'anchor_distances': anchor_distance_values.copy(),
            'requested_uav_coords': requested_coords.copy(),
            'projected_uav_coords': projected_coords.copy(),
            'separation_projection_scale': float(separation_scale),
            'minimum_flight_separation_m': float(
                self._minimum_separation_along_joint_motion(
                    current_coords,
                    actual_delta_xy,
                )
            ),
            'min_uav_separation_m': float(self.min_uav_separation_m),
            'pause_service_during_flight': self.pause_service_during_flight,
            'macro_transition_steps': transition_steps,
            'macro_service_pause_steps': (
                self.macro_service_pause_steps_remaining
            ),
        }

    def _mission_macro_step(self, delta):
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != (self.N, 2) or not np.isfinite(delta).all():
            raise ValueError("Mission displacement must be a finite (N,2) matrix")
        if self.global_micro_step % self.micro_steps_per_macro:
            raise ValueError("Mission commands must start at a macro boundary")
        if self.macro_fly_steps_remaining:
            raise ValueError("Previous continuous macro is unfinished")
        delta = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True) / 20.0, 1.0)
        before = self.uav_coords.copy()
        requested = np.clip(before + delta, 0.0, 1000.0)
        actual, scale = self._project_joint_motion_to_safe_separation(before, requested - before)
        self.current_macro_target = before + actual
        self.current_macro_velocity_xy = actual / self.macro_dt
        self.current_macro_v = np.linalg.norm(self.current_macro_velocity_xy, axis=1)
        self.current_macro_theta = np.arctan2(actual[:, 1], actual[:, 0])
        self.macro_fly_steps_remaining = self.micro_steps_per_macro if np.any(actual) else 0
        self.macro_service_pause_steps_remaining = 0
        self._resource_frame_prepared = False
        energy = self._compute_tcom_flying_power(self.current_macro_v) * self.macro_dt
        hover = self._compute_tcom_flying_power(np.zeros(self.N)) * self.macro_dt
        info = {
            "motion_protocol": self.motion_protocol,
            "energy_model": "quasi_steady_no_acceleration_or_turn_cost",
            "E_macro_total": float(energy.sum()), "E_macro_per_uav": energy.copy(),
            "E_propulsion_increment": float((energy - hover).sum()),
            "E_propulsion_increment_per_uav": energy - hover,
            "E_steady_motion": float(energy.sum()), "E_deployment_transient": 0.0,
            "requested_uav_coords": requested, "projected_uav_coords": self.current_macro_target.copy(),
            "actual_displacement_m": actual, "velocity_xy_mps": self.current_macro_velocity_xy.copy(),
            "motion_duration_s": self.macro_dt, "macro_service_pause_steps": 0,
            "separation_projection_scale": float(scale),
            "minimum_flight_separation_m": float(self._minimum_separation_along_joint_motion(before, actual)),
        }
        return self.get_raw_state(), 0.0, info

    def render_trajectory(self, save_path=None, show=False):
        if not self.record_trajectory: return
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import os
        if save_path: os.makedirs(os.path.dirname(save_path), exist_ok=True)

        installed_fonts = {font.name for font in fm.fontManager.ttflist}
        for font_name in (
            'Microsoft YaHei',
            'SimHei',
            'Noto Sans CJK SC',
            'Source Han Sans SC',
        ):
            if font_name in installed_fonts:
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                break
        plt.rcParams['axes.unicode_minus'] = False
                
        uav_traj = np.array(self.uav_trajectory)
        cent_traj = np.array(self.centroid_trajectory)
        users_traj = np.array(self.users_trajectory)
        
        plt.figure(figsize=(9.2, 8.2))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for i in range(self.N):
            c = colors[i % len(colors)]
            
            # 画每个区域内真实独立用户的细碎游走轨迹
            zone_users_idx = np.where(self.user_allocations == i)[0]
            for u_idx in zone_users_idx:
                u_traj_x = users_traj[:, u_idx, 0]
                u_traj_y = users_traj[:, u_idx, 1]
                # 只给第一个贴图例标签防止重复
                l = f'区域{i+1}用户轨迹' if u_idx == zone_users_idx[0] else None
                # 加深地名用户的颜色，使得更容易看清
                plt.plot(u_traj_x, u_traj_y, color=c, linestyle=':', linewidth=1.2, alpha=0.6, label=l)
                # 画出最终落点小圆点
                plt.scatter(u_traj_x[-1], u_traj_y[-1], color=c, marker='o', s=12, alpha=0.8)
            
            # 画区域引力质心点游走
            plt.plot(cent_traj[:, i, 0], cent_traj[:, i, 1], color=c, linestyle='--', linewidth=1.3, alpha=0.8, label=f'区域{i+1}业务质心')
            plt.scatter(cent_traj[0, i, 0], cent_traj[0, i, 1], color=c, marker='X', s=40, edgecolors='black', zorder=5) 
            plt.scatter(cent_traj[-1, i, 0], cent_traj[-1, i, 1], color=c, marker='P', s=60, edgecolors='black', zorder=5)
            
            # 飞行动作 (进一步调细线条和起落标志)
            plt.plot(uav_traj[:, i, 0], uav_traj[:, i, 1], color=c, linestyle='-', linewidth=1.8, marker='o', markersize=2.4, alpha=0.9, label=f'无人机{i+1}轨迹')
            plt.scatter(uav_traj[0, i, 0], uav_traj[0, i, 1], color=c, marker='s', s=20, edgecolors='black', zorder=5) 
            plt.scatter(uav_traj[-1, i, 0], uav_traj[-1, i, 1], color=c, marker='*', s=80, edgecolors='black', zorder=5) 
            
        plt.scatter([], [], color='gray', marker='s', edgecolors='black', s=20, label='无人机起点')
        plt.scatter([], [], color='gray', marker='*', edgecolors='black', s=80, label='无人机终点')
        plt.scatter([], [], color='gray', marker='X', edgecolors='black', s=40, label='质心初始位置')
        plt.scatter([], [], color='gray', marker='P', edgecolors='black', s=60, label='质心最终位置')
        
        title = '无人机轨迹与业务质心' if self.users_static else '无人机轨迹与动态业务质心'
        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel('X坐标（m）', fontsize=11)
        plt.ylabel('Y坐标（m）', fontsize=11)
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            fontsize=8.2,
            columnspacing=1.2,
            handlelength=2.2,
        )
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show: plt.show()
        plt.close()
