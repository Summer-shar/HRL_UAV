import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import UAV_MEC_Env
from .routing import centered_stick_breaking_to_simplex
from .routing import project_rows_to_simplex


def normalized_power_excess(energy_ratio):
    """Return only normalized consumption above the power budget."""
    values = np.asarray(energy_ratio, dtype=np.float64)
    excess = np.maximum(values - 1.0, 0.0)
    if values.ndim == 0:
        return float(excess)
    return excess


def normalized_power_residual(energy_ratio):
    """Return the signed residual used by a long-term power constraint."""
    values = np.asarray(energy_ratio, dtype=np.float64)
    residual = values - 1.0
    if values.ndim == 0:
        return float(residual)
    return residual


class HRL_Wrapper(gym.Env):
    """
    RL 算法适配层。

    - [MAPPO UPGRADE] 宏观层采用参数共享的 MAPPO Actor，
      每架 UAV 使用自己的 local observation 独立决策。
    - [MAPPO UPGRADE] 宏观层 Critic 仍然接收全局拼接状态，
      保持 CTDE（Centralized Training, Decentralized Execution）架构。
    - [REWARD SHAPING] 微观层 SAC 的状态、奖励和熔断逻辑在此统一处理。
    """

    def __init__(
        self,
        num_uavs=3,
        num_users=39,
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
        micro_diag_bias=0.5,
        micro_action_scale=1.0,
        micro_action_encoding="projected_full",
        micro_stick_breaking_scale=4.0,
        micro_reward_scale=10.0,
        micro_state_mode="pressure",
        u2u_state_representation="nominal_rate",
        u2u_goal_representation="match_observation",
        u2u_contention_aware_pressure=None,
        u2u_prev_active_state=False,
        cpu_alloc_mode="priority_kkt",
        kkt_v=1.0,
        cpu_energy_prices=None,
        fixed_cpu_util=0.5,
        cpu_power_cap_w=0.0,
        load_scale=1.0,
        zone_loads=None,
        uav_initial_positions=None,
        macro_comm_weight=2.0,
        macro_comm_flow_ema=0.8,
        macro_flow_state=True,
        macro_load_state=False,
        macro_capacity_state=False,
        macro_u2g_cost_weight=2.0,
        macro_zone_ingress_weight=0.5,
        macro_action_mode="polar",
        macro_waypoint_radius_m=200.0,
        macro_waypoint_tolerance_m=0.5,
        macro_reward_mode="strict_global",
        strict_hierarchy_validation=False,
        pause_service_during_flight=True,
        arrivals_during_deployment=False,
        objective_mode="weighted",
        power_budget_w=0.0,
        energy_lagrange_multiplier=0.0,
        power_constraint_scope="system",
        uav_power_budgets_w=None,
        energy_lagrange_multipliers=None,
        energy_dual_state_norm=10.0,
        system_dual_state=False,
        users_static=False,
        gm_mean_speed_mps=0.5,
        gm_mean_heading_rad=np.pi / 4.0,
        gm_speed_memory=0.5,
        gm_heading_memory=0.5,
        gm_speed_noise_std=0.1,
        gm_heading_noise_std=0.1,
        user_zone_radius=80.0,
        min_uav_separation_m=5.0,
        gm_heading_mode="shared_mean",
        rpgm_group_speed_mps=0.35,
        rpgm_group_speed_memory=0.9,
        rpgm_group_speed_noise_std=0.04,
        rpgm_group_turn_std_rad_sqrt_s=0.12,
        rpgm_local_velocity_memory=0.9,
        rpgm_local_velocity_std_mps=0.18,
        rpgm_local_restore_rate_s2=0.0002,
        rpgm_local_speed_cap_mps=0.45,
        delay_objective_weight=0.7,
        energy_objective_weight=0.3,
    ):
        super(HRL_Wrapper, self).__init__()

        self.env = UAV_MEC_Env(
            num_uavs=num_uavs,
            num_users=num_users,
            record_trajectory=record_trajectory,
            traffic_profile=traffic_profile,
            traffic_scale=traffic_scale,
            arrival_dt=arrival_dt,
            resource_frame_dt=resource_frame_dt,
            deployment_dt=deployment_dt,
            deployment_transient_energy_mode=(
                deployment_transient_energy_mode
            ),
            uav_mass_kg=uav_mass_kg,
            uplink_bandwidth_mhz=uplink_bandwidth_mhz,
            u2u_bandwidth_mhz=u2u_bandwidth_mhz,
            cpu_alloc_mode=cpu_alloc_mode,
            kkt_v=kkt_v,
            cpu_energy_prices=cpu_energy_prices,
            fixed_cpu_util=fixed_cpu_util,
            cpu_power_cap_w=cpu_power_cap_w,
            load_scale=load_scale,
            zone_loads=zone_loads,
            uav_initial_positions=uav_initial_positions,
            pause_service_during_flight=pause_service_during_flight,
            arrivals_during_deployment=arrivals_during_deployment,
            users_static=users_static,
            gm_mean_speed_mps=gm_mean_speed_mps,
            gm_mean_heading_rad=gm_mean_heading_rad,
            gm_speed_memory=gm_speed_memory,
            gm_heading_memory=gm_heading_memory,
            gm_speed_noise_std=gm_speed_noise_std,
            gm_heading_noise_std=gm_heading_noise_std,
            user_zone_radius=user_zone_radius,
            min_uav_separation_m=min_uav_separation_m,
            gm_heading_mode=gm_heading_mode,
            rpgm_group_speed_mps=rpgm_group_speed_mps,
            rpgm_group_speed_memory=rpgm_group_speed_memory,
            rpgm_group_speed_noise_std=rpgm_group_speed_noise_std,
            rpgm_group_turn_std_rad_sqrt_s=(
                rpgm_group_turn_std_rad_sqrt_s
            ),
            rpgm_local_velocity_memory=rpgm_local_velocity_memory,
            rpgm_local_velocity_std_mps=rpgm_local_velocity_std_mps,
            rpgm_local_restore_rate_s2=rpgm_local_restore_rate_s2,
            rpgm_local_speed_cap_mps=rpgm_local_speed_cap_mps,
        )

        self.N = num_uavs
        self.micro_reward_scale = float(micro_reward_scale)
        if (
            not np.isfinite(self.micro_reward_scale)
            or self.micro_reward_scale <= 0.0
        ):
            raise ValueError("micro_reward_scale must be finite and positive.")
        self.micro_diag_bias = float(micro_diag_bias)
        self.micro_action_scale = float(micro_action_scale)
        self.micro_action_encoding = str(micro_action_encoding).lower()
        if self.micro_action_encoding not in (
            "projected_full",
            "stick_breaking",
        ):
            raise ValueError(
                "micro_action_encoding must be 'projected_full' or "
                "'stick_breaking'."
            )
        self.micro_stick_breaking_scale = float(
            micro_stick_breaking_scale
        )
        if (
            not np.isfinite(self.micro_stick_breaking_scale)
            or self.micro_stick_breaking_scale <= 0.0
        ):
            raise ValueError(
                "micro_stick_breaking_scale must be finite and positive."
            )
        self.micro_holding_weight = 1.0
        self.micro_queue_weight = 0.0
        self.micro_energy_weight = 1.0
        self.force_local_routing = False
        self.objective_mode = str(objective_mode).lower()
        if self.objective_mode not in ("weighted", "constrained_delay"):
            raise ValueError(
                "objective_mode must be 'weighted' or 'constrained_delay'."
            )
        self.power_budget_w = float(power_budget_w)
        self.power_constraint_scope = str(power_constraint_scope).lower()
        if self.power_constraint_scope not in ("system", "per_uav"):
            raise ValueError(
                "power_constraint_scope must be 'system' or 'per_uav'."
            )
        self.system_dual_state = bool(system_dual_state)
        self.energy_lagrange_multiplier = max(
            float(energy_lagrange_multiplier),
            0.0,
        )
        if uav_power_budgets_w is None:
            if self.power_constraint_scope == "per_uav" and self.power_budget_w > 0.0:
                uav_power_budgets_w = self.power_budget_w / max(num_uavs, 1)
            else:
                uav_power_budgets_w = 0.0
        budgets = np.asarray(uav_power_budgets_w, dtype=np.float64)
        if budgets.ndim == 0:
            budgets = np.full(num_uavs, float(budgets))
        elif budgets.shape == (1,):
            budgets = np.full(num_uavs, float(budgets[0]))
        if budgets.shape != (num_uavs,):
            raise ValueError(
                "uav_power_budgets_w must be scalar or have shape "
                f"({num_uavs},), got {budgets.shape}."
            )
        if not np.all(np.isfinite(budgets)) or np.any(budgets < 0.0):
            raise ValueError("uav_power_budgets_w must be finite and non-negative.")
        self.uav_power_budgets_w = budgets.copy()

        if energy_lagrange_multipliers is None:
            energy_lagrange_multipliers = self.energy_lagrange_multiplier
        duals = np.asarray(energy_lagrange_multipliers, dtype=np.float64)
        if duals.ndim == 0:
            duals = np.full(num_uavs, float(duals))
        elif duals.shape == (1,):
            duals = np.full(num_uavs, float(duals[0]))
        if duals.shape != (num_uavs,):
            raise ValueError(
                "energy_lagrange_multipliers must be scalar or have shape "
                f"({num_uavs},), got {duals.shape}."
            )
        if not np.all(np.isfinite(duals)) or np.any(duals < 0.0):
            raise ValueError(
                "energy_lagrange_multipliers must be finite and non-negative."
            )
        self.energy_lagrange_multipliers = duals.copy()
        self.energy_dual_state_norm = max(float(energy_dual_state_norm), 1e-8)
        if self.objective_mode == "constrained_delay":
            if self.power_constraint_scope == "system":
                if self.power_budget_w <= 0.0:
                    raise ValueError(
                        "system constrained_delay requires power_budget_w > 0."
                    )
            elif np.any(self.uav_power_budgets_w <= 0.0):
                raise ValueError(
                    "per_uav constrained_delay requires positive UAV budgets."
                )
        self._sync_cpu_energy_prices()
        self.micro_state_mode = str(micro_state_mode).lower()
        if self.micro_state_mode not in ("pressure", "legacy"):
            raise ValueError(
                "micro_state_mode must be either 'pressure' or 'legacy'."
            )
        self.u2u_state_representation = str(
            u2u_state_representation
        ).lower()
        if self.u2u_state_representation not in (
            "nominal_rate",
            "spectral_efficiency",
        ):
            raise ValueError(
                "u2u_state_representation must be 'nominal_rate' or "
                "'spectral_efficiency'."
            )
        self.u2u_goal_representation = str(
            u2u_goal_representation
        ).strip().lower()
        if self.u2u_goal_representation not in (
            "match_observation",
            "nominal_rate",
        ):
            raise ValueError(
                "u2u_goal_representation must be 'match_observation' or "
                "'nominal_rate'."
            )
        if u2u_contention_aware_pressure is None:
            self.u2u_contention_aware_pressure = (
                self.u2u_state_representation == "spectral_efficiency"
            )
        else:
            self.u2u_contention_aware_pressure = bool(
                u2u_contention_aware_pressure
            )
        self.u2u_prev_active_state = bool(u2u_prev_active_state)
        self.u2u_total_directed_links = self.N * max(self.N - 1, 0)
        anchor_eta = self.env.compute_u2u_spectral_efficiency(
            self.env.anchor_points
        )
        off_diagonal = ~np.eye(self.N, dtype=bool)
        self.u2u_eta_reference = float(np.median(anchor_eta[off_diagonal]))
        if not np.isfinite(self.u2u_eta_reference) or self.u2u_eta_reference <= 0.0:
            raise ValueError("The anchor U2U spectral-efficiency reference is invalid.")
        self.prev_u2u_active_link_count = float(
            self.u2u_total_directed_links
        )

        # RL 惩罚系数体系（由包装器掌管，而非底层物理环境）
        self.w_D_high = 10.0
        self.w_D_low = 1.0
        self.w_E = 1.0

        # ============ [UNIFIED OBJECTIVE] 单一系统代价 J ============
        # J = J_w_delay * (加权时延代理) + J_w_energy * (能耗)
        # 三处信号(微观奖励 / 宏观奖励 / 评估)全部由同一个 J 导出,
        # 从根本上消除“训练信号与考核信号错位”。
        objective_weights = np.asarray(
            [delay_objective_weight, energy_objective_weight],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(objective_weights))
            or np.any(objective_weights < 0.0)
            or float(np.sum(objective_weights)) <= 0.0
        ):
            raise ValueError(
                "delay_objective_weight and energy_objective_weight must be "
                "finite, non-negative, and not both zero."
            )
        objective_weights /= float(np.sum(objective_weights))
        self.J_w_delay = float(objective_weights[0])
        self.J_w_energy = float(objective_weights[1])
        self.J_delay_norm = 10.0      # 全局加权 holding cost 归一 (smoke 校准)
        self.J_energy_norm = 14.0     # 每微观步总能耗(J)归一 (smoke 校准)
        self.J_pressure_norm = 8.0    # per-UAV 加权 backlog/容量 归一
        self.J_energy_pu_norm = 6.0   # per-UAV 能耗归一 (smoke 校准)
        self.J_cost_clip = 5.0        # 单步代价上界, 防梯度爆炸
        self.J_drift_weight = 0.0     # Lyapunov drift 系数 (先关, 调通后再开)
        self.J_drift_norm = 200.0
        self.overflow_penalty_weight = 2.0  # 取代 -50 尖峰的有界溢出惩罚
        self.prev_q_total = 0.0
        # 宏观奖励: feudal 主信号 + 衰减的规范势能 shaping
        self.macro_gamma = 0.9        # 与 PPO GAMMA 一致(规范势能 shaping)
        self.macro_feudal_weight = 1.0    # 实测微观产出为宏观主信号
        self.macro_shaping_lambda = 0.3   # 几何势能 shaping 初始权重(可退火至0)

        # [REWARD SHAPING] 系统最大排队容忍阈值与奖励缩放参数
        self.max_Q_threshold = 500.0
        self.max_macro_displacement = 20.0
        self.macro_action_deadzone = 0.05
        self.macro_queue_norm = 80.0
        self.macro_arrival_norm = 20.0
        self.max_u2g_distance = 200.0
        self.max_u2u_rate_mbps = 60.0
        self.target_u2g_distance = 60.0
        self.target_anchor_distance = 90.0
        self.target_u2u_rate_mbps = 18.0
        self.macro_zone_ingress_radius = 95.0

        self.macro_move_penalty_weight = 0.15
        self.macro_propulsion_tiebreak_weight = 0.0
        self.macro_propulsion_energy_norm = 100.0
        self.macro_upload_penalty_weight = 10

        # [REWARD SHAPING FIX] 削弱排队成本的支配地位，增强地面和队友的空间几何引导
        self.macro_queue_cost_weight = 1
        self.macro_u2g_cost_weight = float(macro_u2g_cost_weight)
        self.macro_u2u_cost_weight = 0
        self.macro_anchor_cost_weight = 0.0
        self.macro_zone_ingress_weight = float(macro_zone_ingress_weight)
        self.macro_action_mode = str(macro_action_mode).lower()
        if self.macro_action_mode not in (
            "polar",
            "delta",
            "waypoint",
            "centroid_waypoint",
        ):
            raise ValueError(
                "macro_action_mode must be 'polar', 'delta', 'waypoint', "
                "or 'centroid_waypoint'."
            )
        self.macro_waypoint_radius_m = float(macro_waypoint_radius_m)
        self.macro_waypoint_tolerance_m = float(macro_waypoint_tolerance_m)
        if self.macro_waypoint_radius_m <= 0.0:
            raise ValueError("macro_waypoint_radius_m must be positive.")
        if self.macro_waypoint_tolerance_m < 0.0:
            raise ValueError("macro_waypoint_tolerance_m must be non-negative.")
        self.macro_waypoint_origins = np.asarray(
            self.env.uav_coords,
            dtype=np.float32,
        ).copy()
        self.macro_comm_weight = float(macro_comm_weight)
        self.macro_comm_flow_ema = float(
            np.clip(macro_comm_flow_ema, 0.0, 0.999)
        )
        self.macro_comm_cost_norm_s = max(float(self.env.macro_dt), 1e-8)
        self.macro_flow_state = bool(macro_flow_state)
        self.macro_load_state = bool(macro_load_state)
        self.macro_capacity_state = bool(macro_capacity_state)
        self.macro_cpu_norm_ghz = 16.0
        self.macro_flow_norm_mbits = 100.0
        self.macro_reward_mode = str(macro_reward_mode).lower()
        if self.macro_reward_mode not in ("strict_global", "legacy_feudal"):
            raise ValueError(
                "macro_reward_mode must be 'strict_global' or 'legacy_feudal'."
            )
        self.strict_hierarchy_validation = bool(strict_hierarchy_validation)
        self._macro_slot_active = False
        self._macro_slot_index = 0

        # 行动空间定义（统一归一化到 [-1, 1]）
        # Macro Action: normalized [rho, theta] for N UAVs
        self.macro_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.N, 2),
            dtype=np.float32,
        )

        micro_action_dim = (
            2 * self.N * (self.N - 1)
            if self.micro_action_encoding == "stick_breaking"
            else 2 * self.N * self.N
        )
        self.micro_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(micro_action_dim,),
            dtype=np.float32,
        )

        # [STATE FIX] 宏观层局部/全局状态维度
        # 让宏观观测能看到“它要优化的量”和“队友是否过载”:
        #   centroid offset (2)
        #   own summary (5): [Q_high, Q_low, arrival, outside_zone, mean_u2u_rate]
        #   teammate feature (4 each): [rel_x, rel_y, Q_high, Q_low]
        self.macro_base_pos_dim = 2
        self.macro_own_summary_dim = 5
        self.macro_teammate_feature_dim = 4
        if self.macro_flow_state:
            self.macro_own_summary_dim += 2
            self.macro_teammate_feature_dim += 2
        if self.macro_load_state:
            self.macro_own_summary_dim += 1
            self.macro_teammate_feature_dim += 1
        if self.macro_capacity_state:
            self.macro_own_summary_dim += 1
            self.macro_teammate_feature_dim += 1
        if self.power_constraint_scope == "per_uav":
            self.macro_own_summary_dim += 1
            self.macro_teammate_feature_dim += 1
        elif self.system_dual_state:
            self.macro_own_summary_dim += 1
        self.macro_local_obs_dim = (
            self.macro_base_pos_dim
            + self.macro_own_summary_dim
            + self.macro_teammate_feature_dim * (self.N - 1)
        )
        self.macro_global_obs_dim = self.macro_local_obs_dim * self.N

        # SAC 微观层状态维度:
        #   4N  : Q_loc(高/低) + A_t(高/低)
        #   3N^2: Q_tx(高/低) + norm_u2u_rate
        #   N^2 : [HIERARCHY] 宏观派生的建议卸载分布 goal G
        self.sac_obs_dim = (4 * self.N) + (4 * self.N * self.N)
        if self.u2u_prev_active_state:
            self.sac_obs_dim += 1
        if self.power_constraint_scope == "per_uav":
            self.sac_obs_dim += self.N
        elif self.system_dual_state:
            self.sac_obs_dim += 1

        # 奖励缓存
        self.accumulated_micro_reward = 0.0
        self.accumulated_micro_reward_per_uav = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.accumulated_primal_cost = 0.0
        self.accumulated_energy_constraint_ratio = 0.0
        self.accumulated_energy_constraint_ratio_per_uav = np.zeros(
            self.N,
            dtype=np.float64,
        )
        self.current_flying_penalty = 0.0
        self.accumulated_upload_delay_penalty = 0.0
        self.accumulated_upload_delay_penalty_per_uav = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.current_macro_micro_steps = 0
        self.prev_macro_positioning_cost = 0.0
        self.prev_macro_positioning_cost_per_uav = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.current_macro_positioning_cost = 0.0
        self.current_macro_positioning_cost_per_uav = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.current_macro_positioning_improvement_per_uav = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.current_macro_movement_penalty_per_uav = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.current_macro_propulsion_delta_per_uav = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.current_macro_propulsion_energy_share = 0.0
        self.current_macro_propulsion_energy_per_uav_share = np.zeros(
            self.N,
            dtype=np.float64,
        )
        self.current_macro_propulsion_delta_share = 0.0
        self.current_macro_action_norm_mean = 0.0
        self.current_macro_hover_ratio = 1.0
        # SCA warm starts are episode-local implementation details, not part
        # of the controller state.  Reset them so fixed-seed evaluations do
        # not depend on the order in which episodes are evaluated.
        for attr_name in (
            "_sca_cached_key",
            "_sca_cached_action",
            "_sca_action_warm",
            "_sca_rho_warm",
            "_sca_f_warm",
        ):
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.last_macro_reward_info = {}

        # 宏观统计特征缓存
        self.prev_macro_q_up_mean = np.zeros(self.N, dtype=np.float32)
        self.prev_macro_q_loc_mean = np.zeros(self.N, dtype=np.float32)
        self.prev_macro_q_tx_out_mean = np.zeros(self.N, dtype=np.float32)
        self.prev_macro_cpu_util_mean = np.zeros(self.N, dtype=np.float32)
        self.prev_macro_u2g_mean = np.zeros(self.N, dtype=np.float32)

        self.curr_macro_q_up_sum = np.zeros(self.N, dtype=np.float32)
        self.curr_macro_q_loc_sum = np.zeros(self.N, dtype=np.float32)
        self.curr_macro_q_tx_out_sum = np.zeros(self.N, dtype=np.float32)
        self.curr_macro_cpu_util_sum = np.zeros(self.N, dtype=np.float32)
        self.curr_macro_u2g_sum = np.zeros(self.N, dtype=np.float32)
        self.curr_macro_stat_count = 0
        self.curr_macro_stats_dirty = False

        self.prev_macro_flow_ema_mbits = np.zeros(
            (self.N, self.N),
            dtype=np.float32,
        )
        self.curr_macro_flow_mbits = np.zeros(
            (self.N, self.N),
            dtype=np.float32,
        )
        self.current_macro_flow_reference_mbits = np.zeros(
            (self.N, self.N),
            dtype=np.float32,
        )
        self._last_coop_flow_mbits = np.zeros(
            (self.N, self.N),
            dtype=np.float32,
        )
        self.curr_macro_flow_dirty = False

        # U2U 拓扑缓存：UAV 坐标不变时复用上次结果
        self._u2u_cache_coords = None
        self._u2u_cache_result = None

        # [PERF] 最近一次 micro_step 的 raw_state, 供外部复用避免重复拷贝
        self.last_raw_state = None

    def _sync_cpu_energy_prices(self):
        if self.env.cpu_alloc_mode != "priced_maxweight":
            return
        if self.power_constraint_scope == "per_uav":
            prices = np.divide(
                self.energy_lagrange_multipliers,
                np.maximum(self.uav_power_budgets_w, 1e-8),
            )
        else:
            prices = np.full(
                self.N,
                self.energy_lagrange_multiplier
                / max(self.power_budget_w, 1e-8),
            )
        self.env.set_cpu_energy_prices(prices)

    def set_energy_lagrange_multipliers(self, multipliers):
        values = np.asarray(multipliers, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(self.N, float(values))
        elif values.shape == (1,):
            values = np.full(self.N, float(values[0]))
        if values.shape != (self.N,):
            raise ValueError(
                "Energy multipliers must be scalar or have shape "
                f"({self.N},), got {values.shape}."
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "Energy multipliers must be finite and non-negative."
            )
        self.energy_lagrange_multipliers = values.copy()
        self.energy_lagrange_multiplier = float(np.mean(values))
        self._sync_cpu_energy_prices()

    def _reset_macro_summary_accumulator(self):
        self.curr_macro_q_up_sum.fill(0.0)
        self.curr_macro_q_loc_sum.fill(0.0)
        self.curr_macro_q_tx_out_sum.fill(0.0)
        self.curr_macro_cpu_util_sum.fill(0.0)
        self.curr_macro_u2g_sum.fill(0.0)
        self.curr_macro_stat_count = 0
        self.curr_macro_stats_dirty = False

    def _finalize_macro_summary_if_needed(self):
        if not self.curr_macro_stats_dirty or self.curr_macro_stat_count <= 0:
            return

        denom = float(self.curr_macro_stat_count)

        self.prev_macro_q_up_mean = (
            self.curr_macro_q_up_sum / denom
        ).astype(np.float32)
        self.prev_macro_q_loc_mean = (
            self.curr_macro_q_loc_sum / denom
        ).astype(np.float32)
        self.prev_macro_q_tx_out_mean = (
            self.curr_macro_q_tx_out_sum / denom
        ).astype(np.float32)
        self.prev_macro_cpu_util_mean = (
            self.curr_macro_cpu_util_sum / denom
        ).astype(np.float32)
        self.prev_macro_u2g_mean = (
            self.curr_macro_u2g_sum / denom
        ).astype(np.float32)

        self._reset_macro_summary_accumulator()

    def _expected_macro_upload_mbits(self):
        user_expected_mbps = (
            np.asarray(self.env.user_lambda_high, dtype=np.float64)
            * np.asarray(
                self.env.user_task_size_high_mbits,
                dtype=np.float64,
            )
            + np.asarray(self.env.user_lambda_low, dtype=np.float64)
            * np.asarray(
                self.env.user_task_size_low_mbits,
                dtype=np.float64,
            )
        )
        expected_mbps = np.zeros(self.N, dtype=np.float64)
        if (
            user_expected_mbps.shape == (self.env.M,)
            and float(np.sum(user_expected_mbps)) > 1e-8
        ):
            associations = np.asarray(
                self.env.user_allocations,
                dtype=np.int64,
            )
            np.add.at(expected_mbps, associations, user_expected_mbps)

        if expected_mbps.shape[0] != self.N or np.sum(expected_mbps) <= 1e-8:
            expected_mbps = np.zeros(self.N, dtype=np.float64)
            for i in range(self.N):
                if hasattr(self.env, "_get_zone_target_load_mbps"):
                    expected_mbps[i] = float(self.env._get_zone_target_load_mbps(i))
                else:
                    expected_mbps[i] = 1.0

        expected = expected_mbps * float(self.env.macro_dt)
        expected = np.maximum(expected, 0.0).astype(np.float32)
        self.macro_flow_norm_mbits = float(max(np.max(expected), 1.0))
        return expected

    def _reset_macro_flow_accumulator(self):
        self.curr_macro_flow_mbits.fill(0.0)
        self.curr_macro_flow_dirty = False
        current_flow = np.asarray(
            getattr(self.env, "coop_flow_mbits", np.zeros((self.N, self.N))),
            dtype=np.float32,
        )
        if current_flow.shape == (self.N, self.N):
            self._last_coop_flow_mbits = current_flow.copy()
        else:
            self._last_coop_flow_mbits.fill(0.0)

    def _estimate_macro_flow_reference(self, raw_state):
        q_tx = (
            np.asarray(raw_state["Q_tx_high"], dtype=np.float32)
            + np.asarray(raw_state["Q_tx_low"], dtype=np.float32)
        )
        recent_flow = np.asarray(
            self.prev_macro_flow_ema_mbits,
            dtype=np.float32,
        )
        observed = 0.7 * recent_flow + 0.3 * np.maximum(q_tx, 0.0)
        expected_upload = self._expected_macro_upload_mbits()

        flow = np.zeros((self.N, self.N), dtype=np.float32)
        for i in range(self.N):
            row = np.maximum(observed[i], 0.0).astype(np.float32)
            row_sum = float(np.sum(row))
            if row_sum <= 1e-8:
                flow[i, i] = expected_upload[i]
            else:
                flow[i] = row * (expected_upload[i] / row_sum)
        return flow

    def _compute_uplink_rates_mbps(self, raw_state, positions):
        positions = np.asarray(positions, dtype=np.float64)
        user_coords = np.asarray(
            raw_state.get("user_coords", self.env.user_coords),
            dtype=np.float64,
        )
        assigned = np.asarray(
            self.env.user_allocations,
            dtype=np.int64,
        )
        user_rate_mbps = (
            self.env.compute_u2g_rate_bps(
                user_coords=user_coords,
                uav_coords=positions,
                associations=assigned,
            )
            / 1e6
        )

        zone_rates = np.zeros(self.N, dtype=np.float64)
        np.add.at(zone_rates, assigned, user_rate_mbps)
        return np.maximum(zone_rates, 1e-9).astype(np.float32)

    def _compute_flow_u2u_rates_mbps(self, positions, flow_mbits):
        positions = np.asarray(positions, dtype=np.float64)
        flow = np.asarray(flow_mbits, dtype=np.float64)
        active = flow > 1e-9
        np.fill_diagonal(active, False)

        rates = np.zeros((self.N, self.N), dtype=np.float64)
        rates_bps = self.env.compute_u2u_rate_bps(
            uav_coords=positions,
        )
        rates[active] = rates_bps[active] / 1e6
        return rates.astype(np.float32)

    def _compute_flow_aware_comm_cost(self, raw_state, flow_mbits):
        positions = np.asarray(raw_state["uav_coords"], dtype=np.float32)
        flow = np.asarray(flow_mbits, dtype=np.float32).copy()
        if flow.shape != (self.N, self.N):
            flow = self._estimate_macro_flow_reference(raw_state)
        flow = np.maximum(flow, 0.0)

        uplink_rates = self._compute_uplink_rates_mbps(raw_state, positions)
        upload_volume = np.sum(flow, axis=1).astype(np.float32)
        uplink_time = upload_volume / np.maximum(uplink_rates, 1e-9)

        u2u_rates = self._compute_flow_u2u_rates_mbps(positions, flow)
        offdiag = ~np.eye(self.N, dtype=bool)
        active = offdiag & (flow > 1e-9)
        u2u_time_matrix = np.zeros((self.N, self.N), dtype=np.float32)
        u2u_time_matrix[active] = flow[active] / np.maximum(
            u2u_rates[active],
            1e-9,
        )

        u2u_time_out = np.sum(u2u_time_matrix, axis=1)
        u2u_time_in = np.sum(u2u_time_matrix, axis=0)
        comm_time_per_uav = (
            uplink_time + 0.5 * (u2u_time_out + u2u_time_in)
        ).astype(np.float32)
        comm_time = float(np.sum(uplink_time) + np.sum(u2u_time_matrix))

        norm = max(float(self.macro_comm_cost_norm_s), 1e-8)
        return {
            "comm_cost": float(comm_time / norm),
            "comm_cost_per_uav": (comm_time_per_uav / norm).astype(np.float32),
            "comm_time_s": float(comm_time),
            "comm_uplink_time_s": float(np.sum(uplink_time)),
            "comm_u2u_time_s": float(np.sum(u2u_time_matrix)),
            "comm_uplink_time_per_uav_s": uplink_time.astype(np.float32),
            "comm_u2u_time_matrix_s": u2u_time_matrix.astype(np.float32),
            "comm_uplink_rates_mbps": uplink_rates.astype(np.float32),
            "comm_u2u_rates_mbps": u2u_rates.astype(np.float32),
            "flow_reference_mbits": flow.astype(np.float32),
        }

    def _update_macro_flow_ema(self):
        if not self.curr_macro_flow_dirty:
            return
        realized = np.maximum(self.curr_macro_flow_mbits, 0.0).astype(np.float32)
        if float(np.sum(realized)) <= 1e-8:
            return
        beta = self.macro_comm_flow_ema
        self.prev_macro_flow_ema_mbits = (
            beta * self.prev_macro_flow_ema_mbits
            + (1.0 - beta) * realized
        ).astype(np.float32)

    def _compute_u2u_topology_metrics(self, uav_coords, q_tx_matrix):
        """
        [MAPPO UPGRADE]
        计算当前 UAV-UAV 拓扑的距离、名义链路速率与双向链路需求。
        distance/rate 仅依赖 uav_coords，对坐标未变的步骤使用缓存。
        """
        # 只有坐标改变时才重新计算 distance 和 rate
        if (
            self._u2u_cache_coords is None
            or not np.array_equal(uav_coords, self._u2u_cache_coords)
        ):
            diff = uav_coords[:, None, :] - uav_coords[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)
            nominal_rate_bps = self.env.compute_u2u_rate_bps(
                uav_coords=uav_coords,
            )
            spectral_efficiency = self.env.compute_u2u_spectral_efficiency(
                uav_coords=uav_coords,
            )

            self._u2u_cache_coords = uav_coords.copy()
            self._u2u_cache_result = (
                dist_matrix.astype(np.float32),
                (nominal_rate_bps / 1e6).astype(np.float32),
                spectral_efficiency.astype(np.float32),
            )

        dist_matrix, rate_mbps, _ = self._u2u_cache_result

        link_demand_matrix = q_tx_matrix + q_tx_matrix.T
        np.fill_diagonal(link_demand_matrix, 0.0)

        return dist_matrix, rate_mbps, link_demand_matrix.astype(np.float32)

    def _normalized_u2u_observation(self, uav_coords, nominal_rate_mbps=None):
        if self.u2u_state_representation == "nominal_rate":
            if nominal_rate_mbps is None:
                _, nominal_rate_mbps, _ = self._compute_u2u_topology_metrics(
                    uav_coords,
                    np.zeros((self.N, self.N), dtype=np.float32),
                )
            return np.clip(
                np.asarray(nominal_rate_mbps, dtype=np.float32)
                / max(self.max_u2u_rate_mbps, 1e-8),
                0.0,
                1.0,
            ).astype(np.float32)

        self._compute_u2u_topology_metrics(
            uav_coords,
            np.zeros((self.N, self.N), dtype=np.float32),
        )
        spectral_efficiency = self._u2u_cache_result[2]
        return (
            spectral_efficiency / max(self.u2u_eta_reference, 1e-8)
        ).astype(np.float32)

    def _previous_u2u_active_fraction(self):
        return float(
            self.prev_u2u_active_link_count
            / max(self.u2u_total_directed_links, 1)
        )

    def _goal_u2u_observation(self, uav_coords, nominal_rate_mbps):
        if self.u2u_goal_representation == "match_observation":
            return self._normalized_u2u_observation(
                uav_coords,
                nominal_rate_mbps=nominal_rate_mbps,
            )
        return np.clip(
            np.asarray(nominal_rate_mbps, dtype=np.float32)
            / max(self.max_u2u_rate_mbps, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)

    def _estimated_u2u_rate_mbps(self, uav_coords):
        spectral_efficiency = self.env.compute_u2u_spectral_efficiency(
            uav_coords=uav_coords
        )
        active_count = float(self.prev_u2u_active_link_count)
        if active_count <= 0.0:
            active_count = float(self.u2u_total_directed_links)
        bandwidth_hz = self.env.B_u2u_total / max(active_count, 1.0)
        return (bandwidth_hz * spectral_efficiency / 1e6).astype(np.float32)

    def _update_previous_u2u_contention(self, infos):
        counts = []
        off_diagonal = ~np.eye(self.N, dtype=bool)
        for info in infos:
            bandwidth = np.asarray(
                info.get(
                    "u2u_link_bandwidth_hz",
                    np.zeros((self.N, self.N)),
                ),
                dtype=np.float64,
            )
            if bandwidth.shape != (self.N, self.N):
                continue
            counts.append(float(np.count_nonzero(bandwidth[off_diagonal] > 0.0)))
        mean_count = float(np.mean(counts)) if counts else 0.0
        if mean_count <= 0.0:
            mean_count = float(self.u2u_total_directed_links)
        self.prev_u2u_active_link_count = mean_count

    def _build_macro_snapshot(self, raw_state):
        """
        [MAPPO UPGRADE]
        汇总宏观层做站位决策所需的当前拓扑、负载与链路信息。
        """
        q_loc_total = (
            raw_state["Q_loc_high"] + raw_state["Q_loc_low"]
        ).astype(np.float32)
        q_tx_matrix = (
            raw_state["Q_tx_high"] + raw_state["Q_tx_low"]
        ).astype(np.float32)
        q_tx_out_total = np.sum(q_tx_matrix, axis=1).astype(np.float32)
        q_tx_in_total = np.sum(q_tx_matrix, axis=0).astype(np.float32)

        current_arrival = (
            raw_state["A_t_high"] + raw_state["A_t_low"]
        ).astype(np.float32)
        arrival_signal = (
            0.5 * self.prev_macro_q_up_mean + 0.5 * current_arrival
        ).astype(np.float32)

        cpu_util = np.clip(
            self.prev_macro_cpu_util_mean.copy(),
            0.0,
            1.0,
        ).astype(np.float32)

        u2g_distance = np.linalg.norm(
            raw_state["uav_coords"] - raw_state["weighted_centroids"],
            axis=1,
        ).astype(np.float32)
        anchor_distance = np.linalg.norm(
            raw_state["uav_coords"] - self.env.anchor_points,
            axis=1,
        ).astype(np.float32)

        u2u_distance, u2u_rate_mbps, link_demand_matrix = (
            self._compute_u2u_topology_metrics(
                raw_state["uav_coords"],
                q_tx_matrix,
            )
        )

        macro_load = (
            arrival_signal
            + q_loc_total
            + q_tx_out_total
            + q_tx_in_total
        ).astype(np.float32)
        return {
            "q_loc_total": q_loc_total,
            "q_tx_matrix": q_tx_matrix,
            "q_tx_out_total": q_tx_out_total,
            "q_tx_in_total": q_tx_in_total,
            "arrival_signal": arrival_signal,
            "cpu_util": cpu_util,
            "u2g_distance": u2g_distance,
            "anchor_distance": anchor_distance,
            "u2u_distance": u2u_distance,
            "u2u_rate_mbps": u2u_rate_mbps,
            "link_demand_matrix": link_demand_matrix,
            "macro_load": macro_load,
        }

    def _compute_macro_positioning_cost(self, raw_state):
        global_info, _ = self._compute_macro_costs(raw_state)
        return global_info

    def _compute_macro_costs(self, raw_state, flow_mbits=None):
        # 合并版本：一次 snapshot，同时返回全局代价和 per-uav 代价。
        snapshot = self._build_macro_snapshot(raw_state)
        if flow_mbits is None:
            flow_mbits = self._estimate_macro_flow_reference(raw_state)
        comm_info = self._compute_flow_aware_comm_cost(raw_state, flow_mbits)

        # --- 全局量 ---
        q_loc_h = float(np.sum(raw_state["Q_loc_high"]))
        q_tx_h = float(np.sum(raw_state["Q_tx_high"]))
        q_loc_l = float(np.sum(raw_state["Q_loc_low"]))
        q_tx_l = float(np.sum(raw_state["Q_tx_low"]))

        queue_cost = (
            10.0 * q_loc_h + 10.0 * q_tx_h + 1.0 * q_loc_l + 1.0 * q_tx_l
        ) / self.macro_queue_norm

        base_u2g_cost = (snapshot["u2g_distance"] / max(self.target_u2g_distance, 1e-8)) ** 2
        load_signal = np.maximum(snapshot["macro_load"], 1.0)
        load_weights = load_signal / np.sum(load_signal)
        u2g_cost = float(np.sum(load_weights * base_u2g_cost))

        zone_ingress_violation = np.maximum(
            snapshot["u2g_distance"] - self.macro_zone_ingress_radius, 0.0
        )
        zone_ingress_cost = float(
            np.mean((zone_ingress_violation / max(self.macro_zone_ingress_radius, 1e-8)) ** 2)
        )
        anchor_cost = float(
            np.mean((snapshot["anchor_distance"] / max(self.target_anchor_distance, 1e-8)) ** 2)
        )

        # 提前初始化，避免静态分析器报 possibly-undefined 警告
        baseline_u2u_cost = 0.0
        mean_nominal_u2u_rate_mbps = 0.0
        mean_u2u_distance_m = 0.0
        u2u_cost_per_uav = np.zeros(self.N, dtype=np.float32)

        pair_mask = ~np.eye(self.N, dtype=bool)
        if np.any(pair_mask):
            pair_rate_shortfall = (
                self.max_u2u_rate_mbps - snapshot["u2u_rate_mbps"][pair_mask]
            ) / max(self.target_u2u_rate_mbps, 1e-8)
            baseline_u2u_cost = float(np.mean(np.maximum(pair_rate_shortfall, 0.0)))
            mean_nominal_u2u_rate_mbps = float(np.mean(snapshot["u2u_rate_mbps"][pair_mask]))
            mean_u2u_distance_m = float(np.mean(snapshot["u2u_distance"][pair_mask]))

            rate_shortfall_matrix = np.maximum(
                (self.max_u2u_rate_mbps - snapshot["u2u_rate_mbps"])
                / max(self.target_u2u_rate_mbps, 1e-8),
                0.0,
            ).astype(np.float32)
            np.fill_diagonal(rate_shortfall_matrix, 0.0)

            for i in range(self.N):
                neighbor_mask = np.ones(self.N, dtype=bool)
                neighbor_mask[i] = False
                pair_terms = rate_shortfall_matrix[i, neighbor_mask]
                baseline_local = float(np.mean(pair_terms)) if pair_terms.size > 0 else 0.0
                local_demand = (
                    snapshot["link_demand_matrix"][i] + snapshot["link_demand_matrix"][:, i]
                )[neighbor_mask].astype(np.float32)
                local_demand_sum = float(np.sum(local_demand))
                if local_demand_sum > 1e-8 and pair_terms.size > 0:
                    demand_local = float(np.sum(local_demand * pair_terms) / local_demand_sum)
                else:
                    demand_local = baseline_local
                u2u_cost_per_uav[i] = 0.2 * baseline_local + 0.8 * demand_local

        link_demand_sum = float(np.sum(snapshot["link_demand_matrix"]))
        if link_demand_sum > 1e-8:
            demand_weights = snapshot["link_demand_matrix"] / link_demand_sum
            demand_rate_shortfall = (
                self.max_u2u_rate_mbps - snapshot["u2u_rate_mbps"]
            ) / max(self.target_u2u_rate_mbps, 1e-8)
            demand_u2u_cost = float(
                np.sum(demand_weights * np.maximum(demand_rate_shortfall, 0.0))
            )
            u2u_cost = 0.2 * baseline_u2u_cost + 0.8 * demand_u2u_cost
        else:
            u2u_cost = baseline_u2u_cost

        total_cost = (
            self.macro_queue_cost_weight * queue_cost
            + self.macro_u2g_cost_weight * u2g_cost
            + self.macro_u2u_cost_weight * u2u_cost
            + self.macro_anchor_cost_weight * anchor_cost
            + self.macro_zone_ingress_weight * zone_ingress_cost
            + self.macro_comm_weight * comm_info["comm_cost"]
        )

        global_info = {
            "total_cost": float(total_cost),
            "queue_cost": float(queue_cost),
            "u2g_cost": float(u2g_cost),
            "u2u_cost": float(u2u_cost),
            "comm_cost": float(comm_info["comm_cost"]),
            "comm_time_s": float(comm_info["comm_time_s"]),
            "comm_uplink_time_s": float(comm_info["comm_uplink_time_s"]),
            "comm_u2u_time_s": float(comm_info["comm_u2u_time_s"]),
            "macro_comm_weight": float(self.macro_comm_weight),
            "flow_reference_mbits": comm_info["flow_reference_mbits"].copy(),
            "comm_uplink_rates_mbps": comm_info[
                "comm_uplink_rates_mbps"
            ].copy(),
            "comm_u2u_rates_mbps": comm_info["comm_u2u_rates_mbps"].copy(),
            "comm_u2u_time_matrix_s": comm_info[
                "comm_u2u_time_matrix_s"
            ].copy(),
            "anchor_cost": float(anchor_cost),
            "zone_ingress_cost": float(zone_ingress_cost),
            "mean_u2g_distance_m": float(np.mean(snapshot["u2g_distance"])),
            "mean_u2u_distance_m": mean_u2u_distance_m,
            "mean_nominal_u2u_rate_mbps": mean_nominal_u2u_rate_mbps,
            "snapshot": snapshot,
        }

        # --- Per-UAV 量（复用同一 snapshot 和已算好的 u2u_cost_per_uav）---
        q_loc_high = raw_state["Q_loc_high"].astype(np.float32)
        q_loc_low = raw_state["Q_loc_low"].astype(np.float32)
        q_tx_high = raw_state["Q_tx_high"].astype(np.float32)
        q_tx_low = raw_state["Q_tx_low"].astype(np.float32)

        q_tx_out_high = np.sum(q_tx_high, axis=1)
        q_tx_out_low = np.sum(q_tx_low, axis=1)
        q_tx_in_high = np.sum(q_tx_high, axis=0)
        q_tx_in_low = np.sum(q_tx_low, axis=0)

        queue_cost_per_uav = (
            10.0 * (q_loc_high + q_tx_out_high + q_tx_in_high)
            + 1.0 * (q_loc_low + q_tx_out_low + q_tx_in_low)
        ) / self.macro_queue_norm

        u2g_cost_per_uav = (
            (snapshot["u2g_distance"] / max(self.target_u2g_distance, 1e-8)) ** 2
        ).astype(np.float32)

        zone_ingress_cost_per_uav = (
            (zone_ingress_violation / max(self.macro_zone_ingress_radius, 1e-8)) ** 2
        ).astype(np.float32)

        anchor_cost_per_uav = (
            (snapshot["anchor_distance"] / max(self.target_anchor_distance, 1e-8)) ** 2
        ).astype(np.float32)

        total_cost_per_uav = (
            self.macro_queue_cost_weight * queue_cost_per_uav
            + self.macro_u2g_cost_weight * u2g_cost_per_uav
            + self.macro_u2u_cost_weight * u2u_cost_per_uav
            + self.macro_anchor_cost_weight * anchor_cost_per_uav
            + self.macro_zone_ingress_weight * zone_ingress_cost_per_uav
            + self.macro_comm_weight * comm_info["comm_cost_per_uav"]
        ).astype(np.float32)

        per_uav_info = {
            "total_cost_per_uav": total_cost_per_uav.copy(),
            "queue_cost_per_uav": queue_cost_per_uav.copy(),
            "u2g_cost_per_uav": u2g_cost_per_uav.copy(),
            "u2u_cost_per_uav": u2u_cost_per_uav.copy(),
            "comm_cost_per_uav": comm_info["comm_cost_per_uav"].copy(),
            "comm_uplink_time_per_uav_s": comm_info[
                "comm_uplink_time_per_uav_s"
            ].copy(),
            "anchor_cost_per_uav": anchor_cost_per_uav.copy(),
            "zone_ingress_cost_per_uav": zone_ingress_cost_per_uav.copy(),
            "mean_u2g_distance_m": float(np.mean(snapshot["u2g_distance"])),
            "mean_u2u_distance_m": mean_u2u_distance_m,
            "mean_nominal_u2u_rate_mbps": mean_nominal_u2u_rate_mbps,
            "snapshot": snapshot,
        }

        return global_info, per_uav_info

    def reset(self, seed=None):
        _, info = self.env.reset(seed=seed)
        self.macro_waypoint_origins = np.asarray(
            self.env.uav_coords,
            dtype=np.float32,
        ).copy()

        self.accumulated_micro_reward = 0.0
        self.accumulated_micro_reward_per_uav.fill(0.0)
        self.accumulated_primal_cost = 0.0
        self.accumulated_energy_constraint_ratio = 0.0
        self.accumulated_energy_constraint_ratio_per_uav.fill(0.0)
        self.current_flying_penalty = 0.0
        self.accumulated_upload_delay_penalty = 0.0
        self.accumulated_upload_delay_penalty_per_uav.fill(0.0)
        self.current_macro_micro_steps = 0
        self._macro_slot_active = False
        self._macro_slot_index = 0
        self.prev_q_total = 0.0
        # 清除 U2U 拓扑缓存
        self._u2u_cache_coords = None
        self._u2u_cache_result = None
        self.prev_u2u_active_link_count = float(
            self.u2u_total_directed_links
        )
        self.current_macro_movement_penalty_per_uav.fill(0.0)
        self.current_macro_propulsion_delta_per_uav.fill(0.0)
        self.current_macro_propulsion_energy_share = 0.0
        self.current_macro_propulsion_energy_per_uav_share.fill(0.0)
        self.current_macro_propulsion_delta_share = 0.0
        self.current_macro_action_norm_mean = 0.0
        self.current_macro_hover_ratio = 1.0
        self._sync_cpu_energy_prices()

        self.prev_macro_q_up_mean.fill(0.0)
        self.prev_macro_q_loc_mean.fill(0.0)
        self.prev_macro_q_tx_out_mean.fill(0.0)
        self.prev_macro_cpu_util_mean.fill(0.0)
        self.prev_macro_u2g_mean.fill(0.0)

        self._reset_macro_summary_accumulator()
        expected_flow = np.diag(self._expected_macro_upload_mbits()).astype(
            np.float32,
        )
        self.prev_macro_flow_ema_mbits = expected_flow.copy()
        self.current_macro_flow_reference_mbits = expected_flow.copy()
        self._reset_macro_flow_accumulator()

        initial_cost_info, initial_cost_info_per_uav = self._compute_macro_costs(
            self.env.get_raw_state()
        )
        self.prev_macro_positioning_cost = initial_cost_info["total_cost"]
        self.current_macro_positioning_cost = initial_cost_info["total_cost"]
        self.prev_macro_positioning_cost_per_uav = initial_cost_info_per_uav[
            "total_cost_per_uav"
        ].copy()
        self.current_macro_positioning_cost_per_uav = (
            self.prev_macro_positioning_cost_per_uav.copy()
        )
        self.current_macro_positioning_improvement_per_uav.fill(0.0)
        self.last_macro_reward_info = {
            "macro_reward": 0.0,
            "macro_rewards_per_uav": np.zeros(self.N, dtype=np.float32),
            "positioning_improvement": 0.0,
            "positioning_improvement_per_uav": np.zeros(
                self.N,
                dtype=np.float32,
            ),
            "movement_penalty": 0.0,
            "movement_penalty_per_uav": np.zeros(
                self.N,
                dtype=np.float32,
            ),
            **initial_cost_info,
            **initial_cost_info_per_uav,
        }

        return self.get_macro_state(), info

    def get_macro_state(self):
        """
        [STATE FIX]
        获取宏观层状态。每架 UAV 的局部观测现在能看到:
        - 到加权质心的偏移 (2)
        - 自身分优先级队列足迹 + 到达 + 出区距离 + 平均 u2u 速率 (5)
        - 每个队友的相对坐标 + 分优先级队列 (4 * (N-1))

        关键修复:
        1. 宏观终于能观测到它要优化的量(分优先级队列)。
        2. 队友特征包含队友负载,使“飞近过载队友去中继”可学习。
        3. 去掉对滞后统计(prev_macro_*)的依赖,保持观测的马尔可夫性。

        - local_obs_list: 长度为 N 的列表，每个元素是单架 UAV 的局部观测
        - global_state: 所有局部观测拼接而成的全局状态，供 Critic 使用
        """
        self._finalize_macro_summary_if_needed()

        raw_state = self.env.get_raw_state()
        uav_coords_m = raw_state["uav_coords"]
        uav_coords = uav_coords_m / 1000.0
        centroids = raw_state["weighted_centroids"] / 1000.0

        # --- 分优先级的每节点队列足迹 (本地 + 转出 + 转入) ---
        q_tx_high = raw_state["Q_tx_high"]
        q_tx_low = raw_state["Q_tx_low"]
        q_high_node = (
            raw_state["Q_loc_high"]
            + np.sum(q_tx_high, axis=1)
            + np.sum(q_tx_high, axis=0)
        ).astype(np.float32)
        q_low_node = (
            raw_state["Q_loc_low"]
            + np.sum(q_tx_low, axis=1)
            + np.sum(q_tx_low, axis=0)
        ).astype(np.float32)
        arrival_node = (
            raw_state["A_t_high"] + raw_state["A_t_low"]
        ).astype(np.float32)

        norm_q_high = np.clip(q_high_node / self.macro_queue_norm, 0.0, 1.0)
        norm_q_low = np.clip(q_low_node / self.macro_queue_norm, 0.0, 1.0)
        norm_arrival = np.clip(
            arrival_node / max(self.macro_arrival_norm, 1e-8), 0.0, 1.0
        )

        # --- u2g 距离与出区违规 ---
        u2g_distance = np.linalg.norm(
            uav_coords_m - raw_state["weighted_centroids"], axis=1
        ).astype(np.float32)
        outside_zone = np.maximum(
            u2g_distance - self.macro_zone_ingress_radius, 0.0
        )
        norm_outside = np.clip(
            outside_zone
            / max(self.max_u2g_distance - self.macro_zone_ingress_radius, 1e-8),
            0.0,
            1.0,
        )

        # --- 自身到队友的平均 u2u 名义速率 ---
        _, u2u_rate_mbps, _ = self._compute_u2u_topology_metrics(
            uav_coords_m, (q_tx_high + q_tx_low).astype(np.float32)
        )
        u2u_observation = self._normalized_u2u_observation(
            uav_coords_m,
            nominal_rate_mbps=u2u_rate_mbps,
        )
        if self.N > 1:
            off_diag = ~np.eye(self.N, dtype=bool)
            own_u2u_mean = np.array(
                [np.mean(u2u_observation[i][off_diag[i]]) for i in range(self.N)],
                dtype=np.float32,
            )
        else:
            own_u2u_mean = np.zeros(self.N, dtype=np.float32)
        norm_u2u_mean = own_u2u_mean.astype(np.float32)
        flow_ref = self._estimate_macro_flow_reference(raw_state)
        flow_norm = max(float(self.macro_flow_norm_mbits), 1e-8)
        norm_flow = np.clip(flow_ref / flow_norm, 0.0, 1.0)
        norm_flow_out = np.clip(
            np.sum(flow_ref, axis=1) / flow_norm,
            0.0,
            1.0,
        ).astype(np.float32)
        norm_flow_in = np.clip(
            np.sum(flow_ref, axis=0) / flow_norm,
            0.0,
            1.0,
        ).astype(np.float32)
        expected_upload = self._expected_macro_upload_mbits()
        current_load_scale = max(
            float(getattr(self.env, "load_scale", 1.0)),
            1e-6,
        )
        base_expected = expected_upload / current_load_scale
        base_expected_norm = max(float(np.max(base_expected)), 1e-8)
        norm_expected_load = np.clip(
            expected_upload / base_expected_norm,
            0.0,
            2.0,
        ).astype(np.float32)
        norm_cpu_capacity = np.clip(
            np.asarray(self.env.F_max_array, dtype=np.float32)
            / 1e9
            / max(float(self.macro_cpu_norm_ghz), 1e-8),
            0.0,
            2.0,
        ).astype(np.float32)
        norm_energy_dual = np.clip(
            self.energy_lagrange_multipliers
            / max(self.energy_dual_state_norm, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
        norm_system_energy_dual = np.float32(
            np.clip(
                self.energy_lagrange_multiplier
                / max(self.energy_dual_state_norm, 1e-8),
                0.0,
                1.0,
            )
        )

        local_obs_list = []
        for i in range(self.N):
            delta_centroid = centroids[i] - uav_coords[i]

            own_summary_values = [
                norm_q_high[i],
                norm_q_low[i],
                norm_arrival[i],
                norm_outside[i],
                norm_u2u_mean[i],
            ]
            if self.macro_flow_state:
                own_summary_values.extend(
                    [
                        norm_flow_out[i],
                        norm_flow_in[i],
                    ]
                )
            if self.macro_load_state:
                own_summary_values.append(norm_expected_load[i])
            if self.macro_capacity_state:
                own_summary_values.append(norm_cpu_capacity[i])
            if self.power_constraint_scope == "per_uav":
                own_summary_values.append(norm_energy_dual[i])
            elif self.system_dual_state:
                own_summary_values.append(norm_system_energy_dual)
            own_summary = np.asarray(own_summary_values, dtype=np.float32)
            if self.macro_action_mode == "waypoint":
                waypoint_offset = (
                    uav_coords_m[i] - self.macro_waypoint_origins[i]
                ) / max(self.macro_waypoint_radius_m, 1e-8)
                own_summary = np.concatenate(
                    [own_summary, waypoint_offset.astype(np.float32)]
                )

            teammate_feature = []
            for j in range(self.N):
                if j == i:
                    continue
                teammate_values = [
                    uav_coords[j, 0] - uav_coords[i, 0],
                    uav_coords[j, 1] - uav_coords[i, 1],
                    norm_q_high[j],
                    norm_q_low[j],
                ]
                if self.macro_flow_state:
                    teammate_values.extend(
                        [
                            norm_flow[i, j],
                            norm_flow[j, i],
                        ]
                    )
                if self.macro_load_state:
                    teammate_values.append(norm_expected_load[j])
                if self.macro_capacity_state:
                    teammate_values.append(norm_cpu_capacity[j])
                if self.power_constraint_scope == "per_uav":
                    teammate_values.append(norm_energy_dual[j])
                teammate_feature.extend(teammate_values)

            obs = np.concatenate(
                [
                    delta_centroid,
                    own_summary,
                    np.asarray(teammate_feature, dtype=np.float32),
                ]
            )

            local_obs_list.append(obs.astype(np.float32))

        global_state = np.concatenate(local_obs_list).astype(np.float32)

        return local_obs_list, global_state

    def macro_step(self, action):
        """
        [MAPPO UPGRADE]
        Map normalized macro actions [rho, theta] to per-slot [dx, dy].
        """
        if self.strict_hierarchy_validation and self._macro_slot_active:
            raise RuntimeError(
                "Previous macro slot was not finalized with get_macro_reward()."
            )
        self._macro_slot_active = True
        self._macro_slot_index += 1
        raw_state_before = self.env.get_raw_state()
        flow_ref = self._estimate_macro_flow_reference(raw_state_before)
        pre_macro_cost_info, pre_macro_cost_info_per_uav = self._compute_macro_costs(
            raw_state_before,
            flow_ref,
        )
        self.prev_macro_positioning_cost = pre_macro_cost_info["total_cost"]
        self.prev_macro_positioning_cost_per_uav = pre_macro_cost_info_per_uav[
            "total_cost_per_uav"
        ].copy()

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        waypoint_targets = None
        waypoint_remaining = None
        if self.macro_action_mode == "delta":
            action_norm = np.linalg.norm(action, axis=1, keepdims=True)
            safe_norm = np.maximum(action_norm, 1.0)
            direction = action / safe_norm
            command_norm = np.minimum(action_norm[:, 0], 1.0)
            hover_mask = command_norm < self.macro_action_deadzone
            direction[hover_mask] = 0.0
            command_norm = np.where(hover_mask, 0.0, command_norm)
            macro_actions = (
                direction * self.max_macro_displacement
            ).astype(np.float32)
        elif self.macro_action_mode == "polar":
            rho_raw = action[:, 0]
            theta_raw = action[:, 1]

            rho = np.abs(rho_raw)
            hover_mask = rho < self.macro_action_deadzone
            rho = np.where(hover_mask, 0.0, rho)
            theta = np.pi * theta_raw

            command_norm = rho
            macro_actions = np.stack(
                [
                    rho * self.max_macro_displacement * np.cos(theta),
                    rho * self.max_macro_displacement * np.sin(theta),
                ],
                axis=1,
            ).astype(np.float32)
        elif self.macro_action_mode == "waypoint":
            action_norm = np.linalg.norm(action, axis=1, keepdims=True)
            safe_norm = np.maximum(action_norm, 1.0)
            bounded_action = action / safe_norm
            explicit_hover = action_norm[:, 0] < self.macro_action_deadzone

            waypoint_targets = np.clip(
                self.macro_waypoint_origins
                + bounded_action * self.macro_waypoint_radius_m,
                0.0,
                1000.0,
            ).astype(np.float32)
            to_target = waypoint_targets - np.asarray(
                raw_state_before["uav_coords"],
                dtype=np.float32,
            )
            waypoint_remaining = np.linalg.norm(
                to_target,
                axis=1,
                keepdims=True,
            )
            arrived = (
                waypoint_remaining[:, 0]
                <= self.macro_waypoint_tolerance_m
            )
            move_distance = np.minimum(
                waypoint_remaining[:, 0],
                self.max_macro_displacement,
            )
            move_direction = to_target / np.maximum(waypoint_remaining, 1e-8)
            macro_actions = (
                move_direction * move_distance[:, None]
            ).astype(np.float32)
            hover_mask = explicit_hover | arrived
            macro_actions[hover_mask] = 0.0
            command_norm = np.where(
                hover_mask,
                0.0,
                move_distance / max(self.max_macro_displacement, 1e-8),
            )
        else:
            action_norm = np.linalg.norm(action, axis=1, keepdims=True)
            safe_norm = np.maximum(action_norm, 1.0)
            bounded_action = action / safe_norm
            centroids = np.asarray(
                raw_state_before["weighted_centroids"],
                dtype=np.float32,
            )
            waypoint_targets = np.clip(
                centroids + bounded_action * self.macro_waypoint_radius_m,
                0.0,
                1000.0,
            ).astype(np.float32)
            to_target = waypoint_targets - np.asarray(
                raw_state_before["uav_coords"],
                dtype=np.float32,
            )
            waypoint_remaining = np.linalg.norm(
                to_target,
                axis=1,
                keepdims=True,
            )
            arrived = (
                waypoint_remaining[:, 0]
                <= self.macro_waypoint_tolerance_m
            )
            move_distance = np.minimum(
                waypoint_remaining[:, 0],
                self.max_macro_displacement,
            )
            move_direction = to_target / np.maximum(waypoint_remaining, 1e-8)
            macro_actions = (
                move_direction * move_distance[:, None]
            ).astype(np.float32)
            hover_mask = arrived
            macro_actions[hover_mask] = 0.0
            command_norm = np.where(
                hover_mask,
                0.0,
                move_distance / max(self.max_macro_displacement, 1e-8),
            )

        _, h_flying_penalty, mac_info = self.env.macro_step(macro_actions)

        mac_info = dict(mac_info)
        projected_uav_coords = np.asarray(
            mac_info.get(
                "projected_uav_coords",
                raw_state_before["uav_coords"],
            ),
            dtype=np.float32,
        )
        actual_move_distances = np.linalg.norm(
            projected_uav_coords - raw_state_before["uav_coords"],
            axis=1,
        ).astype(np.float32)

        projected_state = dict(raw_state_before)
        projected_state["uav_coords"] = projected_uav_coords.copy()
        self.current_macro_flow_reference_mbits = flow_ref.copy()
        projected_cost_info, projected_cost_info_per_uav = self._compute_macro_costs(
            projected_state,
            flow_ref,
        )

        self.current_flying_penalty = h_flying_penalty
        self.current_macro_movement_penalty_per_uav = (
            self.macro_move_penalty_weight
            * actual_move_distances
            / max(self.max_macro_displacement, 1e-8)
        ).astype(np.float32)
        flying_energy = np.asarray(
            mac_info.get("E_flying_per_uav", np.zeros(self.N)),
            dtype=np.float32,
        )
        hover_energy = np.asarray(
            mac_info.get("E_hover_per_uav", np.zeros(self.N)),
            dtype=np.float32,
        )
        # Compare the one-second flight phase with one second of hovering.
        self.current_macro_propulsion_delta_per_uav = (
            flying_energy - hover_energy / 9.0
        ).astype(np.float32)
        macro_micro_steps = max(int(self.env.micro_steps_per_macro), 1)
        self.current_macro_propulsion_energy_share = float(
            mac_info.get("E_macro_total", 0.0)
        ) / macro_micro_steps
        self.current_macro_propulsion_energy_per_uav_share = (
            np.asarray(
                mac_info.get("E_macro_per_uav", np.zeros(self.N)),
                dtype=np.float64,
            )
            / macro_micro_steps
        )
        self.current_macro_propulsion_delta_share = float(
            np.sum(self.current_macro_propulsion_delta_per_uav)
        ) / macro_micro_steps
        self.current_macro_action_norm_mean = float(np.mean(command_norm))
        self.current_macro_hover_ratio = float(
            np.mean(actual_move_distances < 1e-4)
        )
        self.accumulated_micro_reward = 0.0
        self.accumulated_micro_reward_per_uav.fill(0.0)
        self.accumulated_primal_cost = 0.0
        self.accumulated_energy_constraint_ratio = 0.0
        self.accumulated_energy_constraint_ratio_per_uav.fill(0.0)
        self.accumulated_upload_delay_penalty = 0.0
        self.accumulated_upload_delay_penalty_per_uav.fill(0.0)
        self.current_macro_micro_steps = 0
        self._reset_macro_flow_accumulator()

        mac_info["macro_action_norm_mean"] = self.current_macro_action_norm_mean
        mac_info["macro_move_distance_mean"] = float(
            np.mean(actual_move_distances)
        )
        mac_info["macro_hover_ratio"] = self.current_macro_hover_ratio
        mac_info["macro_propulsion_delta_per_uav"] = (
            self.current_macro_propulsion_delta_per_uav.copy()
        )
        mac_info["macro_propulsion_delta_total"] = float(
            np.sum(self.current_macro_propulsion_delta_per_uav)
        )
        mac_info["macro_propulsion_energy_per_micro_step"] = float(
            self.current_macro_propulsion_energy_share
        )
        mac_info["macro_propulsion_delta_per_micro_step"] = float(
            self.current_macro_propulsion_delta_share
        )
        mac_info["pre_macro_position_cost"] = pre_macro_cost_info[
            "total_cost"
        ]
        mac_info["pre_macro_position_cost_per_uav"] = pre_macro_cost_info_per_uav[
            "total_cost_per_uav"
        ].copy()
        mac_info["projected_macro_position_cost"] = projected_cost_info[
            "total_cost"
        ]
        mac_info["projected_macro_position_cost_per_uav"] = (
            projected_cost_info_per_uav["total_cost_per_uav"].copy()
        )
        mac_info["projected_mean_u2g_distance_m"] = projected_cost_info[
            "mean_u2g_distance_m"
        ]
        mac_info["projected_mean_u2u_rate_mbps"] = projected_cost_info[
            "mean_nominal_u2u_rate_mbps"
        ]
        mac_info["pre_macro_comm_cost"] = pre_macro_cost_info["comm_cost"]
        mac_info["projected_macro_comm_cost"] = projected_cost_info["comm_cost"]
        mac_info["projected_comm_uplink_time_s"] = projected_cost_info[
            "comm_uplink_time_s"
        ]
        mac_info["projected_comm_u2u_time_s"] = projected_cost_info[
            "comm_u2u_time_s"
        ]
        mac_info["flow_reference_mbits"] = flow_ref.copy()
        if waypoint_targets is not None:
            mac_info["macro_waypoint_targets_m"] = waypoint_targets.copy()
            mac_info["macro_waypoint_remaining_m"] = (
                waypoint_remaining[:, 0].copy()
            )

        return self.get_macro_state(), h_flying_penalty, mac_info

    def _compute_derived_goal(self, raw_state, u2u_observation):
        """
        [HIERARCHY] 由当前拓扑确定性推导的“管理者建议卸载分布” G (N x N)。

        G[i, j] = 建议把 UAV i 的负载路由到 j 的比例 (每行归一化)。
        attractiveness[i, j] = 队友 j 的可用度 × 归一化 u2u 速率(i->j);
        对角线视为完美本地链路, 吸引力 = 自身可用度。

        宏观层通过定位改变 u2u 速率与负载分布, 从而改变 G —— 这是宏观→微观的
        显式 manager→worker 耦合。goal 仅作为上下文注入微观状态, 不进入微观奖励,
        因此 SAC 仍只优化 J, 可自由遵循或偏离该建议。
        """
        mu_comp = (
            self.env.F_max_array * self.env.micro_dt / (self.env.C * 1e6)
        ).astype(np.float32)
        mu_comp = np.maximum(mu_comp, 1e-6)
        local_load = (
            raw_state["Q_loc_high"] + raw_state["Q_loc_low"]
        ).astype(np.float32)
        # 可用度: 本地队列越空越可用, 落在 (0, 1]
        availability = mu_comp / (mu_comp + local_load + 1e-6)

        rate_norm = np.asarray(u2u_observation, dtype=np.float32)
        # j 的可用度沿列广播, 乘以 i->j 链路质量
        attractiveness = availability[None, :] * rate_norm
        np.fill_diagonal(attractiveness, availability)
        attractiveness = np.maximum(attractiveness, 1e-8)
        goal = attractiveness / attractiveness.sum(axis=1, keepdims=True)
        return goal.astype(np.float32)

    def get_sac_state(self, raw_state=None):
        """
        [REWARD SHAPING]
        产生 SAC 网络的扁平化微观状态输入。

        废除原先的 log1p，统一采用阈值归一化 + clip，
        防止排队爆炸时状态空间 OOD 溢出。
        追加 [HIERARCHY] 宏观派生的建议卸载分布 goal (N^2)。
        """
        if raw_state is None:
            raw_state = self.env.get_raw_state()

        _, u2u_rate_mbps, _ = self._compute_u2u_topology_metrics(
            raw_state["uav_coords"],
            raw_state["Q_tx_high"] + raw_state["Q_tx_low"]
        )
        u2u_observation = self._normalized_u2u_observation(
            raw_state["uav_coords"],
            nominal_rate_mbps=u2u_rate_mbps,
        )
        goal_observation = self._goal_u2u_observation(
            raw_state["uav_coords"],
            nominal_rate_mbps=u2u_rate_mbps,
        )
        goal_matrix = self._compute_derived_goal(
            raw_state, goal_observation
        ).flatten()
        norm_u2u_rate = u2u_observation.flatten()

        if self.micro_state_mode == "pressure":
            mu_comp = (
                self.env.F_max_array * self.env.micro_dt
                / (self.env.C * 1e6)
            ).astype(np.float32)
            mu_comp = np.maximum(mu_comp, 1e-6)
            pressure_rate_mbps = u2u_rate_mbps
            if self.u2u_contention_aware_pressure:
                pressure_rate_mbps = self._estimated_u2u_rate_mbps(
                    raw_state["uav_coords"]
                )
            mu_u2u = np.maximum(
                pressure_rate_mbps * self.env.micro_dt,
                1e-6,
            ).astype(np.float32)
            pressure_clip = 10.0

            norm_Q_loc_high = np.clip(
                raw_state["Q_loc_high"] / mu_comp / pressure_clip,
                0.0,
                1.0,
            )
            norm_Q_loc_low = np.clip(
                raw_state["Q_loc_low"] / mu_comp / pressure_clip,
                0.0,
                1.0,
            )
            norm_Q_tx_high = np.clip(
                raw_state["Q_tx_high"] / mu_u2u / pressure_clip,
                0.0,
                1.0,
            ).flatten()
            norm_Q_tx_low = np.clip(
                raw_state["Q_tx_low"] / mu_u2u / pressure_clip,
                0.0,
                1.0,
            ).flatten()
            norm_A_t_high = np.clip(
                raw_state["A_t_high"] / mu_comp / pressure_clip,
                0.0,
                1.0,
            )
            norm_A_t_low = np.clip(
                raw_state["A_t_low"] / mu_comp / pressure_clip,
                0.0,
                1.0,
            )
        else:
            norm_Q_loc_high = np.clip(
                raw_state["Q_loc_high"] / self.max_Q_threshold,
                0.0,
                1.0,
            )
            norm_Q_loc_low = np.clip(
                raw_state["Q_loc_low"] / self.max_Q_threshold,
                0.0,
                1.0,
            )
            norm_Q_tx_high = np.clip(
                raw_state["Q_tx_high"].flatten() / self.max_Q_threshold,
                0.0,
                1.0,
            )
            norm_Q_tx_low = np.clip(
                raw_state["Q_tx_low"].flatten() / self.max_Q_threshold,
                0.0,
                1.0,
            )
            norm_A_t_high = np.clip(
                raw_state["A_t_high"] / self.max_Q_threshold,
                0.0,
                1.0,
            )
            norm_A_t_low = np.clip(
                raw_state["A_t_low"] / self.max_Q_threshold,
                0.0,
                1.0,
            )

        state_parts = [
            norm_Q_loc_high,
            norm_Q_loc_low,
            norm_Q_tx_high,
            norm_Q_tx_low,
            norm_A_t_high,
            norm_A_t_low,
            norm_u2u_rate,
            goal_matrix,
        ]
        if self.u2u_prev_active_state:
            state_parts.append(
                np.asarray(
                    [self._previous_u2u_active_fraction()],
                    dtype=np.float32,
                )
            )
        if self.power_constraint_scope == "per_uav":
            state_parts.append(
                np.clip(
                    self.energy_lagrange_multipliers
                    / max(self.energy_dual_state_norm, 1e-8),
                    0.0,
                    1.0,
                ).astype(np.float32)
            )
        elif self.system_dual_state:
            state_parts.append(
                np.asarray(
                    [
                        np.clip(
                            self.energy_lagrange_multiplier
                            / max(self.energy_dual_state_norm, 1e-8),
                            0.0,
                            1.0,
                        )
                    ],
                    dtype=np.float32,
                )
            )
        sac_state = np.concatenate(state_parts).astype(np.float32)

        return sac_state

    def micro_step(self, action):
        """
        [REWARD SHAPING]
        - 将连续动作投影到包含边界点的完整路由单纯形
        # 解析连续动作并映射为流量分配比例
        # 微观层输出: 2 * N * N (高优/低优路由矩阵)
        # 这里假设 action 被限制在 [-1, 1]
        """
        if self.strict_hierarchy_validation and not self._macro_slot_active:
            raise RuntimeError(
                "micro_step() requires an active macro slot in strict mode."
            )
        if self.force_local_routing:
            split_ratios_high = np.eye(self.N, dtype=np.float32)
            split_ratios_low = np.eye(self.N, dtype=np.float32)
        else:
            flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
            full_action_dim = 2 * self.N * self.N
            latent_action_dim = 2 * self.N * (self.N - 1)
            if (
                self.micro_action_encoding == "stick_breaking"
                and flat_action.size == latent_action_dim
            ):
                latent_action = np.clip(
                    flat_action.reshape(2, self.N, self.N - 1),
                    -1.0,
                    1.0,
                )
                split_ratios = centered_stick_breaking_to_simplex(
                    latent_action,
                    logit_scale=self.micro_stick_breaking_scale,
                )
                if self.micro_diag_bias != 0.0:
                    split_ratios += (
                        np.eye(self.N, dtype=np.float64)[None, ...]
                        * self.micro_diag_bias
                    )
                    split_ratios /= np.sum(
                        split_ratios,
                        axis=-1,
                        keepdims=True,
                    )
            elif flat_action.size == full_action_dim:
                action_matrix = (
                    flat_action.reshape(2, self.N, self.N)
                    * self.micro_action_scale
                )
                action_matrix = np.clip(
                    action_matrix,
                    -self.micro_action_scale,
                    self.micro_action_scale,
                )
                diag_bias = np.eye(self.N) * self.micro_diag_bias
                action_matrix += diag_bias
                split_ratios = project_rows_to_simplex(action_matrix)
            else:
                raise ValueError(
                    f"Micro action has {flat_action.size} elements; expected "
                    f"{latent_action_dim} for stick_breaking or "
                    f"{full_action_dim} for a full routing matrix."
                )
            split_ratios_high = split_ratios[0]
            split_ratios_low = split_ratios[1]
        split_ratios = np.stack(
            [split_ratios_high, split_ratios_low],
            axis=0,
        )

        self._sync_cpu_energy_prices()
        raw_state, info = self.env.micro_step(
            split_ratios_high,
            split_ratios_low,
        )

        Q_loc_h = np.sum(raw_state["Q_loc_high"])
        Q_tx_h = np.sum(raw_state["Q_tx_high"])
        Q_loc_l = np.sum(raw_state["Q_loc_low"])
        Q_tx_l = np.sum(raw_state["Q_tx_low"])
        service_energy = float(info["total_energy"])
        propulsion_energy_share = float(
            self.current_macro_propulsion_energy_share
        )
        propulsion_delta_share = float(
            self.current_macro_propulsion_delta_share
        )
        full_energy = service_energy + propulsion_energy_share
        battery_service_energy_per_uav = np.asarray(
            info.get("uav_battery_energy_per_uav", np.zeros(self.N)),
            dtype=np.float64,
        )
        full_battery_energy_per_uav = (
            battery_service_energy_per_uav
            + self.current_macro_propulsion_energy_per_uav_share
        )
        # Relative-to-hover propulsion energy can be negative at efficient
        # forward speeds; retain that physical saving instead of clipping it
        # independently in every micro step.
        controllable_energy = service_energy + propulsion_delta_share

        q_total_current = Q_loc_h + Q_tx_h + Q_loc_l + Q_tx_l

        # ============ [UNIFIED J] 微观奖励 = -(ω_D·加权时延 + ω_E·能耗) ============
        # 时延代理: 在线加权 holding cost (Little 定律, 已含 10:1 优先级权重),
        # 取代之前未加权的 task_holding_cost。
        weighted_holding = float(info.get("task_holding_cost_weighted", 0.0))
        cost_delay = float(
            np.clip(
                weighted_holding / max(self.J_delay_norm, 1e-8),
                0.0,
                self.J_cost_clip,
            )
        )
        cost_energy = float(
            np.clip(
                controllable_energy / max(self.J_energy_norm, 1e-8),
                -self.J_cost_clip,
                self.J_cost_clip,
            )
        )
        delay_objective_step = self.J_w_delay * cost_delay
        energy_constraint_ratio_per_uav = np.zeros(
            self.N,
            dtype=np.float64,
        )
        service_energy_constraint_ratio = 0.0
        energy_budget_step_per_uav = np.zeros(
            self.N,
            dtype=np.float64,
        )
        if self.power_constraint_scope == "per_uav":
            energy_budget_step_per_uav = (
                self.uav_power_budgets_w * float(self.env.micro_dt)
            )
            energy_constraint_ratio_per_uav = np.divide(
                full_battery_energy_per_uav,
                np.maximum(energy_budget_step_per_uav, 1e-8),
            )
            energy_budget_step = float(np.sum(energy_budget_step_per_uav))
            service_energy_constraint_ratio = service_energy / max(
                energy_budget_step,
                1e-8,
            )
            energy_constraint_ratio = float(
                np.mean(energy_constraint_ratio_per_uav)
            )
            energy_constraint_violation = float(
                np.max(energy_constraint_ratio_per_uav - 1.0)
            )
            lagrangian_energy_cost = float(
                np.dot(
                    self.energy_lagrange_multipliers,
                    normalized_power_residual(
                        energy_constraint_ratio_per_uav
                    ),
                )
            )
        elif self.power_budget_w > 0.0:
            energy_budget_step = (
                self.power_budget_w * float(self.env.micro_dt)
            )
            service_energy_constraint_ratio = service_energy / max(
                energy_budget_step,
                1e-8,
            )
            energy_constraint_ratio = full_energy / max(
                energy_budget_step,
                1e-8,
            )
            energy_constraint_violation = energy_constraint_ratio - 1.0
            energy_constraint_ratio_per_uav.fill(energy_constraint_ratio)
            energy_budget_step_per_uav.fill(
                energy_budget_step / max(self.N, 1)
            )
            lagrangian_energy_cost = float(
                self.energy_lagrange_multiplier
                * normalized_power_residual(energy_constraint_ratio)
            )
        else:
            energy_budget_step = 0.0
            energy_constraint_ratio = 0.0
            energy_constraint_violation = 0.0
            lagrangian_energy_cost = 0.0
        # Lyapunov drift (默认关闭): 仅惩罚队列增长, 维持稳定性。
        q_drift = (
            (q_total_current ** 2 - self.prev_q_total ** 2)
            / max(self.J_drift_norm ** 2, 1e-8)
        )
        drift_cost = self.J_drift_weight * max(float(q_drift), 0.0)
        # 有界溢出惩罚取代 -50 尖峰; 不再因排队爆炸而提前终止 (连续型排队任务,
        # 每个 episode 固定时长, SAC 始终自举 -> 同时修掉 buffer 不存 done 的隐患)。
        overflow = max(
            q_total_current - self.max_Q_threshold, 0.0
        ) / max(self.max_Q_threshold, 1e-8)
        overflow_cost = self.overflow_penalty_weight * float(
            np.clip(overflow, 0.0, 2.0)
        )

        primal_cost_step = delay_objective_step + drift_cost + overflow_cost
        if self.objective_mode == "constrained_delay":
            J_base_step = delay_objective_step + lagrangian_energy_cost
        else:
            J_base_step = (
                delay_objective_step + self.J_w_energy * cost_energy
            )
        # J_step is the complete cost optimized by the policy and reported by
        # evaluation.  Keeping the overflow term outside J previously made
        # high-load training and evaluation optimize different quantities.
        J_step = J_base_step + drift_cost + overflow_cost
        micro_cost_raw = J_step
        micro_reward = -J_step * self.micro_reward_scale
        done = False
        self.prev_q_total = float(q_total_current)

        # 兼容旧日志字段
        cost_queue = (
            10.0 * Q_loc_h + 10.0 * Q_tx_h + 1.0 * Q_loc_l + 1.0 * Q_tx_l
        ) / self.max_Q_threshold
        cost_holding = cost_delay

        info = dict(info)
        current_coop_flow = np.asarray(
            info.get("coop_flow_mbits", self._last_coop_flow_mbits),
            dtype=np.float32,
        )
        if current_coop_flow.shape == (self.N, self.N):
            delta_flow = np.maximum(
                current_coop_flow - self._last_coop_flow_mbits,
                0.0,
            )
            if float(np.sum(delta_flow)) > 1e-9:
                self.curr_macro_flow_mbits += delta_flow.astype(np.float32)
                self.curr_macro_flow_dirty = True
            self._last_coop_flow_mbits = current_coop_flow.copy()
        info["q_total_current"] = q_total_current
        info["max_Q_threshold"] = self.max_Q_threshold
        info["cost_delay"] = cost_delay
        info["cost_holding"] = cost_holding
        # Backward-compatible alias for old plotting and logs.
        info["cost_wait"] = cost_holding
        info["cost_queue"] = cost_queue
        info["cost_energy"] = cost_energy
        info["objective_mode"] = self.objective_mode
        info["delay_objective_step"] = float(delay_objective_step)
        info["service_energy_step_j"] = service_energy
        info["propulsion_energy_share_step_j"] = propulsion_energy_share
        info["propulsion_delta_share_step_j"] = propulsion_delta_share
        info["controllable_energy_step_j"] = controllable_energy
        info["full_energy_step_j"] = full_energy
        info["uav_battery_energy_step_j"] = float(
            np.sum(full_battery_energy_per_uav)
        )
        info["uav_battery_energy_per_uav_step_j"] = (
            full_battery_energy_per_uav.copy()
        )
        info["power_budget_step_j"] = energy_budget_step
        info["power_budget_per_uav_step_j"] = (
            energy_budget_step_per_uav.copy()
        )
        info["energy_constraint_ratio"] = energy_constraint_ratio
        info["service_energy_constraint_ratio"] = float(
            service_energy_constraint_ratio
        )
        info["energy_constraint_ratio_per_uav"] = (
            energy_constraint_ratio_per_uav.copy()
        )
        info["energy_constraint_violation"] = energy_constraint_violation
        info["power_constraint_scope"] = self.power_constraint_scope
        info["energy_lagrange_multiplier"] = float(
            self.energy_lagrange_multiplier
        )
        info["energy_lagrange_multipliers"] = (
            self.energy_lagrange_multipliers.copy()
        )
        info["lagrangian_energy_cost"] = float(lagrangian_energy_cost)
        info["primal_cost_step"] = float(primal_cost_step)
        info["J_base_step"] = J_base_step
        info["J_step"] = J_step
        info["drift_cost"] = drift_cost
        info["overflow_cost"] = overflow_cost
        info["micro_cost_raw"] = micro_cost_raw
        info["micro_reward_unscaled"] = -micro_cost_raw
        info["micro_action_scale"] = self.micro_action_scale
        info["micro_action_encoding"] = self.micro_action_encoding
        route_entropy = -np.sum(
            split_ratios
            * np.log(np.maximum(split_ratios, 1e-12)),
            axis=-1,
        )
        info["micro_route_entropy_mean"] = float(np.mean(route_entropy))
        info["micro_route_boundary_fraction"] = float(
            np.mean(
                (split_ratios <= 0.01)
                | (split_ratios >= 0.99)
            )
        )
        info["micro_route_max_ratio"] = float(np.max(split_ratios))
        info["micro_holding_weight"] = self.micro_holding_weight
        info["micro_queue_weight"] = self.micro_queue_weight
        info["micro_energy_weight"] = self.micro_energy_weight
        info["micro_reward_scaled"] = micro_reward
        info["micro_reward_scale"] = self.micro_reward_scale
        info["early_stop_triggered"] = False

        # [REWARD SHAPING] upload delay 同样按阈值缩放，避免宏观奖励量级失衡
        upload_delay_penalty = - float(
            info.get("upload_delay_cost", 0.0)
        ) / self.max_Q_threshold
        info["upload_delay_penalty"] = upload_delay_penalty

        q_loc_total = raw_state["Q_loc_high"] + raw_state["Q_loc_low"]
        q_tx_in_total = np.sum(
            raw_state["Q_tx_high"] + raw_state["Q_tx_low"],
            axis=0,
        )
        q_tx_out_total = np.sum(
            raw_state["Q_tx_high"] + raw_state["Q_tx_low"],
            axis=1,
        )
        # ============ [UNIFIED J - per UAV] feudal 信用分配 ============
        # 每架 UAV 的局部代价 = 自身加权 backlog 压力(时延代理) + 自身能耗。
        # 这给 MAPPO 提供真正的 per-agent 区分度 (取代之前所有 UAV 同一标量)。
        pressure_per_uav = np.asarray(
            info.get("pressure_per_uav", np.zeros(self.N)),
            dtype=np.float32,
        )
        if self.power_constraint_scope == "per_uav":
            energy_total_per_uav = full_battery_energy_per_uav.astype(
                np.float32
            )
        else:
            energy_total_per_uav = np.asarray(
                info.get("energy_total_per_uav", np.zeros(self.N)),
                dtype=np.float32,
            )
        cost_delay_pu = np.clip(
            pressure_per_uav / max(self.J_pressure_norm, 1e-8),
            0.0,
            self.J_cost_clip,
        )
        cost_energy_pu = np.clip(
            energy_total_per_uav / max(self.J_energy_pu_norm, 1e-8),
            0.0,
            self.J_cost_clip,
        )
        if (
            self.objective_mode == "constrained_delay"
            and self.power_constraint_scope == "per_uav"
        ):
            J_step_pu = (
                self.J_w_delay * cost_delay_pu
                + self.energy_lagrange_multipliers
                * normalized_power_residual(
                    energy_constraint_ratio_per_uav
                )
            ).astype(np.float32)
        else:
            J_step_pu = (
                self.J_w_delay * cost_delay_pu
                + self.J_w_energy * cost_energy_pu
            ).astype(np.float32)
        micro_reward_per_uav = (
            -J_step_pu * self.micro_reward_scale
        ).astype(np.float32)

        upload_delay_cost_per_uav = np.asarray(
            info.get("upload_delay_cost_per_uav", np.zeros(self.N)),
            dtype=np.float32,
        )
        upload_delay_penalty_per_uav = (
            -upload_delay_cost_per_uav / self.max_Q_threshold
        ).astype(np.float32)
        cpu_util_per_uav = np.asarray(
            info.get("cpu_utilization_per_uav", np.zeros(self.N)),
            dtype=np.float32,
        )
        u2g_distance = np.linalg.norm(
            raw_state["uav_coords"] - raw_state["weighted_centroids"],
            axis=1,
        ).astype(np.float32)
        arrival_total = np.asarray(
            info.get("arrival_total_per_uav", np.zeros(self.N)),
            dtype=np.float32,
        )

        self.curr_macro_q_up_sum += arrival_total
        self.curr_macro_q_loc_sum += q_loc_total.astype(np.float32)
        self.curr_macro_q_tx_out_sum += q_tx_out_total.astype(np.float32)
        self.curr_macro_cpu_util_sum += cpu_util_per_uav
        self.curr_macro_u2g_sum += u2g_distance
        self.curr_macro_stat_count += 1
        self.curr_macro_stats_dirty = True

        self.accumulated_micro_reward += micro_reward
        self.accumulated_primal_cost += float(primal_cost_step)
        self.accumulated_energy_constraint_ratio += float(
            energy_constraint_ratio
        )
        self.accumulated_energy_constraint_ratio_per_uav += (
            energy_constraint_ratio_per_uav
        )
        self.accumulated_micro_reward_per_uav += micro_reward_per_uav
        self.accumulated_upload_delay_penalty += upload_delay_penalty
        self.accumulated_upload_delay_penalty_per_uav += (
            upload_delay_penalty_per_uav
        )
        self.current_macro_micro_steps += 1

        info["Q_tx_in_total_per_uav"] = q_tx_in_total.astype(np.float32)
        info["micro_reward_per_uav"] = micro_reward_per_uav.copy()
        info["upload_delay_penalty_per_uav"] = (
            upload_delay_penalty_per_uav.copy()
        )

        # [PERF] 暴露本步已有的 raw_state, 供训练/评估循环复用,
        # 避免再调用 env.get_raw_state() 多复制一遍所有队列数组。
        self.last_raw_state = raw_state

        return self.get_sac_state(raw_state), micro_reward, done, False, info

    def prepare_resource_frame(self):
        """Expose service-frame arrivals and mobility before policy inference."""
        raw_state = self.env.prepare_service_resource_frame()
        self.last_raw_state = raw_state
        return self.get_sac_state(raw_state)

    def step_resource_frame(self, action):
        """Hold one routing action for five 0.1 s physical integration steps."""
        frame_steps = int(self.env.resource_frame_steps)
        if self.env.global_micro_step % frame_steps != 0:
            raise RuntimeError("Resource-frame stepping requires an aligned start.")

        service_paused = bool(
            self.env.pause_service_during_flight
            and self.env.macro_service_pause_steps_remaining > 0
        )
        if not service_paused:
            self.prepare_resource_frame()

        held_action = np.asarray(action, dtype=np.float32).copy()
        infos = []
        raw_states = []
        rewards = []
        next_state = self.get_sac_state()
        done = False
        for _ in range(frame_steps):
            next_state, reward, step_done, _, info = self.micro_step(held_action)
            infos.append(info)
            raw_states.append(
                {
                    key: np.asarray(value).copy()
                    for key, value in self.last_raw_state.items()
                }
            )
            rewards.append(float(reward))
            done = done or bool(step_done)

        self._update_previous_u2u_contention(infos)
        if self.last_raw_state is not None:
            next_state = self.get_sac_state(self.last_raw_state)

        primal_reward_sum = -self.micro_reward_scale * sum(
            float(info.get("primal_cost_step", 0.0)) for info in infos
        )
        holding_reward_sum = -self.micro_reward_scale * sum(
            float(info.get("delay_objective_step", 0.0)) for info in infos
        )
        energy_constraint_ratio_sum = float(
            sum(
                float(info.get("energy_constraint_ratio", 0.0))
                for info in infos
            )
        )
        service_energy_constraint_ratio_sum = float(
            sum(
                float(info.get("service_energy_constraint_ratio", 0.0))
                for info in infos
            )
        )
        energy_constraint_ratio_per_uav_sum = np.sum(
            [
                np.asarray(
                    info.get(
                        "energy_constraint_ratio_per_uav",
                        np.zeros(self.N),
                    ),
                    dtype=np.float64,
                )
                for info in infos
            ],
            axis=0,
        )
        return {
            "phase": "deployment" if service_paused else "service",
            "duration_s": frame_steps * float(self.env.micro_dt),
            "substeps": frame_steps,
            "next_state": np.asarray(next_state, dtype=np.float32),
            "reward_sum": float(np.sum(rewards)),
            "primal_reward_sum": float(primal_reward_sum),
            "holding_reward_sum": float(holding_reward_sum),
            "energy_constraint_ratio_sum": energy_constraint_ratio_sum,
            "energy_constraint_ratio_mean": (
                energy_constraint_ratio_sum / max(frame_steps, 1)
            ),
            "service_energy_constraint_ratio_sum": (
                service_energy_constraint_ratio_sum
            ),
            "service_energy_constraint_ratio_mean": (
                service_energy_constraint_ratio_sum / max(frame_steps, 1)
            ),
            "energy_constraint_ratio_per_uav_sum": (
                energy_constraint_ratio_per_uav_sum
            ),
            "energy_constraint_ratio_per_uav_mean": (
                energy_constraint_ratio_per_uav_sum / max(frame_steps, 1)
            ),
            "done": bool(done),
            "infos": infos,
            "raw_states": raw_states,
            "rewards": np.asarray(rewards, dtype=np.float32),
        }

    def get_macro_reward(self):
        """
        Finalize one macro slot from the micro rewards actually observed in it.

        ``strict_global`` uses the exact mean of the global micro reward as the
        centralized PPO/TD3 task signal.  The legacy per-UAV pressure proxy is
        retained only for reproducing historical checkpoints.
        """
        expected_micro_steps = int(self.env.micro_steps_per_macro)
        if self.strict_hierarchy_validation:
            if not self._macro_slot_active:
                raise RuntimeError(
                    "get_macro_reward() called without an active macro slot."
                )
            if self.current_macro_micro_steps != expected_micro_steps:
                raise RuntimeError(
                    "Macro slot executed "
                    f"{self.current_macro_micro_steps} micro steps; expected "
                    f"{expected_micro_steps}."
                )
        effective_micro_steps = max(self.current_macro_micro_steps, 1)

        avg_micro_reward = (
            self.accumulated_micro_reward / effective_micro_steps
        )
        avg_primal_cost = (
            self.accumulated_primal_cost / effective_micro_steps
        )
        avg_energy_constraint_ratio = (
            self.accumulated_energy_constraint_ratio / effective_micro_steps
        )
        avg_energy_constraint_ratio_per_uav = (
            self.accumulated_energy_constraint_ratio_per_uav
            / effective_micro_steps
        )
        avg_micro_reward_per_uav = (
            self.accumulated_micro_reward_per_uav / effective_micro_steps
        ).astype(np.float32)
        avg_upload_delay_penalty = (
            self.accumulated_upload_delay_penalty / effective_micro_steps
        )
        avg_upload_delay_penalty_per_uav = (
            self.accumulated_upload_delay_penalty_per_uav
            / effective_micro_steps
        ).astype(np.float32)

        flow_ref = self.current_macro_flow_reference_mbits.copy()
        current_macro_cost_info, current_macro_cost_info_per_uav = self._compute_macro_costs(
            self.env.get_raw_state(),
            flow_ref,
        )
        self.current_macro_positioning_cost = current_macro_cost_info[
            "total_cost"
        ]
        self.current_macro_positioning_cost_per_uav = (
            current_macro_cost_info_per_uav["total_cost_per_uav"].copy()
        )
        self.current_macro_positioning_improvement_per_uav = (
            self.prev_macro_positioning_cost_per_uav
            - self.current_macro_positioning_cost_per_uav
        ).astype(np.float32)

        # 几何定位降级为规范势能 shaping: F = γΦ' - Φ, Φ = -几何代价。
        # 势能 shaping 不改变最优策略(Ng et al. 1999), 仅加速早期探索,
        # macro_shaping_lambda 可随训练退火至 0, 收敛时只剩纯 J。
        phi_prev = -self.prev_macro_positioning_cost_per_uav
        phi_curr = -self.current_macro_positioning_cost_per_uav
        shaping_per_uav = (
            self.macro_shaping_lambda
            * (self.macro_gamma * phi_curr - phi_prev)
        ).astype(np.float32)
        propulsion_penalty_per_uav = (
            float(self.macro_propulsion_tiebreak_weight)
            * self.current_macro_propulsion_delta_per_uav
            / max(float(self.macro_propulsion_energy_norm), 1e-8)
        ).astype(np.float32)

        if self.macro_reward_mode == "strict_global":
            global_micro_signal = float(
                self.macro_feudal_weight * avg_micro_reward
            )
            shaping_signal = float(np.mean(shaping_per_uav))
            movement_penalty = float(
                np.mean(self.current_macro_movement_penalty_per_uav)
            )
            propulsion_penalty = float(np.mean(propulsion_penalty_per_uav))
            macro_reward = float(
                global_micro_signal
                + shaping_signal
                - movement_penalty
                - propulsion_penalty
            )
            macro_reward_per_uav = np.full(
                self.N,
                macro_reward,
                dtype=np.float32,
            )
            feudal_per_uav = np.full(
                self.N,
                global_micro_signal,
                dtype=np.float32,
            )
            upload_signal = 0.0
        else:
            feudal_per_uav = (
                self.macro_feudal_weight * avg_micro_reward_per_uav
                + self.macro_upload_penalty_weight
                * avg_upload_delay_penalty_per_uav
            ).astype(np.float32)
            macro_reward_per_uav = (
                feudal_per_uav
                + shaping_per_uav
                - self.current_macro_movement_penalty_per_uav
                - propulsion_penalty_per_uav
            ).astype(np.float32)
            macro_reward = float(np.mean(macro_reward_per_uav))
            global_micro_signal = float(np.mean(feudal_per_uav))
            shaping_signal = float(np.mean(shaping_per_uav))
            movement_penalty = float(
                np.mean(self.current_macro_movement_penalty_per_uav)
            )
            propulsion_penalty = float(np.mean(propulsion_penalty_per_uav))
            upload_signal = float(
                self.macro_upload_penalty_weight
                * np.mean(avg_upload_delay_penalty_per_uav)
            )

        reconstructed_reward = float(
            global_micro_signal
            + shaping_signal
            - movement_penalty
            - propulsion_penalty
        )
        reward_reconstruction_error = float(
            macro_reward - reconstructed_reward
        )

        realized_macro_flow_mbits = self.curr_macro_flow_mbits.copy()
        self._update_macro_flow_ema()
        self._macro_slot_active = False

        self.last_macro_reward_info = {
            "macro_reward": float(macro_reward),
            "macro_rewards_per_uav": macro_reward_per_uav.copy(),
            "macro_reward_mode": self.macro_reward_mode,
            "macro_slot_index": int(self._macro_slot_index),
            "micro_steps_executed": int(self.current_macro_micro_steps),
            "micro_steps_expected": int(expected_micro_steps),
            "micro_reward_sum": float(self.accumulated_micro_reward),
            "micro_reward_mean": float(avg_micro_reward),
            "primal_cost_mean": float(avg_primal_cost),
            "energy_constraint_ratio_mean": float(
                avg_energy_constraint_ratio
            ),
            "energy_constraint_ratio_per_uav_mean": (
                avg_energy_constraint_ratio_per_uav.copy()
            ),
            "power_constraint_scope": self.power_constraint_scope,
            "energy_lagrange_multipliers": (
                self.energy_lagrange_multipliers.copy()
            ),
            "global_micro_signal": float(global_micro_signal),
            "upload_signal": float(upload_signal),
            "reward_reconstructed": float(reconstructed_reward),
            "reward_reconstruction_error": reward_reconstruction_error,
            "positioning_improvement": float(
                np.mean(self.current_macro_positioning_improvement_per_uav)
            ),
            "positioning_improvement_per_uav": (
                self.current_macro_positioning_improvement_per_uav.copy()
            ),
            "movement_penalty": float(
                np.mean(self.current_macro_movement_penalty_per_uav)
            ),
            "movement_penalty_per_uav": (
                self.current_macro_movement_penalty_per_uav.copy()
            ),
            "propulsion_penalty": float(propulsion_penalty),
            "propulsion_penalty_per_uav": propulsion_penalty_per_uav.copy(),
            "propulsion_delta_energy": float(
                np.mean(self.current_macro_propulsion_delta_per_uav)
            ),
            "propulsion_delta_energy_per_uav": (
                self.current_macro_propulsion_delta_per_uav.copy()
            ),
            "propulsion_tiebreak_weight": float(
                self.macro_propulsion_tiebreak_weight
            ),
            "avg_micro_reward": float(avg_micro_reward),
            "avg_micro_reward_per_uav": avg_micro_reward_per_uav.copy(),
            "feudal_signal": float(np.mean(feudal_per_uav)),
            "shaping_signal": float(shaping_signal),
            "avg_upload_delay_penalty": float(avg_upload_delay_penalty),
            "avg_upload_delay_penalty_per_uav": (
                avg_upload_delay_penalty_per_uav.copy()
            ),
            "macro_action_norm_mean": float(
                self.current_macro_action_norm_mean
            ),
            "macro_hover_ratio": float(self.current_macro_hover_ratio),
            "env_flying_penalty": float(self.current_flying_penalty),
            "realized_macro_flow_mbits": realized_macro_flow_mbits,
            "macro_flow_ema_mbits": self.prev_macro_flow_ema_mbits.copy(),
            **current_macro_cost_info,
            **current_macro_cost_info_per_uav,
        }

        return macro_reward

    def get_last_macro_reward_info(self):
        """
        [MAPPO UPGRADE]
        返回最近一个宏观时隙的奖励/代价分解。
        """
        return dict(self.last_macro_reward_info)

    def render_trajectory(self, *args, **kwargs):
        """
        无缝透传调用底层的轨迹渲染模块。
        """
        return self.env.render_trajectory(*args, **kwargs)

    def record_trajectory_snapshot(self):
        return self.env.record_trajectory_snapshot()
