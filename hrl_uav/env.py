import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

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
        arrival_dt=0.1,
        uplink_bandwidth_mhz=20.0,
        u2u_bandwidth_mhz=3.0,
        cpu_alloc_mode="priority_kkt",
        kkt_v=1.0,
        fixed_cpu_util=0.5,
        load_scale=1.0,
        zone_loads=None,
    ):

        # ================== 1. 系统物理设定与参数 ==================
        self.macro_dt = 10.0   # 宏观时隙 10 秒
        self.micro_dt = 0.1    # 微观时隙 0.1 秒
        self.micro_steps_per_macro = int(self.macro_dt / self.micro_dt)
        
        self.N = num_uavs      # 无人机数量 (默认 3)
        self.M = num_users     # 真实地面用户总数量 (默认 39)
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
        self.traffic_profile = str(traffic_profile).lower()
        self.arrival_dt = max(float(arrival_dt), float(self.micro_dt))
        self.arrival_interval_steps = max(
            1,
            int(round(self.arrival_dt / self.micro_dt)),
        )
        self.arrival_dt = self.arrival_interval_steps * float(self.micro_dt)
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
        self.task_size_high_mbits = 1.0
        self.task_size_low_mbits = 1.0
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
        self.kkt_bisection_iters = 24
        self.fixed_cpu_util = float(fixed_cpu_util)
        self.load_scale = float(load_scale)
        self.zone_target_load_mbps = self._normalize_zone_loads(zone_loads)
        self.valid_cpu_alloc_modes = {
            "legacy_kkt",
            "priority_kkt",
            "fixed_full",
            "fixed_average",
            "fixed_demand",
            "priority_demand",
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
        self.uav_initial_positions = self.base_anchor_points.copy()
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
        self.centroid_ema_beta = 0.99

        self.uav_coords = np.zeros((self.N, 2))
        
        # [军事物理升级] 高斯-马尔可夫移动模型 (GMMM) 参数
        self.alpha_gmmm = 0.8                     # 速度记忆衰减系数
        self.max_infantry_speed = 3.0             # 步兵最大战术推进速度 3m/s
        self.user_velocities = np.zeros((self.M, 2)) # 记录用户的当前速度向量
        self.users_static = True
        self.user_zone_radius = 80.0
        self.user_zone_scatter_std = 40.0
        self.user_zone_restore_gain = 0.015
        self.user_zone_restore_gain_gain = 0.015 # Just in case
        
        # 用于宏观-微观时隙解耦的缓存向量
        self.current_macro_v = np.zeros(self.N)
        self.current_macro_theta = np.zeros(self.N)
        self.macro_fly_steps_remaining = 0
        
        # 【双轨优先级流体队列】
        # _high: 存放 30% 到达的高优先紧急数据
        # _low:  存放 70% 到达的低优先宽带数据
        self.Q_loc_high = np.zeros(self.N)
        self.Q_loc_low = np.zeros(self.N)
        
        self.Q_tx_high = np.zeros((self.N, self.N))
        self.Q_tx_low = np.zeros((self.N, self.N))
        self.pending_upload_jobs = []
        self.global_micro_step = 0
        self._u2u_spectral_cache_coords = None
        self._u2u_spectral_cache_result = None
        self.current_arrival_high = np.zeros(self.N)
        self.current_arrival_low = np.zeros(self.N)
        self.current_arrival_active_users = np.zeros(self.N)
        
        self.A_t_high = np.zeros(self.N)
        self.A_t_low = np.zeros(self.N)
        self._init_task_tracker()
        
        # 3. UAV 部署在区域外围的三角形初始位置（以 (500,500) 为中心）
        self.uav_coords = self.uav_initial_positions.copy()

        self.record_trajectory = record_trajectory
        if self.record_trajectory:
            self.uav_trajectory = []
            self.centroid_trajectory = []
            self.users_trajectory = []

    def record_trajectory_snapshot(self):
        if not self.record_trajectory:
            return
        self.uav_trajectory.append(self.uav_coords.copy())
        self.centroid_trajectory.append(self.weighted_centroids.copy())
        self.users_trajectory.append(self.user_coords.copy())

    def _current_time(self):
        return float(self.global_micro_step) * float(self.micro_dt)

    def _new_loc_task_queues(self):
        return [deque() for _ in range(self.N)]

    def _new_tx_task_queues(self):
        return [[deque() for _ in range(self.N)] for _ in range(self.N)]

    def _init_task_tracker(self):
        self.parent_tasks = {}
        self.next_task_id = 0
        self.next_chunk_id = 0
        self.current_arrival_tasks_by_uav = [[] for _ in range(self.N)]
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

    def _sync_aggregate_queues_from_tasks(self):
        for i in range(self.N):
            self.Q_loc_high[i] = sum(
                chunk.remaining_mbits for chunk in self.Q_loc_high_tasks[i]
            )
            self.Q_loc_low[i] = sum(
                chunk.remaining_mbits for chunk in self.Q_loc_low_tasks[i]
            )

        for i in range(self.N):
            for j in range(self.N):
                self.Q_tx_high[i, j] = sum(
                    chunk.remaining_mbits for chunk in self.Q_tx_high_tasks[i][j]
                )
                self.Q_tx_low[i, j] = sum(
                    chunk.remaining_mbits for chunk in self.Q_tx_low_tasks[i][j]
                )

    def _create_parent_task(self, user_id, priority, size_mbits):
        task_id = self.next_task_id
        self.next_task_id += 1

        source_uav = int(self.user_allocations[user_id])
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

            raw_zone_load = float(
                np.sum(
                    self.user_lambda_high[zone_users]
                    * self.user_task_size_high_mbits[zone_users]
                    + self.user_lambda_low[zone_users]
                    * self.user_task_size_low_mbits[zone_users]
                )
            )
            target_zone_load = float(self._get_zone_target_load_mbps(zone_idx))

            if raw_zone_load > 1e-8:
                zone_rate_scale = max(target_zone_load / raw_zone_load, 1e-6)
                self.user_lambda_high[zone_users] *= zone_rate_scale
                self.user_lambda_low[zone_users] *= zone_rate_scale
                raw_zone_load = float(
                    np.sum(
                        self.user_lambda_high[zone_users]
                        * self.user_task_size_high_mbits[zone_users]
                        + self.user_lambda_low[zone_users]
                        * self.user_task_size_low_mbits[zone_users]
                    )
                )

            self.zone_expected_offered_load_mbps[zone_idx] = raw_zone_load

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
        
        # 限制在各自的管辖半径内
        zone_dist = np.linalg.norm(self.user_offsets, axis=1, keepdims=True)
        safe_zone_dist = np.maximum(zone_dist, 1e-8)
        self.user_offsets = np.where(
            zone_dist > self.user_zone_radius,
            self.user_offsets / safe_zone_dist * self.user_zone_radius,
            self.user_offsets,
        )
        
        self.user_coords = assigned_anchors + self.user_offsets
        self.user_coords = np.clip(self.user_coords, 0.0, 1000.0)
        
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
        
        # 4. 初始化 GMMM 战术移动初速度
        self.instantaneous_centroids = self.weighted_centroids.copy()

        if self.users_static:
            self.user_velocities = np.zeros((self.M, 2))
        else:
            self.user_velocities = self.np_random.uniform(
                -self.max_infantry_speed / 2,
                self.max_infantry_speed / 2,
                size=(self.M, 2),
            )
        
        # 5. 清空控制指令缓存与流水队列
        self.current_macro_v = np.zeros(self.N)
        self.current_macro_theta = np.zeros(self.N)
        self.macro_fly_steps_remaining = 0

        self.Q_loc_high = np.zeros(self.N)
        self.Q_loc_low = np.zeros(self.N)
        self.Q_tx_high = np.zeros((self.N, self.N))
        self.Q_tx_low = np.zeros((self.N, self.N))
        self.pending_upload_jobs = []
        self.global_micro_step = 0
        self._u2u_spectral_cache_coords = None
        self._u2u_spectral_cache_result = None
        self.current_arrival_high = np.zeros(self.N)
        self.current_arrival_low = np.zeros(self.N)
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
            'A_t_low': getattr(self, 'A_t_low', np.zeros(self.N)).copy()
        }

    def _generate_traffic(self):
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
                )
            for _ in range(low_count):
                self._create_parent_task(
                    user_idx,
                    "low",
                    self.user_task_size_low_mbits[user_idx],
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
            zone_mask = self.user_allocations == i
            A_t_total_high[i] = np.sum(high_bits[zone_mask])
            A_t_total_low[i] = np.sum(low_bits[zone_mask])
            active_users_total[i] = np.sum((counts_high[zone_mask] + counts_low[zone_mask]) > 0)
            
        self.current_arrival_high = A_t_total_high
        self.current_arrival_low = A_t_total_low
        self.current_arrival_active_users = active_users_total
        self.A_t_high = A_t_total_high
        self.A_t_low = A_t_total_low

    def _clear_current_arrivals(self, discard_tasks=True):
        if discard_tasks:
            self._discard_current_arrival_tasks()
        else:
            self.current_arrival_tasks_by_uav = [[] for _ in range(self.N)]
        self.current_arrival_high = np.zeros(self.N)
        self.current_arrival_low = np.zeros(self.N)
        self.current_arrival_active_users = np.zeros(self.N)
        self.A_t_high = np.zeros(self.N)
        self.A_t_low = np.zeros(self.N)

    def _build_paused_service_info(self):
        zeros_n = np.zeros(self.N, dtype=np.float32)

        q_high_tot = np.sum(self.Q_loc_high) + np.sum(self.Q_tx_high)
        q_low_tot = np.sum(self.Q_loc_low) + np.sum(self.Q_tx_low)

        info = {
            'service_paused': True,
            'flight_phase': True,
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
            'cpu_energy_model': 'flight_paused',
            'kkt_v': float(self.kkt_v),
            'f_high_per_uav': zeros_n.copy(),
            'f_low_per_uav': zeros_n.copy(),
            'f_total_per_uav': zeros_n.copy(),
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
            'arrival_high_sum': 0.0,
            'arrival_low_sum': 0.0,
            'arrival_total_per_uav': zeros_n.copy(),
            'Q_loc_high_sum': np.sum(self.Q_loc_high),
            'Q_loc_low_sum': np.sum(self.Q_loc_low),
            'Q_tx_high_sum': np.sum(self.Q_tx_high),
            'Q_tx_low_sum': np.sum(self.Q_tx_low),
            'Q_loc_total': np.sum(self.Q_loc_high) + np.sum(self.Q_loc_low),
            'Q_tx_total': np.sum(self.Q_tx_high) + np.sum(self.Q_tx_low),
        }
        info.update(self.get_task_delay_stats())
        return info

    def _advance_flight_only_step(self):
        delta_x_uav = self.current_macro_v * np.cos(self.current_macro_theta) * self.micro_dt
        delta_y_uav = self.current_macro_v * np.sin(self.current_macro_theta) * self.micro_dt
        self.uav_coords[:, 0] += delta_x_uav
        self.uav_coords[:, 1] += delta_y_uav
        self.uav_coords = np.clip(self.uav_coords, 0.0, 1000.0)
        self.macro_fly_steps_remaining -= 1

        self._clear_current_arrivals()
        self.global_micro_step += 1

        if self.macro_fly_steps_remaining <= 0:
            self._generate_traffic()

        self.record_trajectory_snapshot()
        return self.get_raw_state(), self._build_paused_service_info()

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

    def _allocate_cpu_frequencies(self):
        f_high = np.zeros(self.N, dtype=np.float64)
        f_low = np.zeros(self.N, dtype=np.float64)
        energy_model = "total_cubic"

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

    def _get_u2u_spectral_efficiency(self):
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

    def micro_step(self, split_ratios_high, split_ratios_low):
        """
        Discrete arrivals + explicit upload completion events + fluid UAV-side queues.
        """
        if self.macro_fly_steps_remaining > 0:
            return self._advance_flight_only_step()

        arrived_high = np.zeros(self.N)
        arrived_low = np.zeros(self.N)
        arrived_task_ids_by_uav = [[] for _ in range(self.N)]

        if self.pending_upload_jobs:
            active_upload_jobs = []
            for job in self.pending_upload_jobs:
                if job['remaining_steps'] > 0:
                    active_upload_jobs.append(job)
                    continue
                target = job['target_uav']
                arrived_high[target] += job['high_bits']
                arrived_low[target] += job['low_bits']
                arrived_task_ids_by_uav[target].extend(
                    job.get('task_ids', [])
                )
            self.pending_upload_jobs = active_upload_jobs

        if np.sum(self.A_t_high) + np.sum(self.A_t_low) > 0:
            assigned_uavs = self.user_allocations
            user_offsets = self.user_coords - self.uav_coords[assigned_uavs]
            user_dist_sq = np.sum(user_offsets**2, axis=1) + self.H**2
            user_channel_gain = self.beta_0 / np.maximum(user_dist_sq, 1e-8)
            user_bandwidth = self.B_gt_total / max(self.M, 1)
            user_rate_bps = user_bandwidth * np.log2(1.0 + (self.p_gt * user_channel_gain) / self.noise_power)
            user_uplink_step_mbits = (user_rate_bps * self.micro_dt) / 1e6

            for i in range(self.N):
                zone_mask = self.user_allocations == i
                zone_capacity = np.sum(user_uplink_step_mbits[zone_mask])
                total_bits = self.A_t_high[i] + self.A_t_low[i]
                if total_bits <= 1e-12:
                    continue
                task_ids = list(self.current_arrival_tasks_by_uav[i])
                upload_steps = max(1, int(np.ceil(total_bits / max(zone_capacity, 1e-8))))
                self.pending_upload_jobs.append({
                    'target_uav': i,
                    'high_bits': float(self.A_t_high[i]),
                    'low_bits': float(self.A_t_low[i]),
                    'remaining_steps': upload_steps,
                    'active_users': float(self.current_arrival_active_users[i]),
                    'task_ids': task_ids,
                })
                self.current_arrival_tasks_by_uav[i] = []

        assigned_anchors = self.anchor_points[self.user_allocations]
        if self.users_static:
            self.user_velocities.fill(0.0)
            self.user_coords = assigned_anchors + self.user_offsets
            self.user_coords = np.clip(self.user_coords, 0.0, 1000.0)
        else:
            # [Dynamic Group Mobility Update (RPGM)] 群体锚点现在是固定的
            w_t = self.np_random.standard_normal((self.M, 2))
            self.user_velocities = (self.alpha_gmmm * self.user_velocities) + \
                                   (np.sqrt(1 - self.alpha_gmmm**2) * w_t)

            speeds = np.linalg.norm(self.user_velocities, axis=1, keepdims=True)
            self.user_velocities = np.where(
                speeds > self.max_infantry_speed,
                self.user_velocities * (self.max_infantry_speed / np.maximum(speeds, 1e-6)),
                self.user_velocities
            )

            self.user_offsets += self.user_velocities * self.micro_dt
            zone_dist = np.linalg.norm(self.user_offsets, axis=1, keepdims=True)
            safe_zone_dist = np.maximum(zone_dist, 1e-8)
            self.user_offsets = np.where(
                zone_dist > self.user_zone_radius,
                self.user_offsets / safe_zone_dist * self.user_zone_radius,
                self.user_offsets,
            )

            self.user_coords = assigned_anchors + self.user_offsets
            self.user_coords = np.clip(self.user_coords, 0.0, 1000.0)

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

        f_high_array, f_low_array, f_opt_array, cpu_energy_model = (
            self._allocate_cpu_frequencies()
        )
        mu_high_mbits = (f_high_array * self.micro_dt) / (self.C * 1e6)
        mu_low_mbits = (f_low_array * self.micro_dt) / (self.C * 1e6)
        compute_service_priority_mbits = {"high": 0.0, "low": 0.0}
        compute_service_type_mbits = {
            name: 0.0 for name in self.device_type_names
        }

        actual_exe_high = self._serve_compute_task_queues(
            mu_high_mbits,
            "high",
            service_by_priority=compute_service_priority_mbits,
            service_by_type=compute_service_type_mbits,
        )
        actual_exe_low = self._serve_compute_task_queues(
            mu_low_mbits,
            "low",
            service_by_priority=compute_service_priority_mbits,
            service_by_type=compute_service_type_mbits,
        )

        if cpu_energy_model == "separable_cubic":
            energy_comp_per_uav = (
                self.kappa
                * (f_high_array**3 + f_low_array**3)
                * self.micro_dt
            )
        else:
            energy_comp_per_uav = self.kappa * (f_opt_array**3) * self.micro_dt
        total_E_comp = np.sum(energy_comp_per_uav)

        spectral_efficiency = self._get_u2u_spectral_efficiency()

        tx_demand_mask = (self.Q_tx_high + self.Q_tx_low) > 1e-9
        np.fill_diagonal(tx_demand_mask, False)
        num_active_links = np.count_nonzero(tx_demand_mask)

        link_bandwidth = np.zeros((self.N, self.N))
        if num_active_links > 0:
            link_bandwidth[tx_demand_mask] = self.B_u2u_total / num_active_links

        rate_bps = link_bandwidth * spectral_efficiency
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

        upload_delay_cost_per_uav = np.zeros(self.N, dtype=np.float32)
        energy_uplink_per_uav = np.zeros(self.N, dtype=np.float32)
        upload_delay_cost_high = 0.0
        upload_delay_cost_low = 0.0
        for job in self.pending_upload_jobs:
            target = int(job['target_uav'])
            high_delay = job['high_bits'] * job['remaining_steps'] * self.micro_dt
            low_delay = job['low_bits'] * job['remaining_steps'] * self.micro_dt
            upload_delay_cost_high += high_delay
            upload_delay_cost_low += low_delay
            if 0 <= target < self.N:
                energy_uplink_per_uav[target] += (
                    job['active_users'] * self.p_gt * self.micro_dt
                )
                upload_delay_cost_per_uav[target] += (
                    5.0 * high_delay + low_delay
                )
            job['remaining_steps'] -= 1
        total_E_uplink = float(np.sum(energy_uplink_per_uav))
        upload_delay_cost = 5.0 * upload_delay_cost_high + upload_delay_cost_low

        active_u2u_links = ((actual_trans_high > 0) | (actual_trans_low > 0)).astype(float)
        energy_trans_per_uav = np.sum(
            active_u2u_links * self.p_u2u * self.micro_dt,
            axis=1,
        ).astype(np.float32)
        total_E_trans = float(np.sum(energy_trans_per_uav))

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
            'upload_delay_cost': upload_delay_cost,
            'upload_delay_cost_high': upload_delay_cost_high,
            'upload_delay_cost_low': upload_delay_cost_low,
            'upload_delay_cost_per_uav': upload_delay_cost_per_uav.copy(),
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
            'f_high_per_uav': f_high_array.copy(),
            'f_low_per_uav': f_low_array.copy(),
            'f_total_per_uav': f_opt_array.copy(),
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
            'arrival_high_sum': np.sum(self.A_t_high),
            'arrival_low_sum': np.sum(self.A_t_low),
            'arrival_total_per_uav': (self.A_t_high + self.A_t_low).copy(),
            'Q_loc_high_sum': np.sum(self.Q_loc_high),
            'Q_loc_low_sum': np.sum(self.Q_loc_low),
            'Q_tx_high_sum': np.sum(self.Q_tx_high),
            'Q_tx_low_sum': np.sum(self.Q_tx_low),
            'Q_loc_total': np.sum(self.Q_loc_high) + np.sum(self.Q_loc_low),
            'Q_tx_total': np.sum(self.Q_tx_high) + np.sum(self.Q_tx_low)
        }
        for task_type in self.device_type_names:
            info[f'compute_service_{task_type}_mbits'] = (
                compute_service_type_mbits.get(task_type, 0.0)
            )
            info[f'compute_service_{task_type}_cycles'] = (
                compute_service_type_cycles.get(task_type, 0.0)
            )
        info.update(self.get_task_delay_stats())

        self.global_micro_step += 1
        self._generate_traffic()

        self.record_trajectory_snapshot()

        return self.get_raw_state(), info

    def macro_step(self, macro_actions):
        """
        Fly-Hover protocol for one 10s macro slot.
        The action is interpreted as a desired 2D displacement [dx, dy] in
        meters. UAVs fly for 1s and then hover for 9s.
        """
        macro_actions = np.asarray(macro_actions, dtype=np.float64)
        delta_xy = np.clip(macro_actions, -20.0, 20.0)
        t_fly = 1.0
        t_hover = 9.0

        current_coords = self.uav_coords.copy()
        projected_coords = np.clip(current_coords + delta_xy, 0.0, 1000.0)

        # [REMOVED FENCE LOGIC] 废除物理硬限墙，依靠强化学习的软惩罚来约束区域飞行
        # 这允许 UAV 更加自由、连贯地向目标中心移动，避免在边界处发生位置卡壳。
        actual_delta_xy = projected_coords - current_coords
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
        self.macro_fly_steps_remaining = int(round(t_fly / self.micro_dt))
        self._clear_current_arrivals()

        E_fly = self._compute_tcom_flying_power(fly_v) * t_fly
        E_hover = self._compute_tcom_flying_power(np.zeros(self.N)) * t_hover
        E_macro_per_uav = E_fly + E_hover
        max_macro_flying_E = 3000.0
        h_flying_penalty = - (np.sum(E_fly) / max_macro_flying_E) + u2g_penalty + anchor_penalty
        E_macro_total = np.sum(E_macro_per_uav)

        return self.get_raw_state(), h_flying_penalty, {
            'E_flying': np.sum(E_fly),
            'E_hover': np.sum(E_hover),
            'E_macro_total': E_macro_total,
            'E_flying_per_uav': E_fly.copy(),
            'E_hover_per_uav': E_hover.copy(),
            'E_macro_per_uav': E_macro_per_uav.copy(),
            'u2g_penalty': u2g_penalty,
            'anchor_penalty': anchor_penalty,
            'macro_penalty': h_flying_penalty,
            'u2g_distances': centroid_distances.copy(),
            'anchor_distances': anchor_distance_values.copy(),
            'projected_uav_coords': projected_coords.copy()
        }

    def render_trajectory(self, save_path=None, show=False):
        if not self.record_trajectory: return
        import matplotlib.pyplot as plt
        import os
        if save_path: os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
        uav_traj = np.array(self.uav_trajectory)
        cent_traj = np.array(self.centroid_trajectory)
        users_traj = np.array(self.users_trajectory)
        
        plt.figure(figsize=(10, 10))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for i in range(self.N):
            c = colors[i % len(colors)]
            
            # 画每个区域内真实独立用户的细碎游走轨迹
            zone_users_idx = np.where(self.user_allocations == i)[0]
            for u_idx in zone_users_idx:
                u_traj_x = users_traj[:, u_idx, 0]
                u_traj_y = users_traj[:, u_idx, 1]
                # 只给第一个贴图例标签防止重复
                l = f'Ground Users (Zone {i+1})' if u_idx == zone_users_idx[0] else None
                # 加深地名用户的颜色，使得更容易看清
                plt.plot(u_traj_x, u_traj_y, color=c, linestyle=':', linewidth=1.2, alpha=0.6, label=l)
                # 画出最终落点小圆点
                plt.scatter(u_traj_x[-1], u_traj_y[-1], color=c, marker='o', s=12, alpha=0.8)
            
            # 画区域引力质心点游走
            plt.plot(cent_traj[:, i, 0], cent_traj[:, i, 1], color=c, linestyle='--', linewidth=1.0, alpha=0.8, label=f'Smoothed Flow Centroid (Zone {i+1})')
            plt.scatter(cent_traj[0, i, 0], cent_traj[0, i, 1], color=c, marker='X', s=40, edgecolors='black', zorder=5) 
            plt.scatter(cent_traj[-1, i, 0], cent_traj[-1, i, 1], color=c, marker='P', s=60, edgecolors='black', zorder=5)
            
            # 飞行动作 (进一步调细线条和起落标志)
            plt.plot(uav_traj[:, i, 0], uav_traj[:, i, 1], color=c, linestyle='-', linewidth=1.0, marker='o', markersize=2, alpha=0.9, label=f'UAV-{i+1} Traj')
            plt.scatter(uav_traj[0, i, 0], uav_traj[0, i, 1], color=c, marker='s', s=20, edgecolors='black', zorder=5) 
            plt.scatter(uav_traj[-1, i, 0], uav_traj[-1, i, 1], color=c, marker='*', s=80, edgecolors='black', zorder=5) 
            
        plt.scatter([], [], color='gray', marker='s', edgecolors='black', s=20, label='UAV Start')
        plt.scatter([], [], color='gray', marker='*', edgecolors='black', s=80, label='UAV End')
        plt.scatter([], [], color='gray', marker='X', edgecolors='black', s=40, label='Centroid Init')
        plt.scatter([], [], color='gray', marker='P', edgecolors='black', s=60, label='Centroid Final')
        
        plt.title('UAV-MEC System: Trajectories & Dynamic Duty Centroids Map', fontsize=15, fontweight='bold')
        plt.xlabel('X Coordinate (m)', fontsize=12)
        plt.ylabel('Y Coordinate (m)', fontsize=12)
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show: plt.show()
        plt.close()
