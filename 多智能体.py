# -*- coding: utf-8 -*-
"""
支持 QoS (服务质量) 保障的多智能体低轨卫星跳波束资源调度系统 (MoE-MARL)

QoS 核心设计：
1. 业务分级：实时业务 (RT) 高优先级/时延敏感，非实时业务 (NRT) 低优先级/吞吐量敏感。
2. 动态 QoS 权重分配：功率与波束分配优先保障高时延、高丢包风险的波位。
3. QoS 惩罚机制：将 QoS 违约（时延超时丢包）直接作为 Negative Reward 反馈给强化学习。
4. 多维度 QoS 观测：包含 RT/NRT 队列时延、丢包率与业务满意度。
"""

import random
from collections import deque
import numpy as np
from scipy.special import jv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================================
# 全局物理与 QoS 参数定义
# ============================================================================
T_noise = 300  # 系统噪声温度，单位 K (开尔文)
Bo = 1.38e-23  # 玻尔兹曼常数，单位 J/K


# ============================================================================
# 第一部分：考虑 QoS 保障的多智能体卫星环境
# ============================================================================
class MultiAgentLEOSatEnvQoS:
    """
    支持 QoS 保障的多智能体 LEO 跳波束环境
    包含 12 个波位 Agent，单时隙最多允许同时激活 K=4 个波束
    """
    def __init__(self, objective='qos_optimized'):
        # 1. 智能体与空间结构配置
        self.N = 12  # 地面波位 Agent 的总数量
        self.K = 4   # 每时隙允许激活的最大波束数量上限
        self.agents = [f"agent_{i}" for i in range(self.N)]  # 实例化 Agent 名称列表
        self.agent_name_to_idx = {name: i for i, name in enumerate(self.agents)}  # Agent 名称到索引映射

        # 2. 卫星通信物理层参数
        self.h = 570e3            # 卫星轨道高度，570 km (米)
        self.fc = 20e9            # 载波频率，20 GHz
        self.bandwidth = 200e6    # 单波束信道带宽，200 MHz
        self.total_power = 120    # 卫星总发射功率，120 W
        self.max_beam_power = 60  # 单波束最大发射功率上限，60 W
        self.G_t = 40             # 发射天线增益，40 dB
        self.G_r = 50             # 接收天线增益，50 dB
        self.slot_duration = 0.01 # 跳波束单时隙持续时间，10 ms
        self.packet_size = 10 * 1024 * 8  # 单个数据包的大小，10 kbit
        self.lambda_wave = 3e8 / self.fc   # 电磁波波长 (米)

        # 3. QoS 核心约束与参数配置
        self.delay_threshold_rt = 0.1   # RT 实时业务最大容忍排队时延：100 ms (强 QoS 约束)
        self.delay_threshold_nrt = 1.0  # NRT 非实时业务最大容忍排队时延：1000 ms
        self.qos_rt_weight = 3.0        # RT 业务 QoS 优先级权重因子
        self.qos_nrt_weight = 1.0       # NRT 业务 QoS 优先级权重因子

        # 4. 几何坐标与物理层干扰预计算
        self.spot_positions = self._generate_spot_positions()        # 计算 12 波位蜂窝平面坐标
        self.interference_matrix = self._precompute_interference()   # 预计算 12x12 物理层干涉矩阵

        # 5. 业务队列与 QoS 统计指标初始化
        self.rt_queue = [deque() for _ in range(self.N)]    # 保存 RT 数据包入队时隙编号的双端队列
        self.nrt_queue = [deque() for _ in range(self.N)]   # 保存 NRT 数据包入队时隙编号的双端队列
        
        self.dropped_rt_packets = np.zeros(self.N)   # 各波位累计超时丢弃的 RT 包数
        self.dropped_nrt_packets = np.zeros(self.N)  # 各波位累计超时丢弃的 NRT 包数
        self.served_rt_packets = np.zeros(self.N)     # 各波位累计成功传输服务的 RT 包数
        self.served_nrt_packets = np.zeros(self.N)    # 各波位累计成功传输服务的 NRT 包数
        self.total_rt_arrived = np.zeros(self.N)      # 各波位累计到达的 RT 包数
        self.total_nrt_arrived = np.zeros(self.N)     # 各波位累计到达的 NRT 包数

        # 6. 空间与时间维度业务流量分布建模
        self.base_demand = np.array([800, 700, 1300, 300, 980, 250,
                                     1000, 275, 80, 600, 50, 200])  # 基础流量需求
        self.spatial_factor = self.base_demand / np.mean(self.base_demand)  # 归一化空间需求因子
        self.FIG7_TIME_PROFILE = np.array([
            0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
            0.15, 0.26, 0.42, 0.60, 1.00, 0.90,
            0.85, 0.78, 0.66, 0.78, 0.82, 0.68,
            0.42, 0.32, 0.18, 0.10, 0.06, 0.03])  # 24小时日流量起伏曲线
        self.total_slots_per_day = 8_640_000     # 全天总跳波束时隙数
        self.base_packet_rate = 50             # 基础数据包生成速率

        self.current_slot = 0    # 当前跳波束时隙计数器
        self.objective = objective

    def _generate_spot_positions(self):
        """生成 12 个波位在地面投影的蜂窝结构二维坐标 (单位: 公里)"""
        R = 73                     # 蜂窝半径
        d = np.sqrt(3) * R          # 中心到邻居的距离
        positions = [(0, 0)]       # 中心波位 0

        # 生成第一层内圈 6 个波位
        for k in range(6):
            angle = np.deg2rad(60 * k + 30)
            positions.append((d * np.cos(angle), d * np.sin(angle)))

        # 生成第二层外圈 5 个波位
        outer_offsets = [
            (d * np.cos(np.deg2rad(30)) + d * np.cos(np.deg2rad(90)),
             d * np.sin(np.deg2rad(30)) + d * np.sin(np.deg2rad(90))),
            (d * np.cos(np.deg2rad(90)) + d * np.cos(np.deg2rad(150)),
             d * np.sin(np.deg2rad(90)) + d * np.sin(np.deg2rad(150))),
            (2 * d * np.cos(np.deg2rad(150)), 2 * d * np.sin(np.deg2rad(150))),
            (d * np.cos(np.deg2rad(150)) + d * np.cos(np.deg2rad(210)),
             d * np.sin(np.deg2rad(150)) + d * np.sin(np.deg2rad(210))),
            (d * np.cos(np.deg2rad(210)) + d * np.cos(np.deg2rad(270)),
             d * np.sin(np.deg2rad(210)) + d * np.sin(np.deg2rad(270)))
        ]
        positions.extend(outer_offsets)
        return np.array(positions)

    def _precompute_interference(self):
        """预计算 12x12 波位间的物理层同频干扰增益矩阵 (基于贝塞尔函数天线方向图)"""
        N = self.N
        interference = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue  # 不考虑自干涉
                xi, yi = self.spot_positions[i]
                xj, yj = self.spot_positions[j]
                
                # 计算两波位间水平距离与斜距
                d_horiz = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) * 1000
                d_i = np.sqrt(xi ** 2 + yi ** 2 + (self.h / 1000) ** 2) * 1000
                d_j = np.sqrt(xj ** 2 + yj ** 2 + (self.h / 1000) ** 2) * 1000

                # 通过余弦定理计算卫星视角下波束间的离轴夹角
                cos_theta = (d_i ** 2 + d_j ** 2 - d_horiz ** 2) / (2 * d_i * d_j)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta_mn = np.arccos(cos_theta)

                # 依据贝塞尔函数计算天线副瓣增益
                sin_theta_3db = 0.12703
                u_mn = 2.07123 * np.sin(theta_mn) / sin_theta_3db
                if u_mn == 0:
                    G_theta = 1.0
                else:
                    J1, J3 = jv(1, u_mn), jv(3, u_mn)
                    G_theta = (10 ** (self.G_t / 10)) * ((J1 / (2 * u_mn) + 36 * J3 / (u_mn ** 3)) ** 2)

                # 自由空间路径损耗
                path_loss = (self.lambda_wave / (4 * np.pi * d_horiz)) ** 2
                interference[i][j] = G_theta * path_loss
        return interference

    def _get_traffic_rates(self, current_slot):
        """结合时间与空间因子，计算当前时隙各波位 RT 与 NRT 业务的泊松到达率"""
        hour_idx = int((current_slot / self.total_slots_per_day) * 24) % 24
        time_factor = self.FIG7_TIME_PROFILE[hour_idx]
        total_expected_rate = self.spatial_factor * time_factor * self.base_packet_rate
        # 设定业务配比：实时业务 (RT) 占比 40%，非实时业务 (NRT) 占比 60%
        return total_expected_rate * 0.4, total_expected_rate * 0.6

    def reset(self):
        """重置环境状态、队列数据与 QoS 统计指标"""
        self.rt_queue = [deque() for _ in range(self.N)]
        self.nrt_queue = [deque() for _ in range(self.N)]
        self.dropped_rt_packets = np.zeros(self.N)
        self.dropped_nrt_packets = np.zeros(self.N)
        self.served_rt_packets = np.zeros(self.N)
        self.served_nrt_packets = np.zeros(self.N)
        self.total_rt_arrived = np.zeros(self.N)
        self.total_nrt_arrived = np.zeros(self.N)
        self.current_slot = 0
        return self._get_obs(), self._get_state()

    def _calc_avg_delay(self, queue):
        """计算指定队列中数据包的平均排队等待时延 (单位: 秒)"""
        if len(queue) == 0:
            return 0.0
        waiting_slots = self.current_slot - np.array(queue)
        return np.mean(waiting_slots) * self.slot_duration

    def _get_obs(self):
        """
        获取各个 Agent 的局部观测向量 (Local Observation, 6 维)
        特征定义: [RT队列长度, NRT队列长度, RT平均排队时延, NRT平均排队时延, RT丢包率, 物理干扰强度]
        """
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            rt_delay = self._calc_avg_delay(self.rt_queue[i])
            nrt_delay = self._calc_avg_delay(self.nrt_queue[i])
            
            # 计算当前 RT 业务的丢包率 (作为衡量 QoS 破坏程度的核心指标)
            rt_drop_rate = self.dropped_rt_packets[i] / (self.total_rt_arrived[i] + 1e-8)
            interf = np.mean(self.interference_matrix[i])

            obs_dict[agent] = np.array([
                len(self.rt_queue[i]),    # 特征 1: RT 实时队列包数
                len(self.nrt_queue[i]),   # 特征 2: NRT 非实时队列包数
                rt_delay,                 # 特征 3: RT 实时排队平均时延
                nrt_delay,                # 特征 4: NRT 非实时排队平均时延
                rt_drop_rate,             # 特征 5: RT 丢包率 (QoS 指标)
                interf                    # 特征 6: 邻居同频信道干扰强度
            ], dtype=np.float32)
        return obs_dict

    def _get_state(self):
        """获取全局状态向量 (Global State, 36 维，供 Centralized Critic 网络评估)"""
        rt_lens = np.array([len(q) for q in self.rt_queue], dtype=np.float32)
        nrt_lens = np.array([len(q) for q in self.nrt_queue], dtype=np.float32)
        rt_delays = np.array([self._calc_avg_delay(q) for q in self.rt_queue], dtype=np.float32)
        return np.concatenate([rt_lens, nrt_lens, rt_delays])

    def step(self, action_dict):
        """
        执行跳波束时隙步进逻辑
        包括：波束仲裁、QoS 加权功率分配、SINR 与容量计算、队列服务及 QoS 惩罚计算
        """
        # Step 1: 融合 RL 动作得分与 QoS 紧急度偏置，排序选取 Top-K (K=4) 个激活波束
        requested_agents = []
        for agent, act in action_dict.items():
            i = self.agent_name_to_idx[agent]
            # 计算 QoS 紧急程度因子 = (RT时延 / RT时延阈值)^2
            rt_delay = self._calc_avg_delay(self.rt_queue[i])
            qos_urgency = (rt_delay / self.delay_threshold_rt) ** 2
            
            # 将智能体网络输出得分与 QoS 紧急度进行加权融合
            combined_score = act + 0.5 * qos_urgency
            requested_agents.append((i, combined_score))

        # 降序排列选取得分最高的前 K 个波位激活波束
        requested_agents.sort(key=lambda x: x[1], reverse=True)
        action_list = [idx for idx, _ in requested_agents[:self.K]]

        # Step 2: 基于 QoS 需求的非对称功率分配 (高 QoS 紧急度的波位分配更大功率)
        weights = {}
        for i in action_list:
            rt_delay = self._calc_avg_delay(self.rt_queue[i])
            nrt_delay = self._calc_avg_delay(self.nrt_queue[i])
            # 根据 RT/NRT 队列长度及超时比率计算非对称功率权重
            w_i = (len(self.rt_queue[i]) * self.qos_rt_weight * (1 + rt_delay / self.delay_threshold_rt) +
                   len(self.nrt_queue[i]) * self.qos_nrt_weight * (1 + nrt_delay / self.delay_threshold_nrt) + 1e-5)
            weights[i] = w_i

        total_w = sum(weights.values())
        # 按比例归一化分配卫星总功率，且不超过单波束最大功率限制
        allocated_power = {i: min((weights[i] / total_w) * self.total_power, self.max_beam_power) for i in action_list}

        # Step 3: 计算干涉、SINR 与香农信道容量 (包数)
        capacity_packets = {}
        for i in action_list:
            # 汇总来自其他同时激活波束的同频干扰
            interf_sum = sum(self.interference_matrix[i][j] * allocated_power[j] for j in action_list if i != j)
            noise_power = Bo * T_noise * self.bandwidth

            # 计算链路路径损耗与到达接收功率
            xi_m, yi_m = self.spot_positions[i] * 1000.0
            d_i_m = np.sqrt(xi_m ** 2 + yi_m ** 2 + self.h ** 2)
            path_loss = (self.lambda_wave / (4 * np.pi * d_i_m)) ** 2

            signal_power = allocated_power[i] * (10 ** (self.G_t / 10)) * (10 ** (self.G_r / 10)) * path_loss
            sinr = signal_power / (interf_sum + noise_power)
            
            # 换算当前时隙最大可传输的数据包数量
            cap_bps = self.bandwidth * np.log2(1 + sinr)
            capacity_packets[i] = int(np.floor((cap_bps * self.slot_duration) / self.packet_size))

        # Step 4: 产生当前时隙的新数据包到达并入队
        lambda_rt, lambda_nrt = self._get_traffic_rates(self.current_slot)
        for i in range(self.N):
            arr_rt = np.random.poisson(lambda_rt[i])
            arr_nrt = np.random.poisson(lambda_nrt[i])
            self.rt_queue[i].extend([self.current_slot] * arr_rt)
            self.nrt_queue[i].extend([self.current_slot] * arr_nrt)
            self.total_rt_arrived[i] += arr_rt
            self.total_nrt_arrived[i] += arr_nrt

        # Step 5: 严格按 QoS 优先级执行出队服务（优先清空 RT 队列，剩余容量服务 NRT 队列）
        served_rt_step = np.zeros(self.N)
        served_nrt_step = np.zeros(self.N)
        
        for i in action_list:
            cap = capacity_packets[i]
            # 优先服务 RT 实时数据包
            s_rt = min(len(self.rt_queue[i]), cap)
            for _ in range(s_rt):
                self.rt_queue[i].popleft()
            cap -= s_rt
            served_rt_step[i] = s_rt
            self.served_rt_packets[i] += s_rt

            # 利用余量传输能力服务 NRT 非实时数据包
            s_nrt = min(len(self.nrt_queue[i]), cap)
            for _ in range(s_nrt):
                self.nrt_queue[i].popleft()
            served_nrt_step[i] = s_nrt
            self.served_nrt_packets[i] += s_nrt

        # Step 6: 检查 QoS 违约：超时未传输的数据包将被强行丢弃，并统计违约包数
        max_rt_slots = int(self.delay_threshold_rt / self.slot_duration)
        max_nrt_slots = int(self.delay_threshold_nrt / self.slot_duration)
        
        step_dropped_rt = np.zeros(self.N)
        step_dropped_nrt = np.zeros(self.N)

        for i in range(self.N):
            # 遍历检查 RT 数据包是否超过最大容忍时延
            while len(self.rt_queue[i]) > 0 and (self.current_slot - self.rt_queue[i][0]) > max_rt_slots:
                self.rt_queue[i].popleft()
                step_dropped_rt[i] += 1
                self.dropped_rt_packets[i] += 1

            # 遍历检查 NRT 数据包是否超时
            while len(self.nrt_queue[i]) > 0 and (self.current_slot - self.nrt_queue[i][0]) > max_nrt_slots:
                self.nrt_queue[i].popleft()
                step_dropped_nrt[i] += 1
                self.dropped_nrt_packets[i] += 1

        # Step 7: 构建包含 QoS 传输收益与 QoS 违约惩罚的综合奖励函数
        rewards = {}
        for i, agent in enumerate(self.agents):
            # 正向收益：经过 QoS 权重加权后的实际有效传输包数
            r_throughput = self.qos_rt_weight * served_rt_step[i] + self.qos_nrt_weight * served_nrt_step[i]
            
            # 负向惩罚 (QoS Penalty)：针对 RT 时延违约与丢包施加高额惩罚
            rt_delay = self._calc_avg_delay(self.rt_queue[i])
            penalty_delay = 2.0 * max(0.0, rt_delay - self.delay_threshold_rt)   # 超时时延惩罚
            penalty_drop = 5.0 * step_dropped_rt[i] + 1.0 * step_dropped_nrt[i]  # 强制丢包惩罚
            
            # 计算该 Agent 本时隙的最终 QoS 综合奖励得分
            rewards[agent] = float(r_throughput - penalty_delay - penalty_drop)

        self.current_slot += 1
        dones = {agent: False for agent in self.agents}
        dones["__all__"] = False

        return self._get_obs(), rewards, dones, {
            "active_actions": action_list,
            "dropped_rt": np.sum(step_dropped_rt),
            "served_rt": np.sum(served_rt_step)
        }


# ============================================================================
# 第二部分：支持 QoS 决策的 MoE (Mixture of Experts) 神经网络
# ============================================================================
class SingleExpertQoS(nn.Module):
    """单专家神经网络：用于拟合特定 QoS 状态模式下的决策映射"""
    def __init__(self, obs_dim, hidden_dim=64):
        super(SingleExpertQoS, self).__init__()
        # 定义三层全连接前馈神经网络
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),   # 输入特征映射到隐藏层
            nn.ReLU(),                         # 激活函数
            nn.Linear(hidden_dim, hidden_dim), # 隐藏层特征提取
            nn.ReLU(),                         # 激活函数
            nn.Linear(hidden_dim, 1)           # 输出未归一化的波束激活偏好得分
        )

    def forward(self, obs):
        return self.fc(obs)


class MoEActorQoS(nn.Module):
    """
    QoS 驱动的 MoE Policy Network (Actor)
    引入多专家与门控网络，实现根据 QoS 状态动态选择专家的稀疏路由机制
    """
    def __init__(self, obs_dim=6, num_experts=3, k_top=2):
        super(MoEActorQoS, self).__init__()
        self.num_experts = num_experts  # 专家网络数量
        self.k_top = k_top              # 稀疏路由保留的 Top-K 专家数

        # 1. 实例化多个独立权重的专家网络
        self.experts = nn.ModuleList([SingleExpertQoS(obs_dim) for _ in range(num_experts)])

        # 2. 门控路由网络：根据 6 维 QoS 观测向量评估各专家的权重
        self.gating = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts)
        )

    def forward(self, obs):
        # 若输入为单条向量，增加 Batch 维度
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Step A: 计算门控路由网络输出并转为概率分布
        gate_logits = self.gating(obs)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Step B: 执行 Top-K 稀疏门控截断与权重重新归一化
        if self.k_top < self.num_experts:
            top_k_weights, top_k_indices = torch.topk(gate_weights, self.k_top, dim=-1)
            # 对 Top-K 概率重新归一化，使其和为 1
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
            zeros = torch.zeros_like(gate_weights)
            gate_weights = zeros.scatter(-1, top_k_indices, top_k_weights)

        # Step C: 并行计算各个专家的预测得分 [batch_size, 1, num_experts]
        expert_outputs = torch.stack([expert(obs) for expert in self.experts], dim=-1)
        
        # Step D: 将门控权重加权融合到各专家输出，并通过 Sigmoid 映射到 [0, 1]
        final_output = torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=-1)
        return torch.sigmoid(final_output).squeeze(-1), gate_weights


class MoECriticQoS(nn.Module):
    """
    Centralized MoE Critic 网络 (适用于 CTDE 架构)
    接收全局 QoS 状态向量，使用多专家门控预测全局状态价值 V(s)
    """
    def __init__(self, state_dim=36, num_experts=3):
        super(MoECriticQoS, self).__init__()
        # 实例化 Critic 的多专家结构
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_experts)
        ])
        # Critic 侧的全局门控路由网络
        self.gating = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        # 扩充 Batch 维度
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # 计算 Critic 门控权重与各专家估值，融合得到全局 V(s)
        gate_weights = self.gating(state)
        expert_vals = torch.stack([expert(state) for expert in self.experts], dim=-1)
        final_val = torch.sum(expert_vals * gate_weights.unsqueeze(1), dim=-1)
        
        return final_val.squeeze(-1)


# ============================================================================
# 第三部分：辅助损失计算与训练主程序
# ============================================================================
def compute_moe_aux_loss(gate_weights):
    """
    计算负载均衡辅助损失 (Load Balancing Auxiliary Loss)
    防止门控网络退化塌陷到单条路由，强迫训练过程均匀激活各个 QoS 专家
    """
    # 计算当前 Batch/Agents 在所有专家上的平均权重激活密度
    density = torch.mean(gate_weights, dim=0)
    # 计算方差偏离惩罚项
    loss = torch.sum(density * density) * gate_weights.size(-1)
    return loss


def train_qos_moe():
    """QoS-MoE 跳波束调度策略主训练流程"""
    # 1. 初始化 QoS 跳波束环境与模型维度参数
    env = MultiAgentLEOSatEnvQoS(objective='qos_optimized')
    obs_dim = 6      # 单 Agent 局部观测维度: [RT队列, NRT队列, RT时延, NRT时延, RT丢包率, 物理干扰]
    state_dim = 36   # 全局状态维度: 12*RT队列 + 12*NRT队列 + 12*RT时延

    num_experts = 3  # 构建 3 个不同分工的 QoS 专家
    k_top = 2        # 稀疏门控选择前 2 个专家

    # 2. 实例化网络与优化器
    actor_moe = MoEActorQoS(obs_dim=obs_dim, num_experts=num_experts, k_top=k_top)
    critic_moe = MoECriticQoS(state_dim=state_dim, num_experts=num_experts)

    opt_actor = optim.Adam(actor_moe.parameters(), lr=1e-3)
    opt_critic = optim.Adam(critic_moe.parameters(), lr=1e-3)

    episodes = 10           # 训练 Episode 轮数
    steps_per_episode = 100 # 每轮演进的时隙步数

    print("=== 开始支持 QoS (服务质量) 保障的 MoE-MARL 跳波束调度训练 ===")

    # 3. 跨 Episode 主训练循环
    for ep in range(episodes):
        obs_dict, global_state = env.reset()
        total_team_reward = 0       # 累加团队总奖励
        total_rt_dropped = 0        # 累加 RT 丢包数
        total_rt_served = 0         # 累加 RT 成功传输数
        gate_weight_history = []     # 记录门控权重选择历史

        for step in range(steps_per_episode):
            action_dict = {}
            step_gate_weights = []

            # Step 3.1: 12 个 Agent 依次前向传播，通过 MoE 计算动作得分与路由权重
            for agent_name, obs in obs_dict.items():
                obs_t = torch.FloatTensor(obs)
                with torch.no_grad():
                    act_score, g_weights = actor_moe(obs_t)
                action_dict[agent_name] = act_score.item()
                step_gate_weights.append(g_weights.squeeze(0))

            # Step 3.2: 联合动作提交环境，执行物理演进并获取下一步观测与 QoS 反馈
            next_obs_dict, rewards, dones, info = env.step(action_dict)
            next_global_state = env._get_state()

            # Step 3.3: 统计 QoS 性能指标
            team_reward = sum(rewards.values())
            total_team_reward += team_reward
            total_rt_dropped += info["dropped_rt"]
            total_rt_served += info["served_rt"]

            # Step 3.4: 求解损失并更新网络参数
            state_t = torch.FloatTensor(global_state)
            val_pred = critic_moe(state_t)  # Critic 价值估计
            
            critic_loss = (val_pred - team_reward) ** 2  # 状态价值均方误差损失

            # 组合负载均衡辅助损失，避免专家崩塌
            all_gates_t = torch.stack(step_gate_weights)
            aux_loss = compute_moe_aux_loss(all_gates_t)

            # 最终优化总损失
            total_loss = critic_loss + 0.01 * aux_loss

            # 梯度的清除、反向传播与参数更新
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            total_loss.backward()
            opt_critic.step()
            opt_actor.step()

            # 演进状态与记录门控历史
            obs_dict = next_obs_dict
            global_state = next_global_state
            gate_weight_history.append(all_gates_t.mean(dim=0).detach().numpy())

        # Step 3.5: 打印各 Episode 的 QoS 保障结果与专家网络路由利用率
        total_rt_packets = total_rt_served + total_rt_dropped
        rt_drop_rate = (total_rt_dropped / (total_rt_packets + 1e-8)) * 100
        avg_expert_usage = np.mean(gate_weight_history, axis=0)

        print(f"Episode {ep + 1:2d}/{episodes} | 全局QoS总奖励: {total_team_reward:8.2f} | "
              f"RT业务丢包率: {rt_drop_rate:5.2f}% | "
              f"专家占比: 专家1={avg_expert_usage[0]*100:4.1f}%, 专家2={avg_expert_usage[1]*100:4.1f}%, 专家3={avg_expert_usage[2]*100:4.1f}%")

    print("=== 支持 QoS 的 MoE 多智能体模型训练完成 ===")


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    train_qos_moe()