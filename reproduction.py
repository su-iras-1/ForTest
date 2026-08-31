import numpy as np  # 导入 NumPy 科学计算库，用于矩阵计算与数值运算
from scipy.special import jv  # 从 SciPy 库中导入一阶和多阶贝塞尔函数 jv，用于计算天线方向图增益
import random  # 导入 random 随机数生成模块，用于动作采样与随机决策
from collections import deque  # 从 collections 模块导入双端队列 deque，用于高效跟踪数据包入队与出队的时间戳

T_noise = 300  # 定义系统噪声温度为 300 开尔文 (K)
Bo = 1.38e-23  # 定义玻尔兹曼常数，单位为 J/K

class LEOSatEnv:  # 定义低轨卫星跳波束环境类 LEOSatEnv
    """
    模拟卫星环境
    单颗低轨卫星跳波束环境
    特点：
    1. 12个固定地面波位 (图5)
    2. 4个可同时激活的波束 (表1)
    3. 包含同频干扰的物理层计算 (公式2~7)
    4. 实时/非实时双队列模型 (公式9~11)
    5. 支持单目标切换 (throughput / delay / satisfaction)
    """  # 类文档字符串说明环境特点
    
    def __init__(self, objective='throughput'):  # 定义环境类的初始化构造函数，默认优化目标为吞吐量 'throughput'
        """
        生成类
        self为关键字 指代这个类自己
        初始化环境（对应论文 表1 参数）
        参数:
            objective (str): 单专家目标，可选 'throughput' | 'delay' | 'satisfaction'
        """  # 初始化函数文档说明
        
        # ---------- 1. 空间与物理参数 (表1) ----------  # 属性配置段落标识：空间与物理参数
        self.t_num = 36  # 设置星座中的轨道总数为 36 条
        self.signle_num = 20  # 设置单条轨道上的卫星数量为 20 颗
        self.h = 570e3  # 设置卫星运行轨道高度为 570 km（转换为米，即 570,000 米）
        self.A = 70  # 设置卫星轨道倾角为 70 度
        self.Total_num = 720  # 设置整个星座的卫星总数为 720 颗
        self.N = 12  # 设置地面服务区域的总波位数（点阵数）为 12 个
        self.K = 4  # 设置单颗卫星可同时激活/点亮的最大波束数量为 4 个
        self.fc = 20e9  # 设置下行射频载波频率为 20 GHz（即 20 * 10^9 Hz）
        self.bandwidth = 200e6  # 设置单个波束的总信道带宽为 200 MHz（即 200 * 10^6 Hz）
        self.total_power = 120  # 设置卫星载荷可提供的发射总功率上限为 120 瓦 (W)
        self.max_beam_power = 60  # 设置单个射频波束可分配的最大发射功率上限为 60 瓦 (W)
        self.G_t = 40  # 设置卫星发射天线峰值增益为 40 dB
        self.G_r = 50  # 设置用户地面终端接收天线峰值增益为 50 dB
        self.slot_duration = 0.01  # 设置跳波束单时隙的持续时间为 0.01 秒（即 10 毫秒）
        self.delay_threshold = 0.4  # 设置实时数据包可容忍的最大排队时延阈值为 0.4 秒（400 毫秒）
        self.packet_size = 10 * 1024 * 8  # 设置单个业务数据包的大小为 10 KB（转换为 bit，即 81,920 比特）
        self.lambda_wave = 3e8 / self.fc  # 根据光速 3e8 m/s 除以载频计算电磁波波长 (米)
        
        # ---------- 2. 波位几何布局 ----------  # 属性配置段落标识：波位几何布局
        self.spot_positions = self._generate_spot_positions()  # 调用内部私有方法生成 12 个地面前蜂窝波位的二维平面坐标矩阵
        
        # ---------- 3. 预计算干扰矩阵 ----------  # 属性配置段落标识：同频干扰预计算
        self.interference_matrix = self._precompute_interference()  # 调用内部私有方法预先计算所有波位间互相产生的同频干扰因子矩阵
        
        # ---------- 4. 队列与统计变量 ----------  # 属性配置段落标识：双队列与业务统计
        self.rt_queue_timestamps = [deque() for _ in range(self.N)]  # 初始化 12 个波位的实时业务(ψ1)时间戳队列列表
        self.nrt_queue_timestamps = [deque() for _ in range(self.N)]  # 初始化 12 个波位的非实时业务(ψ2)时间戳队列列表
        
        self.base_demand = np.array([800, 700, 1300, 300, 980, 250,
                                     1000, 275, 80, 600, 50, 200])  # 设置 12 个波位的基准业务需求数据数组
        
        mean_demand = np.mean(self.base_demand)  # 计算 12 个波位基准需求的平均值
        std_demand = np.std(self.base_demand)  # 计算 12 个波位基准需求的标准差
        self.zeta = std_demand / mean_demand  # 计算空间业务需求的离散系数 (变异系数 CV)
        
        self.spatial_factor = self.base_demand / mean_demand  # 计算归一化的波位空间业务不均匀分布因子向量（均值为 1）
        
        self.FIG7_TIME_PROFILE = np.array([
            0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
            0.15, 0.26, 0.42, 0.60, 1.00, 0.90,
            0.85, 0.78, 0.66, 0.78, 0.82, 0.68,
            0.42, 0.32, 0.18, 0.10, 0.06, 0.03   ])  # 建立一整天 24 小时的归一化时间业务量波动曲线数组
            
        self.total_slots_per_day = 8_640_000  # 计算一整天（24小时）包含的总时隙数量 (24*3600/0.01 = 8,640,000)
        self.base_packet_rate = 50  # 设置每个 10ms 时隙内系统的基准数据包到达率（平均包数）
        
        self.cumulative_served = np.zeros(self.N)  # 初始化各波位累计已成功服务的包数计数器数组，初值全为 0
        self.cumulative_demanded = np.zeros(self.N)  # 初始化各波位累计总到达的业务需求包数计数器数组，初值全为 0
        
        self.current_slot = 0  # 初始化环境当前的运行时隙序号为 0
        self.lambda_realtime = None  # 初始化实时业务到达率向量变量为 None
        self.lambda_nrt = None  # 初始化非实时业务到达率向量变量为 None
        
        self.objective = objective  # 保存传入的优化目标字符串标识
        
        print(f"[Env] 初始化完成 | 目标: {objective} | 波位数: {self.N} | 波束数: {self.K}")  # 打印输出环境初始化完成的日志提示

    # ========================================================================
    # 1. 波位布局生成
    # ========================================================================
    def _generate_spot_positions(self):  # 定义私有方法：生成 12 个固定地面波位的地理坐标
        R = 73  # 设置单个蜂窝波位的覆盖半径为 73 km
        d = np.sqrt(3) * R  # 根据正六边形几何关系计算相邻波位中心的间距
        positions = [(0, 0)]  # 初始化坐标列表，并将序号 1 的中心波位设为原点 (0, 0)
        
        for k in range(6):  # 循环生成内层（第一环）的 6 个环形波位坐标
            angle = np.deg2rad(60 * k + 30)  # 将极坐标角度转换为弧度（从 30° 开始，每隔 60° 分布一个）
            x = d * np.cos(angle)  # 根据极坐标转换公式计算波位的 X 轴坐标
            y = d * np.sin(angle)  # 根据极坐标转换公式计算波位的 Y 轴坐标
            positions.append((x, y))  # 将生成的内层波位坐标追加到列表
            
        outer_offsets = [  # 手动定义外层（第二环）外围 5 个波位的几何矢量偏移坐标（波位 8 ~ 12）
            (d * np.cos(np.deg2rad(30)) + d * np.cos(np.deg2rad(90)),
             d * np.sin(np.deg2rad(30)) + d * np.sin(np.deg2rad(90))),  # 波位 8 的坐标元组
            (d * np.cos(np.deg2rad(90)) + d * np.cos(np.deg2rad(150)),
             d * np.sin(np.deg2rad(90)) + d * np.sin(np.deg2rad(150))),  # 波位 9 的坐标元组
            (2 * d * np.cos(np.deg2rad(150)),
             2 * d * np.sin(np.deg2rad(150))),  # 波位 10 的坐标元组
            (d * np.cos(np.deg2rad(150)) + d * np.cos(np.deg2rad(210)),
             d * np.sin(np.deg2rad(150)) + d * np.sin(np.deg2rad(210))),  # 波位 11 的坐标元组
            (d * np.cos(np.deg2rad(210)) + d * np.cos(np.deg2rad(270)),
             d * np.sin(np.deg2rad(210)) + d * np.sin(np.deg2rad(270)))  # 波位 12 的坐标元组
        ]  # 外层波位相对坐标定义结束
        
        for x, y in outer_offsets:  # 遍历外层波位坐标列表
            positions.append((x, y))  # 将外层波位坐标追加到位置列表
            
        return np.array(positions)  # 将最终包含 12 个波位坐标的列表转换为 NumPy 数组并返回

    # ========================================================================
    # 2. 同频干扰矩阵预计算
    # ========================================================================
    def _precompute_interference(self):  # 定义私有方法：预计算同频干扰功率因子矩阵
        """
        提前算好 12x12 的干扰功率矩阵 (W)
        避免在 step() 中重复计算贝塞尔函数，大幅提升训练速度
        用以计算 self对象 由在之前的position 得出其他波位对自己的I_mn/P_m
        """  # 方法文档说明
        N = self.N  # 提取总波位数 N (12)
        interference = np.zeros((N, N))  # 创建一个 12x12 大小全为 0 的矩阵，用于存储相对干扰系数
        
        for i in range(N):  # 外层循环：遍历目标受干扰波位 i
            for j in range(N):  # 内层循环：遍历产生干扰的发射波位 j
                if i == j:  # 如果目标波位和发射波位是同一个
                    continue  # 跳过自身对自身的干扰计算
                    
                xi, yi = self.spot_positions[i]  # 获取波位 i 的二维坐标 (km)
                xj, yj = self.spot_positions[j]  # 获取波位 j 的二维坐标 (km)
                
                d_horizontal_ij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)  # 计算波位 i 与波位 j 之间的地面水平距离 (km)
                d_i = np.sqrt(xi ** 2 + yi ** 2 + (self.h / 1000) ** 2)  # 计算卫星到波位 i 的斜距 (km)
                d_j = np.sqrt(xj ** 2 + yj ** 2 + (self.h / 1000) ** 2)  # 计算卫星到波位 j 的斜距 (km)
                
                d_i_m = d_i * 1000  # 将卫星到波位 i 的距离转换为米 (m)
                d_j_n = d_j * 1000  # 将卫星到波位 j 的距离转换为米 (m)
                d_horizontal_ij_m = d_horizontal_ij * 1000  # 将波位间的水平距离转换为米 (m)
                
                cos_theta = (d_i_m ** 2 + d_j_n ** 2 - d_horizontal_ij_m ** 2) / (2 * d_i_m * d_j_n)  # 根据余弦定理计算夹角 θ_ij 的余弦值
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 将余弦值限制在 [-1.0, 1.0] 范围内，防止数值计算溢出报错
                theta_mn = np.arccos(cos_theta)  # 利用反余弦函数计算离轴角 θ_ij（弧度）
                
                sin_theta_3db = 0.12703  # 设置 3dB 辅助波束角的正弦常量值
                u_mn = 2.07123 * np.sin(theta_mn) / sin_theta_3db  # 根据公式计算天线方向图无量纲变量 u_mn
                
                if u_mn == 0:  # 防止零作为分母
                    G_theta = 1.0  # 若主轴夹角为 0，增益衰减因子设为 1.0
                else:  # 若不为零
                    J1 = jv(1, u_mn)  # 计算 u_mn 处的 1 阶贝塞尔函数值
                    J3 = jv(3, u_mn)  # 计算 u_mn 处的 3 阶贝塞尔函数值
                    G_theta = (10**(self.G_t/10)) * ((J1 / (2 * u_mn) + 36 * J3 / (u_mn ** 3)) ** 2)  # 根据贝塞尔函数计算天线轴外增益
                    
                path_loss = (self.lambda_wave / (4 * np.pi * d_horizontal_ij_m)) ** 2  # 计算传播路径自由空间损耗因子
                interference[i][j] = G_theta * path_loss  # 计算波位 j 对波位 i 的归一化干扰增益乘积并存入矩阵
                
        return interference  # 返回预计算好的 12x12 干扰矩阵

    # ========================================================================
    # 3. 业务到达率生成
    # ========================================================================
    def _get_traffic_rates(self, current_slot):  # 定义私有方法：根据当前时隙计算各波位业务包到达率
        hour_idx = int((current_slot / self.total_slots_per_day) * 24) % 24  # 计算当前时隙对应一天中的第几小时 (0~23)
        time_factor = self.FIG7_TIME_PROFILE[hour_idx]  # 从全天业务曲线中查表获取当前小时的时间加权系数
        
        total_expected_rate = self.spatial_factor * time_factor * self.base_packet_rate  # 结合空间因子、时间因子与基准率计算总期望到达包数
        
        lambda_rt_expected = np.maximum(total_expected_rate * 0.5, 0.0)  # 按 50% 比例分配实时业务到达率（下限截断为 0）
        lambda_nrt_expected = np.maximum(total_expected_rate * 0.5, 0.0)  # 按 50% 比例分配非实时业务到达率（下限截断为 0）
        
        return lambda_rt_expected, lambda_nrt_expected  # 返回各波位的实时与非实时业务到达率向量

    # ========================================================================
    # 4. 环境重置
    # ========================================================================
    def reset(self):  # 定义环境重置方法，在每轮 Episode 初始时调用
        self.rt_queue_timestamps = [deque() for _ in range(self.N)]  # 重置并清空所有 12 个波位的实时队列
        self.nrt_queue_timestamps = [deque() for _ in range(self.N)]  # 重置并清空所有 12 个波位的非实时队列
        
        self.cumulative_served = np.zeros(self.N)  # 清零累计已服务包数计数器
        self.cumulative_demanded = np.zeros(self.N)  # 清零累计总需求包数计数器
        
        self.current_slot = 0  # 重置当前时隙计数为 0
        
        self.lambda_realtime, self.lambda_nrt = self._get_traffic_rates(self.current_slot)  # 初始化获取第 0 时隙的到达率
        
        return self._get_state()  # 构造并返回初始环境状态信息

    # ========================================================================
    # 5. 状态构造
    # ========================================================================
    def _get_state(self):  # 定义私有方法：打包构造智能体观察到的状态 (State)
        rt_lengths = [len(q) for q in self.rt_queue_timestamps]  # 统计当前 12 个波位各自的实时队列积压包数
        nrt_lengths = [len(q) for q in self.nrt_queue_timestamps]  # 统计当前 12 个波位各自的非实时队列积压包数
        
        packet_matrix = np.vstack([
            np.array(rt_lengths, dtype=np.float32),
            np.array(nrt_lengths, dtype=np.float32)
        ])  # 将实时与非实时队列长度垂直堆叠为一个 (2, 12) 的矩阵
        
        satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)  # 计算当前各波位服务的累积满意度比例 (加平滑项防除零)
        satisfaction = np.clip(satisfaction, 0.0, 1.0)  # 将满意度数值安全裁剪在 [0.0, 1.0] 区间内
        
        return {
            'packet_matrix': packet_matrix,
            'satisfaction': satisfaction
        }  # 返回包含队列矩阵和满意度的状态字典

    # ========================================================================
    # 6. 计算平均时延
    # ========================================================================
    @property
    def realtime_queue(self):  # 定义属性方法：快速获取各波位实时队列长度的 NumPy 数组
        return np.array([len(q) for q in self.rt_queue_timestamps], dtype=np.int32)  # 返回包含 12 个整数的数组

    @property
    def nrt_queue(self):  # 定义属性方法：快速获取各波位非实时队列长度的 NumPy 数组
        return np.array([len(q) for q in self.nrt_queue_timestamps], dtype=np.int32)  # 返回包含 12 个整数的数组

    def _calculate_avg_delay(self, capacity_packets=None):  # 定义内部方法：计算实时业务数据包的平均排队时延
        """
        根据公式(9)精确计算实时数据包的平均排队时延 (秒/ms)
        参数:
            capacity_packets (dict, optional): 当前时隙各波位的服务容量(包数)
        """  # 方法文档说明
        total_rt_packets = np.sum(self.realtime_queue)  # 计算所有波位积压的实时数据包总数
        if total_rt_packets == 0:  # 如果系统中没有任何积压包
            return 0.0  # 直接返回时延 0.0 秒
            
        if capacity_packets is not None and sum(capacity_packets.values()) > 0:  # 若提供了容量参数且容量大于 0
            total_capacity = sum(capacity_packets.values())  # 统计总容量包数
            avg_delay = (total_rt_packets / total_capacity) * self.slot_duration  # 根据 Little 定律用总包数除以服务速率计算时延
        else:  # 若未指定容量
            avg_delay = np.mean(self.realtime_queue) * self.slot_duration  # 根据队列积压平均包数乘以单时隙时长估算时延
            
        return float(avg_delay)  # 将计算出的时延转为浮点数并返回

    # ========================================================================
    # 7. 核心: 执行一步动作
    # ========================================================================
    def step(self, action):  # 定义环境核心交互函数 step，接收选择点亮波位的动作列表
        # ------ (0) 动作前置校验 ------  # 步骤 0 标识：校验与补齐动作波位数
        action = list(set(action))  # 对输入的波位动作进行去重操作
        if len(action) < self.K:  # 如果去重后选中的有效波位数不足 K 个 (4个)
            remaining = [i for i in range(self.N) if i not in action]  # 从未被选中的波位中筛选候选波位
            action += random.sample(remaining, self.K - len(action))  # 随机补充缺少的波位数量直到满足 K 个
        action = action[:self.K]  # 截取前 K 个波位索引作为最终确定的点亮波束组合

        # ------ (1) 排队时延 & 功率分配 ------  # 步骤 1 标识：计算排队时延并进行星上动态功率分配
        current_rt_delays = np.zeros(self.N)  # 创建全零数组，记录 12 个波位各自当前实时包的平均排队时延
        for i in range(self.N):  # 遍历 12 个波位
            rt_q = self.rt_queue_timestamps[i]  # 提取第 i 个波位的实时包到达时间戳双端队列
            if len(rt_q) > 0:  # 若当前波位队列中有等待的数据包
                waiting_slots = self.current_slot - np.array(rt_q)  # 计算每个包在队列中已等待的时隙数
                current_rt_delays[i] = np.mean(waiting_slots) * self.slot_duration  # 计算包平均等待时间（转换为秒）
            else:  # 若队列为空
                current_rt_delays[i] = 0.0  # 排队时延记为 0.0

        weights = {}  # 初始化波束功率分配权重字典
        for i in action:  # 遍历被选中的 4 个激活波位
            total_packets = len(self.rt_queue_timestamps[i]) + len(self.nrt_queue_timestamps[i])  # 计算该波位总积压包数
            delay_weight = current_rt_delays[i]  # 提取该波位当前的实时包排队时延
            weights[i] = (total_packets + 1) * (delay_weight + 1e-5)  # 综合考虑包数与时延计算该波位的功率分配权重数值

        total_weight = sum(weights.values())  # 汇总 4 个激活波位的总权重
        allocated_power = {}  # 初始化分配功率结果字典
        for i in action:  # 遍历 4 个激活波位
            p_i = (weights[i] / total_weight) * self.total_power  # 按权重比例瓜分卫星发射总功率
            allocated_power[i] = min(p_i, self.max_beam_power)  # 将波束分配功率限制在单波束最大功率上限内

        # ------ (2) 干扰计算与信道容量 ------  # 步骤 2 标识：计算同频干扰、SINR及实际传输容量
        capacity_packets = {}  # 初始化各被选波位在当前时隙的最大服务包数字典
        for i in action:  # 遍历被选中的激活波位 i
            interference_sum = 0.0  # 初始化目标波位 i 接收到的总同频干扰功率为 0.0
            for j in action:  # 遍历其他同时激活的发射波位 j
                if i != j:  # 排除自身
                    interference_sum += self.interference_matrix[i][j] * allocated_power[j]  # 累加来自波位 j 的干扰功率（相对因子 * 发射功率）
                    
            noise_power = Bo * T_noise * self.bandwidth  # 根据热噪声公式 N0 = k * T * B 计算总热噪声功率
            
            xi_m, yi_m = self.spot_positions[i] * 1000.0  # 将波位 i 的千米坐标转为米坐标
            d_i_m = np.sqrt(xi_m**2 + yi_m**2 + self.h**2)  # 计算卫星到波位 i 接收端的物理直线斜距 (米)
            path_loss_i = (self.lambda_wave / (4 * np.pi * d_i_m)) ** 2  # 计算有用信号的自由空间路径衰减因子
            
            signal_power = allocated_power[i] * (10 ** (self.G_t / 10)) * (10 ** (self.G_r / 10)) * path_loss_i  # 计算目标波位接收到的有用信号功率
            sinr = signal_power / (interference_sum + noise_power)  # 计算信噪干扰比 (SINR)
            
            capacity_bps = self.bandwidth * np.log2(1 + sinr)  # 根据香农公式计算该波位信道传输容量 (bps)
            max_packets = (capacity_bps * self.slot_duration) / self.packet_size  # 计算该容量在单时隙内最多可传输的数据包个数
            capacity_packets[i] = int(np.floor(max_packets))  # 向下取整，得到当前时隙可完整传输的最大数据包整数值

        # ------ (3) 队列更新: 先入队 ------  # 步骤 3 标识：模拟新业务包到达并入队
        self.lambda_realtime, self.lambda_nrt = self._get_traffic_rates(self.current_slot)  # 获取当前时隙各波位的业务到达率
        for i in range(self.N):  # 遍历所有 12 个波位
            arrive_rt = np.random.poisson(self.lambda_realtime[i])  # 依据泊松分布随机采样生成当前时隙新到达的实时包个数
            arrive_nrt = np.random.poisson(self.lambda_nrt[i])  # 依据泊松分布随机采样生成当前时隙新到达的非实时包个数
            
            for _ in range(arrive_rt):  # 循环将每个新到的实时包
                self.rt_queue_timestamps[i].append(self.current_slot)  # 将其到达的时隙号压入实时队列末尾
            for _ in range(arrive_nrt):  # 循环将每个新到的非实时包
                self.nrt_queue_timestamps[i].append(self.current_slot)  # 将其到达的时隙号压入非实时队列末尾
                
            self.cumulative_demanded[i] += (arrive_rt + arrive_nrt)  # 累加更新该波位的累计总需求包数

        # ------ (4) 队列更新: 后出队 ------  # 步骤 4 标识：按服务容量消耗队列数据包（先实时后非实时，FIFO）
        served_realtime_total = 0  # 初始化当前时隙全网成功服务的实时包总数
        served_nrt_total = 0  # 初始化当前时隙全网成功服务的非实时包总数
        
        for i in action:  # 遍历被点亮的波位
            cap = capacity_packets[i]  # 获取该波位的服务容量包数
            
            served_rt = min(len(self.rt_queue_timestamps[i]), cap)  # 优先服务实时包，计算实际服务的实时包数
            for _ in range(served_rt):  # 循环弹出包
                self.rt_queue_timestamps[i].popleft()  # 从队列左侧弹出最早进入的实时数据包 (FIFO)
            cap -= served_rt  # 扣减已被实时包消耗的服务容量
            served_realtime_total += served_rt  # 累加实时包服务计数
            
            served_nrt = min(len(self.nrt_queue_timestamps[i]), cap)  # 用剩余容量服务非实时包
            for _ in range(served_nrt):  # 循环弹出包
                self.nrt_queue_timestamps[i].popleft()  # 从队列左侧弹出最早进入的非实时数据包 (FIFO)
            served_nrt_total += served_nrt  # 累加非实时包服务计数
            
            self.cumulative_served[i] += (served_rt + served_nrt)  # 累加更新该波位已成功服务的包数

        # ------ (5) 丢包处理 ------  # 步骤 5 标识：清理排队超过时延阈值的超时实时数据包
        max_slots_threshold = int(self.delay_threshold / self.slot_duration)  # 计算最大容忍排队时隙数 (0.4s / 0.01s = 40 时隙)
        for i in range(self.N):  # 遍历 12 个波位
            while len(self.rt_queue_timestamps[i]) > 0:  # 循环检查队列头的最老数据包
                oldest_packet_slot = self.rt_queue_timestamps[i][0]  # 查看队头数据包的到达时隙
                if (self.current_slot - oldest_packet_slot) > max_slots_threshold:  # 如果滞留时间超过最大时隙阈值
                    self.rt_queue_timestamps[i].popleft()  # 将该超时丢弃的数据包弹出队列
                else:  # 如果队头包未超时
                    break  # 后续包滞留时间更短，直接跳出单波位超时检查循环

        # ------ (6) 计算单目标奖励 ------  # 步骤 6 标识：根据设定的优化目标计算强化学习奖励值 Reward
        if self.objective == 'throughput':  # 若目标为最大化吞吐量
            reward = float(served_nrt_total)  # 奖励设为当前时隙服务非实时包的总数
        elif self.objective == 'delay':  # 若目标为最小化时延
            total_rt_packets = sum([len(q) for q in self.rt_queue_timestamps])  # 计算全网实时包总数
            if total_rt_packets > 0:  # 若存在实时包
                system_avg_delay = np.sum(
                    [current_rt_delays[i] * len(self.rt_queue_timestamps[i]) for i in range(self.N)]
                ) / total_rt_packets  # 计算加权全网平均排队时延
            else:  # 若无实时包
                system_avg_delay = 0.0  # 时延记为 0.0
                
            normalized_delay = min(system_avg_delay / self.delay_threshold, 2.0)  # 归一化时延值并设置最大上限防止负奖励爆炸
            reward = -float(normalized_delay)  # 奖励设为负归一化时延
        elif self.objective == 'satisfaction':  # 若目标为最大化满意度
            satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)  # 计算全网各波位满意度
            reward = float(np.mean(satisfaction))  # 奖励设为全网平均满意度

        # ------ (7) 时隙递增与状态生成 ------  # 步骤 7 标识：前进一个时隙并生成下一状态
        self.current_slot += 1  # 当前系统运行时隙自增 1
        next_state = self._get_state()  # 构造生成执行动作后的下一个状态 Next State
        done = False  # 标志位：设为 False，表示单局持续运行不中断

        # ------ (8) 返回 info ------  # 步骤 8 标识：组装详细的调试与统计信息
        info = {
            'avg_delay': np.mean(current_rt_delays),  # 记录当前波位的平均排队时延 (s)
            'throughput': served_nrt_total,  # 记录吞吐量包数
            'satisfaction': np.mean(self.cumulative_served / (self.cumulative_demanded + 1e-8)),  # 记录平均满意度
            'action': action,  # 记录执行的动作
            'capacity': capacity_packets  # 记录分配的服务容量
        }  # 组装 info 字典
        
        return next_state, reward, done, info  # 返回强化学习环境步进标准元组 (下一个状态, 奖励, 结束标志, 附加信息)

# ============================================================================
# 测试代码 (独立运行环境，验证干扰矩阵和队列逻辑)
# ============================================================================
if __name__ == "__main__":  # 判断当前脚本是否为主入口执行
    env = LEOSatEnv(objective='throughput')  # 实例化一个优化吞吐量的低轨卫星环境对象
    state = env.reset()  # 重置环境并获取初始状态
    print(f"初始状态包矩阵形状: {state['packet_matrix'].shape}")  # 打印输出状态中数据包矩阵的 Dimensions (2, 12)
    print(f"初始满意度形状: {state['satisfaction'].shape}")  # 打印输出状态中满意度向量的形状 (12,)
    
    for t in range(5):  # 循环模拟运行 5 个时隙的交互
        action = random.sample(range(env.N), env.K)  # 随机生成选取 4 个波位的动作
        next_state, reward, done, info = env.step(action)  # 与环境交互执行一步
        print(f"时隙 {t + 1}: 动作 {action} -> 奖励 {reward:.2f}, 吞吐量 {info['throughput']}")  # 打印输出当前步的动作、奖励及吞吐量结果
        
    print("环境测试通过！")  # 打印测试成功提示