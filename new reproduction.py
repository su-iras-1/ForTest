
# 本文件对应论文第1章内容（1.1~1.5节），实现公式(1)~(24)描述的系统模型

import numpy as np  # 引入numpy，用于数组运算、三角函数、统计计算等
from scipy.special import jv  # 引入贝塞尔函数jv，用于天线增益公式(3)中的一阶/三阶贝塞尔函数
import random  # 引入random模块，用于随机采样波位动作
from collections import deque  # 引入双端队列deque，用于高效地在队首/队尾增删数据包时间戳

T_noise = 300  # 全局变量：系统噪声温度，单位K，对应论文表1参数
Bo = 1.38e-23  # 全局变量：玻尔兹曼常数，单位J/K，用于计算热噪声功率


class LEOSatEnv:
    # 类文档字符串：说明该环境模拟的是单颗低轨卫星的跳波束资源调度场景
    """
    模拟卫星环境
    单颗低轨卫星跳波束环境
    特点：
    1. 12个固定地面波位 (图5)
    2. 4个可同时激活的波束 (表1)
    3. 包含同频干扰的物理层计算 (公式2~7)
    4. 实时/非实时双队列模型 (公式9~11)
    5. 支持单目标切换 (throughput / delay / satisfaction)
    """

    def __init__(self, objective='throughput'):
        # 构造函数文档字符串：说明初始化时的入参含义
        """
        初始化环境（对应论文 表1 参数）
        参数:
            objective (str): 单目标优化类型，可选 'throughput' | 'delay' | 'satisfaction'
        """
        # ===== 第1部分：空间与物理层参数初始化，对应论文表1 =====
        self.t_num = 36  # 星座轨道面数量，共36条轨道
        self.signle_num = 20  # 每条轨道上部署的卫星数量，共20颗
        self.h = 570e3  # 卫星轨道高度，570公里，换算为米存储
        self.A = 70  # 轨道倾角，单位为度
        self.Total_num = 720  # 星座卫星总数
        self.N = 12  # 每颗卫星覆盖的固定波位（点波束位置）总数
        self.K = 4  # 每个时隙内可同时点亮（激活）的波束数量
        self.fc = 20e9  # 载波频率，20GHz（Ka频段）
        self.bandwidth = 200e6  # 单波束可用带宽，200MHz
        self.total_power = 120  # 卫星平台可分配的总发射功率，120瓦
        self.max_beam_power = 60  # 单个波束允许分配到的最大功率上限，60瓦
        self.G_t = 40  # 卫星发射天线增益，单位dB
        self.G_r = 50  # 地面用户终端接收天线增益，单位dB
        self.slot_duration = 0.01  # 每个跳波束时隙的时长，10毫秒
        self.delay_threshold = 0.4  # 实时业务允许的最大排队时延阈值，400毫秒
        self.packet_size = 10 * 1024 * 8  # 单个数据包大小，10kbit换算为比特数
        self.lambda_wave = 3e8 / self.fc  # 根据光速与载波频率计算电磁波波长

        # ===== 第2部分：波位几何布局初始化，对应论文1.3节、图5 =====
        self.spot_positions = self._generate_spot_positions()  # 调用内部方法生成12个波位的二维坐标

        # ===== 第3部分：预计算同频干扰矩阵，对应公式(2)~(5) =====
        self.interference_matrix = self._precompute_interference()  # 预先算好任意两波位间的干扰系数矩阵，避免运行时重复计算

        # ===== 第4部分：队列与统计变量初始化，对应论文1.4~1.5节 =====
        # 用deque列表代替普通计数数组，是为了能记录每个数据包的实际入队时隙，从而精确算延迟和判断超时
        self.rt_queue_timestamps = [deque() for _ in range(self.N)]  # 每个波位一个实时队列，存放各数据包到达时的时隙编号
        self.nrt_queue_timestamps = [deque() for _ in range(self.N)]  # 每个波位一个非实时队列，同样存放到达时隙编号

        self.base_demand = np.array([800, 700, 1300, 300, 980, 250,
                                     1000, 275, 80, 600, 50, 200])  # 12个波位的基准业务需求量，取自论文图6的估测数据

        # 按论文1.4节公式，用变异系数（标准差/均值）衡量业务空间分布的不均匀程度
        mean_demand = np.mean(self.base_demand)  # 计算12个波位基准需求的均值
        std_demand = np.std(self.base_demand)  # 计算12个波位基准需求的标准差
        self.zeta = std_demand / mean_demand  # 空间离散系数zeta，数值约为0.7205

        self.spatial_factor = self.base_demand / mean_demand  # 将各波位需求除以均值，得到归一化的空间不均匀因子（均值为1）

        # 论文图7给出的24小时归一化业务量曲线，每小时一个采样点
        self.FIG7_TIME_PROFILE = np.array([
            0.03, 0.03, 0.03, 0.03, 0.03, 0.03,  # 凌晨1点到6点，业务量处于低谷
            0.15, 0.26, 0.42, 0.60, 1.00, 0.90,  # 早7点到中午12点，业务量爬升，11点达到全天最高峰
            0.85, 0.78, 0.66, 0.78, 0.82, 0.68,  # 下午1点到18点，午后波动，17点出现次高峰
            0.42, 0.32, 0.18, 0.10, 0.06, 0.03])  # 傍晚到午夜，业务量逐渐回落

        self.total_slots_per_day = 8_640_000  # 一天24小时按10ms/时隙换算出的总时隙数
        self.base_packet_rate = 50  # 基准数据包到达率，即每个10ms时隙平均到达的包数量

        # 满意度统计所需的累加器，对应公式(24)
        self.cumulative_served = np.zeros(self.N)  # 记录每个波位从仿真开始至今累计成功服务的数据包数
        self.cumulative_demanded = np.zeros(self.N)  # 记录每个波位从仿真开始至今累计到达的数据包总需求数

        # 时间推进与到达率相关的状态变量，对应公式(8)及图6、图7
        self.current_slot = 0  # 当前仿真时隙计数器，从0开始
        self.lambda_realtime = None  # 占位：各波位当前的实时业务泊松到达率，运行时由方法计算填充
        self.lambda_nrt = None  # 占位：各波位当前的非实时业务泊松到达率，运行时由方法计算填充

        self.objective = objective  # 保存本次实例化选择的单一优化目标

        print(f"[Env] 初始化完成 | 目标: {objective} | 波位数: {self.N} | 波束数: {self.K}")  # 打印一条初始化完成的提示日志，方便调试确认参数

    # ========================================================================
    # 方法1：生成波位几何布局，对应论文1.3节、图5，坐标单位为公里
    # ========================================================================
    def _generate_spot_positions(self):
        R = 73  # 单个波位覆盖半径，单位公里
        d = np.sqrt(3) * R  # 按正六边形蜂窝布局公式，计算相邻波位中心之间的距离
        positions = [(0, 0)]  # 波位列表初始化，第1个波位固定在星下点原点位置

        # 生成第一层的6个波位：均匀分布在中心波位周围，间隔60度一个，起始角30度
        for k in range(6):
            angle = np.deg2rad(60 * k + 30)  # 将角度换算为弧度，第k个波位对应角度为60k+30度
            x = d * np.cos(angle)  # 该波位相对中心的x坐标
            y = d * np.sin(angle)  # 该波位相对中心的y坐标
            positions.append((x, y))  # 把计算出的坐标追加进波位列表

        # 生成第二层外围的5个波位：利用第一层坐标的矢量组合，使外层波位与内层紧密贴合，对应图5布局
        outer_offsets = [
            (d * np.cos(np.deg2rad(30)) + d * np.cos(np.deg2rad(90)),
             d * np.sin(np.deg2rad(30)) + d * np.sin(np.deg2rad(90))),  # 第8个波位坐标
            (d * np.cos(np.deg2rad(90)) + d * np.cos(np.deg2rad(150)),
             d * np.sin(np.deg2rad(90)) + d * np.sin(np.deg2rad(150))),  # 第9个波位坐标
            (2 * d * np.cos(np.deg2rad(150)),
             2 * d * np.sin(np.deg2rad(150))),  # 第10个波位坐标
            (d * np.cos(np.deg2rad(150)) + d * np.cos(np.deg2rad(210)),
             d * np.sin(np.deg2rad(150)) + d * np.sin(np.deg2rad(210))),  # 第11个波位坐标
            (d * np.cos(np.deg2rad(210)) + d * np.cos(np.deg2rad(270)),
             d * np.sin(np.deg2rad(210)) + d * np.sin(np.deg2rad(270)))  # 第12个波位坐标
        ]
        for x, y in outer_offsets:  # 遍历第二层算好的5组坐标
            positions.append((x, y))  # 依次追加进波位列表

        return np.array(positions)  # 将列表转换为numpy数组并返回，形状为(12, 2)

    # ========================================================================
    # 方法2：预计算同频干扰矩阵，对应论文公式(2)~(5)
    # ========================================================================
    def _precompute_interference(self):
        # 方法文档字符串：说明该方法目的是提前算好干扰矩阵以加速后续训练
        """
        提前算好 12x12 的干扰功率矩阵 (W)
        避免在 step() 中重复计算贝塞尔函数，大幅提升训练速度
        """
        N = self.N  # 取出波位总数12，简化后续下标书写
        interference = np.zeros((N, N))  # 初始化一个12x12的全零矩阵，用于存放每对波位间的干扰系数

        for i in range(N):  # 外层循环遍历被干扰的目标波位i
            for j in range(N):  # 内层循环遍历产生干扰的波位j
                if i == j:
                    continue  # 波位自己不会对自己产生干扰，跳过该组合

                xi, yi = self.spot_positions[i]  # 取出波位i的平面坐标
                xj, yj = self.spot_positions[j]  # 取出波位j的平面坐标

                d_horizontal_ij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)  # 计算波位i与j在地面上的水平距离，单位公里

                d_i = np.sqrt(xi ** 2 + yi ** 2 + (self.h / 1000) ** 2)  # 计算卫星到波位i的空间直线距离，单位公里
                d_j = np.sqrt(xj ** 2 + yj ** 2 + (self.h / 1000) ** 2)  # 计算卫星到波位j的空间直线距离，单位公里

                # 后续公式统一采用米作为单位，这里把公里换算成米
                d_i_m = d_i * 1000  # 卫星到波位i的距离，单位米
                d_j_n = d_j * 1000  # 卫星到波位j的距离，单位米
                d_horizontal_ij_m = d_horizontal_ij * 1000  # 波位i与j的水平距离，单位米

                # 对应公式(5)：利用余弦定理计算卫星视角下波位i、j之间的夹角theta_mn
                cos_theta = (d_i_m ** 2 + d_j_n ** 2 - d_horizontal_ij_m ** 2) / (2 * d_i_m * d_j_n)  # 余弦定理求夹角的余弦值

                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数值截断，防止浮点误差导致余弦值超出[-1,1]范围引发反三角函数报错
                theta_mn = np.arccos(cos_theta)  # 由余弦值反推出夹角theta_mn，单位弧度

                # 对应公式(4)：根据3dB波束宽度换算出归一化角度参数u_mn
                sin_theta_3db = 0.12703  # 3dB波束张角正弦值的预先算好的常数，由波位半径与轨道高度换算得到
                u_mn = 2.07123 * np.sin(theta_mn) / sin_theta_3db  # 按公式(4)计算归一化角度参数u_mn

                # 对应公式(3)：利用贝塞尔函数计算天线方向图增益G(theta)
                if u_mn == 0:
                    G_theta = 1.0  # u_mn为0时（正对波束中心），直接取增益为最大值1.0，避免除零错误
                else:
                    J1 = jv(1, u_mn)  # 计算一阶贝塞尔函数值
                    J3 = jv(3, u_mn)  # 计算三阶贝塞尔函数值
                    G_theta = (10 ** (self.G_t / 10)) * (
                                (J1 / (2 * u_mn) + 36 * J3 / (u_mn ** 3)) ** 2)  # 按公式(3)结合天线主增益算出方向图增益

                # 对应公式(2)：计算干扰功率相关的几何衰减因子
                path_loss = (self.lambda_wave / (4 * np.pi * d_horizontal_ij_m)) ** 2  # 自由空间路径损耗因子，只取决于波长与水平距离

                interference[i][j] = G_theta * path_loss  # 将方向图增益与路径损耗相乘，存入干扰矩阵对应位置，代表单位功率下j对i的干扰强度

        return interference  # 返回填充完毕的12x12干扰系数矩阵

    # ========================================================================
    # 方法3：生成业务到达率，对应论文1.4节公式(8)、图6、图7
    # ========================================================================
    def _get_traffic_rates(self, current_slot):
        hour_idx = int((current_slot / self.total_slots_per_day) * 24) % 24  # 将当前时隙换算成一天中对应的小时序号（0~23）

        time_factor = self.FIG7_TIME_PROFILE[hour_idx]  # 取出该小时对应的时间归一化因子

        total_expected_rate = self.spatial_factor * time_factor * self.base_packet_rate  # 结合空间不均匀因子、时间因子与基准速率，得到每个波位当前时隙的期望总到达率

        # 假设实时与非实时业务各占一半（论文未明确给出比例，此处取0.5:0.5作为近似）
        lambda_rt_expected = np.maximum(total_expected_rate * 0.5, 0.0)  # 实时业务期望到达率，并用np.maximum做非负截断
        lambda_nrt_expected = np.maximum(total_expected_rate * 0.5, 0.0)  # 非实时业务期望到达率，同样做非负截断

        return lambda_rt_expected, lambda_nrt_expected  # 返回两个长度为12的数组，分别是各波位实时/非实时到达率

    # ========================================================================
    # 方法4：环境重置，对应算法1步骤9
    # ========================================================================
    def reset(self):
        self.rt_queue_timestamps = [deque() for _ in range(self.N)]  # 重新创建空的实时队列列表，清空所有历史积压包
        self.nrt_queue_timestamps = [deque() for _ in range(self.N)]  # 重新创建空的非实时队列列表，清空所有历史积压包

        self.cumulative_served = np.zeros(self.N)  # 累计已服务包数清零
        self.cumulative_demanded = np.zeros(self.N)  # 累计总需求包数清零

        self.current_slot = 0  # 时隙计数器重置为0

        self.lambda_realtime, self.lambda_nrt = self._get_traffic_rates(self.current_slot)  # 按第0个时隙重新计算初始到达率

        return self._get_state()  # 返回重置后的初始观测状态

    # ========================================================================
    # 方法5：构造观测状态，对应论文公式(19)~(24)
    # ========================================================================
    def _get_state(self):
        rt_lengths = [len(q) for q in self.rt_queue_timestamps]  # 统计每个波位当前实时队列的积压长度
        nrt_lengths = [len(q) for q in self.nrt_queue_timestamps]  # 统计每个波位当前非实时队列的积压长度

        packet_matrix = np.vstack([
            np.array(rt_lengths, dtype=np.float32),  # 第一行：各波位实时队列长度
            np.array(nrt_lengths, dtype=np.float32)  # 第二行：各波位非实时队列长度
        ])  # 将两行堆叠成形状为(2, 12)的矩阵，代表当前队列状态

        satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)  # 计算各波位累计服务满意度，分母加小量防止除零
        satisfaction = np.clip(satisfaction, 0.0, 1.0)  # 将满意度截断到[0,1]区间，避免异常值

        return {
            'packet_matrix': packet_matrix,  # 状态字典的第一个字段：队列包数矩阵
            'satisfaction': satisfaction  # 状态字典的第二个字段：各波位满意度向量
        }

    # ========================================================================
    # 方法6：计算平均排队时延，对应论文公式(9)
    # ========================================================================
    @property
    def realtime_queue(self):  # 只读属性：动态返回各波位实时队列当前的积压包数数组
        return np.array([len(q) for q in self.rt_queue_timestamps], dtype=np.int32)  # 遍历每个波位的deque取长度，组成数组返回

    @property
    def nrt_queue(self):  # 只读属性：动态返回各波位非实时队列当前的积压包数数组
        return np.array([len(q) for q in self.nrt_queue_timestamps], dtype=np.int32)  # 遍历每个波位的deque取长度，组成数组返回

    def _calculate_avg_delay(self, capacity_packets=None):
        # 方法文档字符串：说明该函数按公式(9)计算实时业务的平均排队时延
        """
        根据公式(9)精确计算实时数据包的平均排队时延 (秒)
        参数:
            capacity_packets (dict, optional): 当前时隙各波位的服务容量(包数)
        """
        total_rt_packets = np.sum(self.realtime_queue)  # 统计全部波位当前实时队列的总积压包数
        if total_rt_packets == 0:
            return 0.0  # 没有积压包时时延直接为0，避免后续除零

        if capacity_packets is not None and sum(capacity_packets.values()) > 0:
            # 方法A：如果提供了本时隙实际服务容量，则按利特尔定律更精确地估算时延
            total_capacity = sum(capacity_packets.values())  # 汇总本时隙全部波位的服务容量
            avg_delay = (total_rt_packets / total_capacity) * self.slot_duration  # 总积压包数除以总服务速率，再乘以单时隙时长
        else:
            # 方法B：未提供容量时，退化为按理论公式(9)用平均队列长度乘以时隙时长估算
            avg_delay = np.mean(self.realtime_queue) * self.slot_duration  # 各波位平均积压包数乘以单时隙时长

        return float(avg_delay)  # 转换为普通浮点数并返回

    # ========================================================================
    # 方法7：核心步进函数，对应算法1步骤11~12
    # ========================================================================
    def step(self, action):
        # ------ 第0步：动作合法性校验，确保最终选出K个不重复波位 ------
        action = list(set(action))  # 先去重，防止调用方传入重复的波位编号
        if len(action) < self.K:
            remaining = [i for i in range(self.N) if i not in action]  # 找出未被选中的波位编号
            action += random.sample(remaining, self.K - len(action))  # 从剩余波位中随机补齐，凑够K个
        action = action[:self.K]  # 如果传入超过K个，截断只保留前K个

        # ------ 第1步：计算各波位当前实际排队时延，并据此分配功率，对应公式(18) ------
        current_rt_delays = np.zeros(self.N)  # 初始化数组，记录本时隙每个波位实时队列的平均等待时延
        for i in range(self.N):  # 遍历全部12个波位
            rt_q = self.rt_queue_timestamps[i]  # 取出第i个波位的实时队列
            if len(rt_q) > 0:
                waiting_slots = self.current_slot - np.array(rt_q)  # 用当前时隙减去每个包的入队时隙，得到各包已等待的时隙数
                current_rt_delays[i] = np.mean(waiting_slots) * self.slot_duration  # 取平均等待时隙数再乘以单时隙时长，换算成秒
            else:
                current_rt_delays[i] = 0.0  # 队列为空则时延记为0

        weights = {}  # 用于存放本次被激活波位的功率分配权重
        for i in action:  # 只针对本时隙被选中激活的K个波位计算权重
            total_packets = len(self.rt_queue_timestamps[i]) + len(self.nrt_queue_timestamps[i])  # 该波位实时与非实时队列的总积压包数
            delay_weight = current_rt_delays[i]  # 取该波位刚算出的实时排队时延作为权重因子之一
            weights[i] = (total_packets + 1) * (delay_weight + 1e-5)  # 按公式(18)，用（总包数+1）乘以（时延+微小平滑项）作为权重
        total_weight = sum(weights.values())  # 汇总本时隙所有激活波位的权重总和，用于归一化

        allocated_power = {}  # 用于存放每个激活波位最终分配到的发射功率
        for i in action:  # 遍历本时隙激活的波位
            p_i = (weights[i] / total_weight) * self.total_power  # 按权重占比，从总功率中分配给该波位的功率
            allocated_power[i] = min(p_i, self.max_beam_power)  # 与单波束最大功率上限取较小值，防止超限

        # ------ 第2步：计算干扰与信道容量，对应公式(2)~(7) ------
        capacity_packets = {}  # 用于存放每个激活波位本时隙能服务的最大包数
        for i in action:  # 遍历每个激活波位i，计算其可获得的信道容量
            interference_sum = 0.0  # 初始化波位i受到的总干扰功率
            for j in action:  # 遍历同一时隙内其它被激活的波位j
                if i != j:
                    interference_sum += self.interference_matrix[i][j] * allocated_power[
                        j]  # 累加j对i的干扰功率贡献（干扰系数乘以j的发射功率）
            noise_power = Bo * T_noise * self.bandwidth  # 按热噪声公式计算接收端噪声功率

            xi_m, yi_m = self.spot_positions[i] * 1000.0  # 取出波位i坐标并从公里换算为米
            d_i_m = np.sqrt(xi_m ** 2 + yi_m ** 2 + self.h ** 2)  # 计算卫星到波位i的直线距离，全部用米为单位
            path_loss_i = (self.lambda_wave / (4 * np.pi * d_i_m)) ** 2  # 计算卫星到波位i的自由空间路径损耗

            signal_power = allocated_power[i] * (10 ** (self.G_t / 10)) * (
                        10 ** (self.G_r / 10)) * path_loss_i  # 结合发射功率、收发天线增益与路径损耗，算出接收端有用信号功率

            sinr = signal_power / (interference_sum + noise_power)  # 计算信干噪比：有用信号功率除以（干扰功率+噪声功率）
            capacity_bps = self.bandwidth * np.log2(1 + sinr)  # 按香农公式计算信道容量，单位比特/秒
            max_packets = (capacity_bps * self.slot_duration) / self.packet_size  # 将容量换算成本时隙内最多能传输的数据包数
            capacity_packets[i] = int(np.floor(max_packets))  # 向下取整，得到本时隙该波位的实际服务容量（包数）

        # ------ 第3步：队列更新——新业务先按泊松分布到达并入队，记录到达时隙 ------
        self.lambda_realtime, self.lambda_nrt = self._get_traffic_rates(self.current_slot)  # 按当前时隙重新计算各波位的到达率

        for i in range(self.N):  # 遍历全部12个波位处理业务到达
            arrive_rt = np.random.poisson(self.lambda_realtime[i])  # 按泊松分布随机采样该波位本时隙新到达的实时包数
            arrive_nrt = np.random.poisson(self.lambda_nrt[i])  # 按泊松分布随机采样该波位本时隙新到达的非实时包数

            for _ in range(arrive_rt):
                self.rt_queue_timestamps[i].append(self.current_slot)  # 把每个新到达的实时包的入队时隙压入队尾
            for _ in range(arrive_nrt):
                self.nrt_queue_timestamps[i].append(self.current_slot)  # 把每个新到达的非实时包的入队时隙压入队尾

            self.cumulative_demanded[i] += (arrive_rt + arrive_nrt)  # 更新该波位的累计总需求包数，对应公式(11)分母部分

        # ------ 第4步：队列更新——按容量出队服务，实时优先、非实时次之，均为先进先出 ------
        served_realtime_total = 0  # 本时隙全部波位累计服务的实时包总数
        served_nrt_total = 0  # 本时隙全部波位累计服务的非实时包总数
        for i in action:  # 只有被激活的波位才能在本时隙提供服务
            cap = capacity_packets[i]  # 取出该波位本时隙的服务容量

            served_rt = min(len(self.rt_queue_timestamps[i]), cap)  # 实时队列优先服务，服务数取队列长度与剩余容量的较小值
            for _ in range(served_rt):
                self.rt_queue_timestamps[i].popleft()  # 按先进先出，从队首弹出被服务掉的实时包
            cap -= served_rt  # 扣减掉已用于服务实时包的容量
            served_realtime_total += served_rt  # 累加进本时隙实时业务服务总数

            served_nrt = min(len(self.nrt_queue_timestamps[i]), cap)  # 用剩余容量服务非实时队列
            for _ in range(served_nrt):
                self.nrt_queue_timestamps[i].popleft()  # 按先进先出，从队首弹出被服务掉的非实时包
            served_nrt_total += served_nrt  # 累加进本时隙非实时业务服务总数

            self.cumulative_served[i] += (served_rt + served_nrt)  # 更新该波位的累计已服务包数，对应公式(24)分子部分

        # ------ 第5步：丢包处理——超过时延阈值的最老实时包被丢弃，对应公式(16) ------
        max_slots_threshold = int(self.delay_threshold / self.slot_duration)  # 把时延阈值换算成对应的时隙数（0.4s/0.01s=40个时隙）
        for i in range(self.N):  # 遍历全部波位检查是否有超时的实时包
            while len(self.rt_queue_timestamps[i]) > 0:
                oldest_packet_slot = self.rt_queue_timestamps[i][0]  # 查看队首（最早入队）的数据包的到达时隙
                if (self.current_slot - oldest_packet_slot) > max_slots_threshold:
                    self.rt_queue_timestamps[i].popleft()  # 已等待超过阈值，直接丢弃这个最老的包
                else:
                    break  # 队首包都还没超时，说明后面更新的包也一定没超时，提前结束循环

        # ------ 第6步：按选定的单一目标计算本时隙奖励，对应公式(9)/(10)/(11) ------
        if self.objective == 'throughput':
            reward = float(served_nrt_total)  # 吞吐量目标：直接用本时隙服务的非实时包数作为奖励
        elif self.objective == 'delay':
            total_rt_packets = sum([len(q) for q in self.rt_queue_timestamps])  # 统计当前全部波位实时队列的总积压包数
            if total_rt_packets > 0:
                system_avg_delay = np.sum(
                    [current_rt_delays[i] * len(self.rt_queue_timestamps[i]) for i in range(self.N)]
                ) / total_rt_packets  # 按各波位积压包数加权，计算全系统平均时延
            else:
                system_avg_delay = 0.0  # 没有积压包时系统平均时延为0
            normalized_delay = min(system_avg_delay / self.delay_threshold, 2.0)  # 用阈值归一化时延，并设上限2.0防止极端值
            reward = -float(normalized_delay)  # 时延目标：时延越小奖励越高，因此取负数
        elif self.objective == 'satisfaction':
            satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)  # 重新计算各波位累计满意度
            reward = float(np.mean(satisfaction))  # 满意度目标：用全部波位满意度的均值作为奖励

        # ------ 第7步：时隙推进并生成下一步观测状态 ------
        self.current_slot += 1  # 时隙计数器加1，进入下一个仿真时隙
        next_state = self._get_state()  # 调用状态构造方法生成新的观测
        done = False  # 该环境不设置终止条件，始终返回False（可根据需要在外部控制episode长度）

        # ------ 第8步：整理并返回本步的辅助信息字典 ------
        info = {
            'avg_delay': np.mean(current_rt_delays),  # 本时隙各波位实时时延的平均值
            'throughput': served_nrt_total,  # 本时隙服务的非实时包总数
            'satisfaction': np.mean(self.cumulative_served / (self.cumulative_demanded + 1e-8)),  # 当前全局平均满意度
            'action': action,  # 本时隙实际执行的动作（被激活的波位编号列表）
            'capacity': capacity_packets  # 本时隙各激活波位的信道容量（包数）
        }

        return next_state, reward, done, info  # 按gym风格接口返回：新状态、奖励、结束标志、辅助信息


# ============================================================================
# 测试代码：独立运行本文件时执行，用于验证干扰矩阵与队列逻辑是否正常
# ============================================================================
if __name__ == "__main__":
    env = LEOSatEnv(objective='throughput')  # 实例化环境，选择吞吐量作为优化目标
    state = env.reset()  # 重置环境，获取初始观测状态
    print(f"初始状态包矩阵形状: {state['packet_matrix'].shape}")  # 打印初始包矩阵的形状，应为(2, 12)
    print(f"初始满意度形状: {state['satisfaction'].shape}")  # 打印初始满意度向量的形状，应为(12,)

    for t in range(5):  # 连续测试执行5个时隙
        action = random.sample(range(env.N), env.K)  # 从12个波位中随机抽取K个作为本时隙动作
        next_state, reward, done, info = env.step(action)  # 执行一步仿真，获取返回结果
        print(f"时隙 {t + 1}: 动作 {action} -> 奖励 {reward:.2f}, 吞吐量 {info['throughput']}")  # 打印本时隙的动作、奖励与吞吐量

    print("环境测试通过！")  # 循环结束后打印测试通过提示