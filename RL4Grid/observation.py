import copy
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *

class Observation:
    def __init__(self, ppc, load_bus, timestep, action_space, steps_to_reconnect_line, count_soft_overflow_steps,
                 rho, v_or, v_ex, a_or, a_ex, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen, gen_start_flag, curstep_renewable_gen_p_max,
                 nextstep_renewable_gen_p_max, future_renewable_gen_p_max, future_load_p, rounded_gen_p, nextstep_load_p):
        self.timestep = timestep
        self.gen_p = ppc['gen'][:, PG]
        self.gen_q = ppc['gen'][:, QG]
        self.gen_v = ppc['gen'][:, VG]
        self.load_p = ppc['bus'][load_bus, PD]
        self.load_q = ppc['bus'][load_bus, QD]
        self.load_v = ppc['bus'][load_bus, VM]
        self.p_or = ppc['branch'][:, PF]
        self.q_or = ppc['branch'][:, QF]
        self.v_or = v_or
        self.a_or = a_or
        self.p_ex = ppc['branch'][:, PT]
        self.q_ex = ppc['branch'][:, QT]
        self.v_ex = v_ex
        self.a_ex = a_ex
        self.line_status = ppc['branch'][:, BR_STATUS]
        self.bus_v = ppc['bus'][:, VM]
        self.bus_ang = ppc['bus'][:, VA]
        self.action_space = action_space                                    # legal action space
        self.steps_to_reconnect_line = steps_to_reconnect_line              # 线路断开后恢复连接的剩余时间步数
        self.count_soft_overflow_steps = count_soft_overflow_steps          # 线路软过载的已持续时间步数
        self.rho = rho
        self.gen_status = ppc['gen'][:, GEN_STATUS]                      # 机组开关机状态（1为开机，0位关机）
        self.steps_to_recover_gen = steps_to_recover_gen                    # 机组关机后可以重新开机的时间步（如果机组状态为开机，则值为0）
        self.steps_to_close_gen = steps_to_close_gen                        # 机组开机后可以重新关机的时间步（如果机组状态为关机，则值为0）
        self.steps_to_min_gen = steps_to_min_gen                            # 机组得到开机指令后达到最小功率所需要的时间步
        self.gen_start_flag = gen_start_flag                                # 机组进入启动程序标志
        self.curstep_renewable_gen_p_max = curstep_renewable_gen_p_max      # 当前时间步新能源机组的最大有功出力
        self.nextstep_renewable_gen_p_max = nextstep_renewable_gen_p_max    # 下一时间步新能源机组的最大有功出力
        self.nextstep_load_p = nextstep_load_p                              # 下一时间步的负荷
        self.future_renewable_gen_p_max = future_renewable_gen_p_max        # 未来X时间步的可再生能源最大功率和
        self.future_load_p = future_load_p                                  # 未来X时间步的负荷功率之和