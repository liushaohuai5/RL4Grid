
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network: case4917_goc
Auto-generated on: 2025-08-31
"""

import numpy as np
import os
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
path = os.path.dirname(__file__) + "/"

def case4917_goc():

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc = dict(version=2)
    
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = np.load(path+"case4917_goc_bus.npy")

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = np.load(path+"case4917_goc_gen.npy")


    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = np.load(path+"case4917_goc_branch.npy")

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = np.load(path+"case4917_goc_gencost.npy")

    # ppc['branch'][:, [RATE_A, RATE_B, RATE_C]] = ppc['branch'][:, [RATE_A, RATE_B, RATE_C]].clip(50, 1e8)
    sensitive_line_idxs = []
    suggested_capacities = []
    for i, idx in enumerate(sensitive_line_idxs):
        ppc['branch'][idx, [RATE_A, RATE_B, RATE_C]] = suggested_capacities[i]

    ppc["network"] = "case4917_goc"
    ppc["num_bus"] = ppc["bus"].shape[0]
    bus_gen = [[] for _ in range(ppc["num_bus"])]
    for i, bus in enumerate(ppc["gen"][:, GEN_BUS].tolist()):
        idx = ppc["bus"][:, BUS_I].tolist().index(bus)
        bus_gen[idx].append(i)
    del_rows = []
    for bg in bus_gen:
        if len(bg) > 1:
            g1 = bg[0]
            for g2 in bg:
                if g2 != g1:
                    ppc["gen"][g1, [PG, QG, QMAX, QMIN, PMAX, PMIN]] += ppc["gen"][g2, [PG, QG, QMAX, QMIN, PMAX, PMIN]]
                    del_rows.append(g2)
    ppc["gen"] = np.delete(ppc["gen"], del_rows, axis=0)
    ppc["gencost"] = np.delete(ppc["gencost"], del_rows, axis=0)
    ppc["num_gen"] = ppc["gen"].shape[0]
    ppc["num_line"] = ppc["branch"].shape[0]
    ppc["gen_bus"] = []
    for bus in ppc["gen"][:, GEN_BUS].tolist():
        idx = ppc["bus"][:, BUS_I].tolist().index(bus)
        ppc["gen_bus"].append(idx)
    ppc["load_bus"] = np.nonzero(ppc["bus"][:, PD])[0].tolist()
    ppc["num_load"] = len(ppc["load_bus"])
    balanced_bus = ppc["bus"][np.where(ppc["bus"][:, BUS_TYPE] == 3)[0][0], BUS_I]
    ppc["balanced_id"] = np.where(ppc["gen"][:, GEN_BUS] == balanced_bus)[0][0]
    k = ppc["gen"].shape[0] // 3
    a = ppc["gen"][:, PMAX].tolist()
    ppc["renewable_ids"] = sorted(range(len(a)), key=a.__getitem__)[:k] 
    ppc["thermal_ids"] = list(set(range(len(a))) - set([ppc["balanced_id"]]) - set(ppc["renewable_ids"]))
    ppc["gen"][ppc["renewable_ids"], PMIN] = 0.0
    ppc["gencost"][ppc["renewable_ids"], -2] = 0
    ppc["gencost"][ppc["renewable_ids"], -1] = 0
    ppc["sorted_controlable_ids"] = sorted(ppc["renewable_ids"] + ppc["thermal_ids"])

    if ppc["balanced_id"] != ppc["gen"][:, PMAX].argmax():
        ppc["gen"][ppc["balanced_id"], PMAX] = ppc["gen"][ppc["gen"][:, PMAX].argmax(), PMAX]

    ppc["gen"][ppc["balanced_id"], PMAX] *= 3
    ppc["gen"][ppc["balanced_id"], QMIN] = -ppc["gen"][ppc["balanced_id"], QMAX]

    ppc["min_gen_p"] = ppc["gen"][:, PMIN].tolist()
    ppc["max_gen_p"] = ppc["gen"][:, PMAX].tolist()
    for i in range(ppc["num_gen"]):
        if i in ppc["thermal_ids"]:
            ppc["min_gen_p"][i] = np.around(0.2 * ppc["gen"][i, PMAX], decimals=2).tolist()
            ppc["gen"][i, PMIN] = ppc["min_gen_p"][i]
    # for i, bus in enumerate(ppc["gen"][:, GEN_BUS].tolist()):
    #     bus_idx = ppc["bus"][:, BUS_I].tolist().index(bus)
    #     if int(ppc["bus"][bus_idx, BUS_TYPE]) not in [2, 3]:
    #         ppc["bus"][bus_idx, BUS_TYPE] = 2
    #     ppc["bus"][bus_idx, BUS_TYPE] = 3 if i == ppc["balanced_id"] else 2

    ppc["min_gen_q"] = ppc["gen"][:, QMIN]
    ppc["max_gen_q"] = ppc["gen"][:, QMAX]
    ppc["min_gen_v"] = [0.9 for _ in range(ppc["num_gen"])]
    ppc["max_gen_v"] = [1.1 for _ in range(ppc["num_gen"])]
    ppc["min_bus_v"] = [0.9 for _ in range(ppc["num_bus"])]
    ppc["max_bus_v"] = [1.1 for _ in range(ppc["num_bus"])]

    # overflow parameters
    ppc["soft_overflow_bound"] = 1
    ppc["max_steps_soft_overflow"] = 4
    ppc["hard_overflow_bound"] = 1.35

    # line disconnection parameters
    ppc["prob_disconnection"] = 0.01
    ppc["max_steps_to_reconnect_line"] = 16
    ppc["line_thermal_limit"] = ppc["branch"][:, RATE_A]
    ppc["white_list_random_disconnection"] = [i for i in range(ppc['branch'].shape[0] // 3)]

    # ref/balanced generator
    ppc["min_balanced_gen_bound"] = 0.9
    ppc["max_balanced_gen_bound"] = 1.1

    ppc["ramp_rate"] = 0.06
    ppc["max_steps_to_recover_gen"] = [10 for _ in range(len(a))]
    ppc["max_steps_to_close_gen"] = [10 for _ in range(len(a))]
    ppc["fast_thermal_gen"] = ppc["thermal_ids"]
    ppc["thermal_start_response_steps"] = 5

    # renewable generators
    ppc["renewable_forecast_horizon"] = 20
    ppc["load_forecast_horizon"] = 50

    # eval reward coeffs
    ppc["coeff_line_over_flow"] = 1
    ppc["coeff_renewable_consumption"] = 2
    ppc["coeff_running_cost"] = 1
    ppc["coeff_balanced_gen"] = 4
    ppc["coeff_gen_reactive_power"] = 1
    ppc["coeff_sub_voltage"] = 1

    # others
    ppc["keep_decimal_digits"] = 2
    ppc["env_allow_precision"] = 0.1
    ppc["action_allow_precision"] = 1e-5
    ppc["num_sample"] = 35132
    return ppc
