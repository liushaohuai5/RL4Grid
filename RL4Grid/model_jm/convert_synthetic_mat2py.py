import os.path

import numpy as np
import chardet
import pandas as pd
import pypower
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
import re
from pathlib import Path
from datetime import date
import tempfile

# def upsert_tail_block(file_path, tag, lines, encoding="utf-8"):
#     p = Path(file_path)
#     if not p.exists():
#         raise FileNotFoundError(p)
#
#     start_marker = f"# <<< {tag} START >>>\n"
#     end_marker   = f"# <<< {tag} END >>>\n"
#
#     text = p.read_text(encoding=encoding)
#
#     if not text.endswith("\n"):
#         text += "\n"
#
#     block_body = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
#     block = f"{start_marker}{block_body}{end_marker}"
#
#     s = text.rfind(start_marker)
#     e = text.rfind(end_marker)
#     if s != -1 and e != -1 and s < e:
#         new_text = text[:s] + block + text[e + len(end_marker):]
#     else:
#         if not text.endswith("\n\n"):
#             text += "\n"
#         new_text = text + block
#
#     with tempfile.NamedTemporaryFile("w", encoding=encoding, newline="", delete=False) as tmp:
#         tmp.write(new_text)
#         tmp_path = tmp.name
#     os.replace(tmp_path, p)
#
# pypower_path = Path(os.path.dirname(pypower.__file__))
# target = pypower_path / "api.py"
# new_lines = []
#
# folder = Path(os.path.dirname(os.getcwd())+'/pglib-opf/')
# paths = sorted(str(p.resolve()) for p in folder.glob("*.py"))
#
# cnt = 0
# for path in paths:
#     network = path.split('/')[-1].split('.py')[0]
#     new_lines.append(f"from .{network} import {network}")
#
# upsert_tail_block(
#     target,
#     tag="AUTOAPPEND",
#     lines=new_lines
# )
# import ipdb
# ipdb.set_trace()

# train_idx = 4*24*365*3
# test_idx = 4*24*365*4
# load = pd.read_csv('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/load.csv')
# norm_load = (load.values - load.values.min(0)) / (load.values.max(0) - load.values.min(0) + 1e-3) * 0.9 + 0.3   # 0.3 - 1.2
# np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/train/load.npy', norm_load[:train_idx])
# np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/test/load.npy', norm_load[train_idx:test_idx])
# solar = pd.read_csv('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/solar.csv')
# norm_solar = (solar.values - solar.values.min(0)) / (solar.values.max(0) - solar.values.min(0) + 1e-3) * 1.4 + 0.1  # 0.1 - 1.5
# np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/train/solar.npy', norm_solar[:train_idx])
# np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/test/solar.npy', norm_solar[train_idx:test_idx])
# wind = pd.read_csv('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/wind.csv')
# norm_wind = (wind.values - wind.values.min(0)) / (wind.values.max(0) - wind.values.min(0) + 1e-3) * 1.4 + 0.1   # 0.1 - 1.5
# np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/train/wind.npy', norm_wind[:train_idx])
# np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/test/wind.npy', norm_wind[train_idx:test_idx])
# import ipdb
# ipdb.set_trace()

def extract_floats(s):
    # pattern = r"-?\d+(?:\.\d+)?"
    pattern = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    float_numbers = re.findall(pattern, s)
    float_numbers = [float(num) for num in float_numbers]
    return np.asarray(float_numbers)

# bus_arrays = []
# gen_arrays = []
# branch_arrays = []
# gencost_arrays = []

network = 'TX2000'   # TX2000 or WE10000
paths = [
    # "C:/Users/sh-li/Downloads/case_ACTIVSg2000 (2).m",
    "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case1_2016summerpeak/Texas2k_series24_case1_2016summerPeak.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case2_2016lowload/Texas2k_series24_case2_2016lowload.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case3_2024summerpeak/Texas2k_series24_case3_2024summerpeak.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case4_2024lowload/Texas2k_series24_case4_2024lowload.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case5_2024highrenewables/Texas2k_series24_case5_2024highrenewables.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case6_2024lowloadwithgfm/Texas2k_series24_case6_2024lowloadwithgfm.m",
]
coor_path = 'C:/Users/sh-li/Downloads/SubstationCoordinates_2000.csv'

# network = 'WE10000'
# paths = [
#     "C:/Users/sh-li/Downloads/ACTIVSg10k/case_ACTIVSg10k.m"
# ]
# coor_path = 'C:/Users/sh-li/Downloads/SubstationCoordinates_10k.csv'


coor_data = pd.read_csv(coor_path)
sub_names = coor_data['Sub Name'].values.tolist()
longitudes = coor_data['Longitude'].values.tolist()
latitudes = coor_data['Latitude'].values.tolist()
coordinates = []

save_path = os.path.dirname(os.getcwd()) + '/pglib-opf/'

folder = Path("C:/Users/sh-li/Downloads/pglib-opf-master/")
paths = sorted(str(p.resolve()) for p in folder.glob("*.m"))

cnt = 0
for path in paths:
    network = path.split('pglib_opf_')[1].split('.m')[0]
    print(network)
    TEMPLATE = f'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network: {network}
Auto-generated on: {date.today()}
"""

import numpy as np
import os
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
path = os.path.dirname(__file__) + "/"

def {network}():

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc = dict(version=2)
    
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = np.load(path+"{network}_bus.npy")

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = np.load(path+"{network}_gen.npy")


    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = np.load(path+"{network}_branch.npy")

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = np.load(path+"{network}_gencost.npy")

    ppc["network"] = {network}
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
                    ppc["gen"][g1, [PG, QG, QMAX, QMIN, PMAX, PMIN, PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC2MAX, RAMP_AGC, RAMP_10,
                                    RAMP_30, RAMP_Q]] \
                        += ppc["gen"][
                        g2, [PG, QG, QMAX, QMIN, PMAX, PMIN, PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC2MAX, RAMP_AGC, RAMP_10,
                             RAMP_30, RAMP_Q]]
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
'''

    outdir = Path(save_path)
    (outdir / f"{network}.py").write_text(TEMPLATE, encoding="utf-8")

    with open(path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print("Detected encoding:", encoding)

    with open(path, "r+", encoding=encoding) as file:
        line = file.readline()
        bus_array = []
        gen_array = []
        branch_array = []
        gencost_array = []
        gen_type = []
        while line:
            if 'mpc.bus =' in line:
                line = file.readline()
                while '];' not in line:
                    floats = extract_floats(line)
                    bus_array.append(floats)
                    line = file.readline()
            elif 'mpc.gen =' in line:
                line = file.readline()
                while '];' not in line:
                    floats = extract_floats(line)
                    gen_array.append(floats)
                    line = file.readline()
            elif 'mpc.branch =' in line:
                line = file.readline()
                while '];' not in line:
                    floats = extract_floats(line)
                    if len(branch_array) > 0:
                        if len(floats) != len(branch_array[-1]):
                            import ipdb
                            ipdb.set_trace()
                    branch_array.append(floats)
                    line = file.readline()
            elif 'mpc.gencost =' in line:
                line = file.readline()
                while '];' not in line:
                    floats = extract_floats(line)
                    gencost_array.append(floats)
                    line = file.readline()
            elif 'mpc.genfuel =' in line:
                line = file.readline()
                while '};' not in line:
                    try:
                        type = line.split("'")[1]
                    except:
                        import ipdb
                        ipdb.set_trace()
                    if type in ['wind', 'solar']:
                        gen_type.append(5)
                    elif type in ['ng', 'hydro', 'coal', 'nuclear']:
                        gen_type.append(1)
                    else:
                        gen_type.append(5)
                    line = file.readline()
            elif 'mpc.bus_name =' in line:
                line = file.readline()
                while '};' not in line:
                    name = re.findall(r"'([A-Za-z0-9\.\- ]+?)(?:\s\d+(?:\s\d+)*)?'\s*;", line)[0]
                    offsets = extract_floats(line).tolist()
                    coor_name = f'{name} {int(offsets[0])}' if len(offsets) == 2 else f'{name}'
                    for i, sub_name in enumerate(sub_names):
                        if sub_name == coor_name:
                            coordinates.append([latitudes[i], longitudes[i]])
                            break
                        if i == len(sub_names) - 1:
                            import ipdb
                            ipdb.set_trace()
                            print('no match')
                    cnt += 1
                    line = file.readline()
            else:
                line = file.readline()

    bus_array = np.asarray(bus_array)
    # if network == 'case240_pserc':
    #     import ipdb
    #     ipdb.set_trace()
    if (bus_array[:, BUS_TYPE] == 3).sum() == 0:
        import ipdb
        ipdb.set_trace()
        max_gen_idx = bus_array[:, PMAX].argmax()
        bus_array[max_gen_idx, BUS_TYPE] = 3
    # bus_arrays.append(bus_array)
    np.save(f"{save_path}{network}_bus.npy", bus_array)
    gen_array = np.asarray(gen_array)
    # gen_arrays.append(gen_array)
    np.save(f"{save_path}{network}_gen.npy", gen_array)
    branch_array = np.asarray(branch_array)
    # branch_arrays.append(branch_array)
    np.save(f"{save_path}{network}_branch.npy", branch_array)
    gencost_array = np.asarray(gencost_array)
    # gencost_arrays.append(gencost_array)
    gencost_array[:, NCOST] = gencost_array[:, NCOST].clip(1, 3)
    np.save(f"{save_path}{network}_gencost.npy", gencost_array)
    np.save(f"{save_path}{network}_gen_type.npy", np.asarray(gen_type))
    # np.save(f"{save_path}{network}_coordinates.npy", np.asarray(coordinates))

import ipdb
ipdb.set_trace()