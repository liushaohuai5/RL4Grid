import numpy as np
import chardet
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
import re


def extract_floats(s):
    pattern = r"-?\d+(?:\.\d+)?"
    float_numbers = re.findall(pattern, s)
    float_numbers = [float(num) for num in float_numbers]
    return np.asarray(float_numbers)

bus_arrays = []
gen_arrays = []
branch_arrays = []
gencost_arrays = []

paths = [
    # "C:/Users/sh-li/Downloads/case_ACTIVSg2000 (2).m",
    "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case1_2016summerpeak/Texas2k_series24_case1_2016summerPeak.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case2_2016lowload/Texas2k_series24_case2_2016lowload.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case3_2024summerpeak/Texas2k_series24_case3_2024summerpeak.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case4_2024lowload/Texas2k_series24_case4_2024lowload.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case5_2024highrenewables/Texas2k_series24_case5_2024highrenewables.m",
    # "C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case6_2024lowloadwithgfm/Texas2k_series24_case6_2024lowloadwithgfm.m",
]

for path in paths:

    with open(path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print("Detected encoding:", encoding)

    with open(path, "r+", encoding=encoding) as file:
    # with open("C:/Users/sh-li/Downloads/Texas2k_series24_cases_with_dynamics/Texas2k_series24_case3_2024summerpeak/Texas2k_series24_case3_2024summerpeak.m", "r", encoding="utf-8") as file:
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
                # import ipdb
                # ipdb.set_trace()
                line = file.readline()
                while '];' not in line:
                    floats = extract_floats(line)
                    gen_array.append(floats)
                    line = file.readline()
            elif 'mpc.branch =' in line:
                line = file.readline()
                while '];' not in line:
                    floats = extract_floats(line)
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
                while '];' not in line:
                    if type in ['wind', 'solar']:
                        gen_type.append(5)
                    if type in ['ng', 'hydro', 'coal', 'nuclear']:
                        gen_type.append(1)
                    line = file.readline()
            else:
                line = file.readline()
    # import ipdb
    # ipdb.set_trace()
    bus_array = np.asarray(bus_array)
    bus_arrays.append(bus_array)
    np.save("C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/model_jm/TX2000_bus.npy", bus_array)
    gen_array = np.asarray(gen_array)
    gen_arrays.append(gen_array)
    np.save("C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/model_jm/TX2000_gen.npy", gen_array)
    # branches = []
    # branch_array_no_redundant = []
    # for branch in branch_array:
    #     f_bus = branch[F_BUS]
    #     t_bus = branch[T_BUS]
    #     if (f_bus, t_bus) in branches or (t_bus, f_bus) in branches:
    #         continue
    #     branches.append((f_bus, t_bus))
    #     branch_array_no_redundant.append(branch)
    # import ipdb
    # ipdb.set_trace()
    branch_array = np.asarray(branch_array)
    branch_arrays.append(branch_array)
    np.save("C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/model_jm/TX2000_branch.npy", branch_array)
    gencost_array = np.asarray(gencost_array)
    gencost_arrays.append(gencost_array)
    gencost_array[:, NCOST] = gencost_array[:, NCOST].clip(1, 3)
    np.save("C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/model_jm/TX2000_gencost.npy", gencost_array)
    np.save("C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/model_jm/TX2000_gencost.npy")


import ipdb
ipdb.set_trace()