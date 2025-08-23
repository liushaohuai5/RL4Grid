import numpy as np
import chardet
import pandas as pd
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
import re
import geopy


train_idx = 4*24*365*3
test_idx = 4*24*365*4
load = pd.read_csv('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/load.csv')
norm_load = (load.values - load.values.min(0)) / (load.values.max(0) - load.values.min(0) + 1e-3)
np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/train/load.npy', norm_load[:train_idx])
np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/test/load.npy', norm_load[train_idx:test_idx])
solar = pd.read_csv('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/solar.csv')
norm_solar = (solar.values - solar.values.min(0)) / (solar.values.max(0) - solar.values.min(0) + 1e-3)
np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/train/solar.npy', norm_solar[:train_idx])
np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/test/solar.npy', norm_solar[train_idx:test_idx])
wind = pd.read_csv('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/wind.csv')
norm_wind = (wind.values - wind.values.min(0)) / (wind.values.max(0) - wind.values.min(0) + 1e-3)
np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/train/wind.npy', norm_wind[:train_idx])
np.save('C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/data/test/wind.npy', norm_wind[train_idx:test_idx])
import ipdb
ipdb.set_trace()

def extract_floats(s):
    # pattern = r"-?\d+(?:\.\d+)?"
    pattern = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    float_numbers = re.findall(pattern, s)
    float_numbers = [float(num) for num in float_numbers]
    return np.asarray(float_numbers)

bus_arrays = []
gen_arrays = []
branch_arrays = []
gencost_arrays = []

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

save_path = "C:/Users/sh-li/Downloads/RL4Grid/RL4Grid/new_model_jm/"

from pathlib import Path
folder = Path("C:/Users/sh-li/Downloads/pglib-opf-master/")
paths = sorted(str(p.resolve()) for p in folder.glob("*.m"))

cnt = 0
for path in paths:
    network = path.split('pglib_opf_')[1].split('.m')[0]
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
    import ipdb
    ipdb.set_trace()
    bus_array = np.asarray(bus_array)
    bus_arrays.append(bus_array)
    np.save(f"{save_path}{network}_bus.npy", bus_array)
    gen_array = np.asarray(gen_array)
    gen_arrays.append(gen_array)
    np.save(f"{save_path}{network}_gen.npy", gen_array)
    branch_array = np.asarray(branch_array)
    branch_arrays.append(branch_array)
    np.save(f"{save_path}{network}_branch.npy", branch_array)
    gencost_array = np.asarray(gencost_array)
    gencost_arrays.append(gencost_array)
    gencost_array[:, NCOST] = gencost_array[:, NCOST].clip(1, 3)
    np.save(f"{save_path}{network}_gencost.npy", gencost_array)
    np.save(f"{save_path}{network}_gen_type.npy", np.asarray(gen_type))
    # np.save(f"{save_path}{network}_coordinates.npy", np.asarray(coordinates))

import ipdb
ipdb.set_trace()