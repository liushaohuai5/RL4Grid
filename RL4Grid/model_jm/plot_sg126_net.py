# import sys
# sys.path.append('C:/Users/sh-li/Downloads/AdversarialGridZero/')
import pandas as pd
import numpy as np
import math
import pypsa
import matplotlib.pyplot as plt
from pypower.api import case126, case39, case800
# import cartopy.crs as ccrs
import os
from torch_utils import profile


# ppc = case126()
# network = pypsa.Network()
# network.import_from_pypower_ppc(ppc)
# area0 = [i for i in range(32)] + [112, 113, 114, 116, 118, 119, 124, 125]
# area1 = [i for i in range(32, 73)] + [120, 121, 115]
# area2 = [i for i in range(74, 112)] + [123, 122, 117]
# areas = np.zeros_like(network.buses['area'].values)
# areas[area0] = 0
# areas[area1] = 1
# areas[area2] = 2
# coordinates = {
#     '1': [49, 173], '2': [239, 171], '3': [111, 289], '4': [159, 369], '5': [111, 479], '6': [247, 475],
#     '7': [321, 479], '8': [159, 691], '9': [169, 763], '10': [161, 815], '11': [277, 371], '12': [431, 303],
#     '13': [495, 471], '14': [637, 401], '15': [657, 469], '16': [391, 579], '17': [563, 623], '18': [689, 621],
#     '19': [755, 579], '20': [733, 763], '21': [771, 811], '22': [795, 861], '23': [759, 965], '24': [939, 891],
#     '25': [671, 1501], '26': [601, 951], '27': [287, 949], '28': [237, 875], '29': [243, 783], '30': [581, 695],
#     '31': [355, 783], '32': [465, 835], '33': [1119, 339], '34': [1227, 493], '35': [1113, 587], '36': [1279, 589],
#     '37': [1309, 423], '38': [1349, 657], '39': [1263, 323], '40': [1339, 245], '41': [1437, 247], '42': [1517, 245],
#     '43': [1465, 481], '44': [1495, 427], '45': [1531, 559], '46': [1493, 613], '47': [1563, 641], '48': [1627, 527],
#     '49': [1727, 649], '50': [1835, 441], '51': [1891, 399], '52': [1637, 315], '53': [1629, 243], '54': [1737, 247],
#     '55': [2019, 245], '56': [1883, 245], '57': [1843, 349], '58': [1891, 329], '59': [2147, 285], '60': [2147, 409],
#     '61': [2147, 511], '62': [2147, 725], '63': [2105, 347], '64': [2083, 493], '65': [1943, 857], '66': [1933, 729],
#     '67': [1963, 619], '68': [1739, 783], '69': [1625, 785], '70': [1101, 895], '71': [1063, 811], '72': [1015, 965],
#     '73': [1053, 769], '74': [1019, 991], '75': [1077, 1041], '76': [1229, 1043], '77': [1313, 1103], '78': [1349, 1019],
#     '79': [1411, 975], '80': [1439, 1043], '81': [1669, 969], '82': [1271, 1191], '83': [1115, 1225], '84': [1091, 1249],
#     '85': [1083, 1311], '86': [1087, 1377], '87': [1137, 1421], '88': [1197, 1313], '89': [1277, 1313],
#     '90': [1285, 1391], '91': [1435, 1393], '92': [1547, 1317], '93': [1541, 1247], '94': [1589, 1189],
#     '95': [1485, 1189], '96': [1509, 1141], '97': [1513, 1095], '98': [1663, 1109], '99': [1735, 1043],
#     '100': [1737, 1191], '101': [1691, 1367], '102': [1531, 1367], '103': [1845, 1367], '104': [1873, 1191],
#     '105': [1991, 1187], '106': [1951, 1085], '107': [2151, 1191], '108': [2009, 1259], '109': [2015, 1313],
#     '110': [1985, 1405], '111': [1917, 1427], '112': [2107, 1403], '113': [471, 663], '114': [427, 879],
#     '115': [493, 883], '116': [1793, 813], '117': [615, 305], '118': [1191, 1041], '119': [51, 95],
#     '120': [243, 1047], '121': [1715, 151], '122': [2273, 259], '123': [2225, 1105], '124': [2109, 1349],
#     '125': [359, 217], '126': [443, 1027]
# }
# # network.buses['area'] = areas
# x, y = [], []
# for i in range(len(network.buses)):
#     x.append(coordinates[str(i+1)][0])
#     y.append(coordinates[str(i+1)][1])
# x = np.array(x)
# y = np.array(y)
# network.buses['x'] = x
# network.buses['y'] = y
#
# carriers = []
# gen_type = [5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 1, 5, 1, 1, 1, 1,
#     1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 5]
# for i in range(len(network.generators)):
#     if gen_type[i] == 1:
#         carriers.append('thermal')
#     elif gen_type[i] == 5:
#         carriers.append('renewable')
#     else:
#         carriers.append('thermal')
# carriers = np.array(carriers)
# network.generators['carrier'] = carriers
#
# load_carriers = []
# for i in range(len(network.loads)):
#     load_carriers.append('load')
# load_carriers = np.array(load_carriers)
# network.loads['carrier'] = load_carriers
#
# network.pf()
#
# # network.generators.carrier[17] = 'closed'
# # network.generators.p_set[17] = 200
# gen = network.generators.assign(g=network.generators.p_set).groupby(["bus", "carrier"]).g.sum()
# load = network.loads.assign(l=network.loads.p_set).groupby(["bus", "carrier"]).l.sum()
# import ipdb
# ipdb.set_trace()
# # flow = pd.Series(10, index=network.branches().index)
# img_b = plt.imread('sg126_map.png')
#
# # Active Power
# # network.plot(
# #     bus_sizes=gen * 5, margin=0.0001,
# #     bus_colors={'renewable': 'green', 'thermal': 'red'},
# #     flow='mean',
# #     line_widths=5, link_widths=0,
# #     title=f'Active Power Generation, Step={0}'
# # )
# # plt.savefig('1.png', dpi=400)
# # plt.close()
# # img_a = plt.imread('1.png')
# # img_a = (img_a * img_b)
# # plt.imshow(img_a)
# # plt.axis('off')
# # plt.show()
# # plt.close()
#
# line_limits = [313.92, 472.14, 899.01, 547.54, 593.39, 510.21, 340.29, 332.76,
#     421.88, 504.58, 598.21, 445.0, 154.59, 327.68, 426.04, 338.65, 327.49, 374.02,
#     370.94, 1120.62, 560.22, 602.95, 534.57, 490.72, 370.68, 346.75, 474.19, 670.54,
#     1595.64, 458.65, 546.74, 660.22, 235.09, 2126.18, 1081.66, 578.59, 558.97, 677.71,
#     393.76, 329.76, 515.94, 407.59, 592.89, 565.51, 609.72, 707.28, 1825.1, 1884.33,
#     2000.32, 5042.53, 2348.87, 982.76, 925.44, 900.04, 297.88, 590.58, 699.74, 816.95,
#     717.04, 425.57, 683.69, 720.87, 655.34, 648.31, 839.28, 439.4, 404.84, 241.6,
#     354.29, 540.04, 505.36, 413.21, 258.2, 984.56, 795.51, 810.9, 529.76, 820.43,
#     539.92, 303.62, 278.99, 243.14, 356.43, 354.18, 353.73, 284.72, 210.79, 335.84,
#     1019.7, 6365.77, 2085.32, 781.7, 651.42, 382.77, 465.75, 508.24, 4382.12, 534.4,
#     430.05, 725.12, 538.49, 1477.84, 1012.15, 522.97, 1026.97, 473.95, 449.73, 413.86,
#     466.4, 454.41, 715.01, 337.2, 465.05, 348.73, 380.07, 148.98, 299.2, 3357.97,
#     1295.32, 1289.9, 647.34, 972.3, 784.19, 436.1, 443.27, 680.91, 734.17, 795.4,
#     353.95, 561.84, 609.4, 1212.41, 321.33, 937.52, 842.59, 733.43, 639.9, 1218.54,
#     760.96, 526.56, 1184.53, 679.77, 949.06, 1147.56, 413.43, 1558.73, 948.07, 963.0,
#     1230.32, 1137.6, 371.19, 659.98, 480.15, 1871.8, 656.35, 296.3, 351.13, 535.05,
#     683.05, 437.39, 444.37, 851.91, 424.47, 659.23, 745.1, 885.21, 421.04, 1651.13,
#     1098.42, 442.49, 278.21, 273.69, 225.8, 365.48, 498.46, 665.39, 876.37, 532.05,
#     660.22, 892.74, 778.44, 871.3, 1651.13, 445.0, 546.74]
# for i in range(len(network.lines)-len(line_limits)):
#     line_limits.append(800)
#
# # Line Loading
# import ipdb
# ipdb.set_trace()
# collection = network.plot(
#     bus_sizes=10, margin=0.0001,
#     flow="mean",
#     line_widths=5,
#     link_widths=0,
#     line_colors=network.lines_t.p0.mean().abs()/np.array(line_limits).clip(0, 1),
#     # bus_alpha=(step * 0.1) % 0.5 + 0.5
#     bus_alpha=0.8,
#     title=f'Line Loading, Step={0}'
# )
# plt.savefig('1.png', dpi=400)
# plt.close()
# img_a = plt.imread('1.png')
# img_a = (img_a * img_b)
# obj = plt.imshow(img_a)
# colbar = plt.colorbar(obj, fraction=0.04, pad=0.004, label="Line Loading Rate")
# obj.set_clim(0, 1)
# plt.axis('off')
# plt.show()
# plt.close()
#
# # Reactive Power
# q = network.buses_t.q.loc['now']
# bus_colors = pd.Series('r', network.buses.index)
# bus_colors[q < 0.0] = 'b'
# network.plot(
#     bus_sizes=5*abs(q),
#     bus_colors=bus_colors,
#     title=f'Reactive Power Feed-in (red=+ve, blue=-ve), Step={0}'
# )
# plt.savefig('1.png', dpi=400)
# plt.close()
# img_a = plt.imread('1.png')
# img_a = (img_a * img_b)
# plt.imshow(img_a)
# plt.axis('off')
# plt.show()
# plt.close()
#
# collection = network.plot(
#     bus_sizes=load.append(gen)*5, margin=0.0001, bus_colors={'renewable': 'green', 'thermal': 'red', 'load': 'orange'},
#     flow="mean",
#     line_widths=5,
#     link_widths=0,
#     line_colors=network.lines_t.p0.mean().abs()
# )
# plt.colorbar(collection[2], fraction=0.04, pad=0.004, label="Flow in MW")
# plt.savefig('1.png', dpi=400)
# plt.close()
# img_a = plt.imread('1.png')
# img_b = plt.imread('sg126_map.png')
# img_a = (img_a*img_b).clip(0, 255)
# plt.imshow(img_a)
# plt.axis('off')
# plt.show()
# plt.close()
# # plt.savefig('1.png', dpi=400)
# import ipdb
# ipdb.set_trace()
# print('finish')


class Visualizer:

    def __init__(self):
        self.ppc = case126()
        self.network = pypsa.Network()
        self.network.import_from_pypower_ppc(self.ppc)

        area0 = [i for i in range(32)] + [112, 113, 114, 116, 118, 119, 124, 125]
        area1 = [i for i in range(32, 73)] + [120, 121, 115]
        area2 = [i for i in range(74, 112)] + [123, 122, 117]
        areas = np.zeros_like(self.network.buses['area'].values)
        areas[area0] = 0
        areas[area1] = 1
        areas[area2] = 2
        self.network.buses['area'] = areas
        self.map = plt.imread('/workspace/AdversarialGridZero/model_jm/sg126_map.png')

        coordinates = {
            '1': [49, 173], '2': [239, 171], '3': [111, 289], '4': [159, 369], '5': [111, 479], '6': [247, 475],
            '7': [321, 479], '8': [159, 691], '9': [169, 763], '10': [161, 815], '11': [277, 371], '12': [431, 303],
            '13': [495, 471], '14': [637, 401], '15': [657, 469], '16': [391, 579], '17': [563, 623], '18': [689, 621],
            '19': [755, 579], '20': [733, 763], '21': [771, 811], '22': [795, 861], '23': [759, 965], '24': [939, 891],
            '25': [671, 1501], '26': [601, 951], '27': [287, 949], '28': [237, 875], '29': [243, 783], '30': [581, 695],
            '31': [355, 783], '32': [465, 835], '33': [1119, 339], '34': [1227, 493], '35': [1113, 587],
            '36': [1279, 589],
            '37': [1309, 423], '38': [1349, 657], '39': [1263, 323], '40': [1339, 245], '41': [1437, 247],
            '42': [1517, 245],
            '43': [1465, 481], '44': [1495, 427], '45': [1531, 559], '46': [1493, 613], '47': [1563, 641],
            '48': [1627, 527],
            '49': [1727, 649], '50': [1835, 441], '51': [1891, 399], '52': [1637, 315], '53': [1629, 243],
            '54': [1737, 247],
            '55': [2019, 245], '56': [1883, 245], '57': [1843, 349], '58': [1891, 329], '59': [2147, 285],
            '60': [2147, 409],
            '61': [2147, 511], '62': [2147, 725], '63': [2105, 347], '64': [2083, 493], '65': [1943, 857],
            '66': [1933, 729],
            '67': [1963, 619], '68': [1739, 783], '69': [1625, 785], '70': [1101, 895], '71': [1063, 811],
            '72': [1015, 965],
            '73': [1053, 769], '74': [1019, 991], '75': [1077, 1041], '76': [1229, 1043], '77': [1313, 1103],
            '78': [1349, 1019],
            '79': [1411, 975], '80': [1439, 1043], '81': [1669, 969], '82': [1271, 1191], '83': [1115, 1225],
            '84': [1091, 1249],
            '85': [1083, 1311], '86': [1087, 1377], '87': [1137, 1421], '88': [1197, 1313], '89': [1277, 1313],
            '90': [1285, 1391], '91': [1435, 1393], '92': [1547, 1317], '93': [1541, 1247], '94': [1589, 1189],
            '95': [1485, 1189], '96': [1509, 1141], '97': [1513, 1095], '98': [1663, 1109], '99': [1735, 1043],
            '100': [1737, 1191], '101': [1691, 1367], '102': [1531, 1367], '103': [1845, 1367], '104': [1873, 1191],
            '105': [1991, 1187], '106': [1951, 1085], '107': [2151, 1191], '108': [2009, 1259], '109': [2015, 1313],
            '110': [1985, 1405], '111': [1917, 1427], '112': [2107, 1403], '113': [471, 663], '114': [427, 879],
            '115': [493, 883], '116': [1793, 813], '117': [615, 305], '118': [1191, 1041], '119': [51, 95],
            '120': [243, 1047], '121': [1715, 151], '122': [2273, 259], '123': [2225, 1105], '124': [2109, 1349],
            '125': [359, 217], '126': [443, 1027]
        }
        x, y = [], []
        for i in range(len(self.network.buses)):
            x.append(coordinates[str(i + 1)][0])
            y.append(coordinates[str(i + 1)][1])
        x = np.array(x)
        y = np.array(y)
        self.network.buses['x'] = x
        self.network.buses['y'] = y

        carriers = []
        gen_type = [5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 1, 5, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 5]
        for i in range(len(self.network.generators)):
            if gen_type[i] == 1:
                carriers.append('thermal')
            elif gen_type[i] == 5:
                carriers.append('renewable')
            else:
                carriers.append('thermal')
        carriers = np.array(carriers)
        self.network.generators['carrier'] = carriers

        load_carriers = []
        for i in range(len(self.network.loads)):
            load_carriers.append('load')
        load_carriers = np.array(load_carriers)
        self.network.loads['carrier'] = load_carriers

        self.line_limits = [313.92, 472.14, 899.01, 547.54, 593.39, 510.21, 340.29, 332.76,
                       421.88, 504.58, 598.21, 445.0, 154.59, 327.68, 426.04, 338.65, 327.49, 374.02,
                       370.94, 1120.62, 560.22, 602.95, 534.57, 490.72, 370.68, 346.75, 474.19, 670.54,
                       1595.64, 458.65, 546.74, 660.22, 235.09, 2126.18, 1081.66, 578.59, 558.97, 677.71,
                       393.76, 329.76, 515.94, 407.59, 592.89, 565.51, 609.72, 707.28, 1825.1, 1884.33,
                       2000.32, 5042.53, 2348.87, 982.76, 925.44, 900.04, 297.88, 590.58, 699.74, 816.95,
                       717.04, 425.57, 683.69, 720.87, 655.34, 648.31, 839.28, 439.4, 404.84, 241.6,
                       354.29, 540.04, 505.36, 413.21, 258.2, 984.56, 795.51, 810.9, 529.76, 820.43,
                       539.92, 303.62, 278.99, 243.14, 356.43, 354.18, 353.73, 284.72, 210.79, 335.84,
                       1019.7, 6365.77, 2085.32, 781.7, 651.42, 382.77, 465.75, 508.24, 4382.12, 534.4,
                       430.05, 725.12, 538.49, 1477.84, 1012.15, 522.97, 1026.97, 473.95, 449.73, 413.86,
                       466.4, 454.41, 715.01, 337.2, 465.05, 348.73, 380.07, 148.98, 299.2, 3357.97,
                       1295.32, 1289.9, 647.34, 972.3, 784.19, 436.1, 443.27, 680.91, 734.17, 795.4,
                       353.95, 561.84, 609.4, 1212.41, 321.33, 937.52, 842.59, 733.43, 639.9, 1218.54,
                       760.96, 526.56, 1184.53, 679.77, 949.06, 1147.56, 413.43, 1558.73, 948.07, 963.0,
                       1230.32, 1137.6, 371.19, 659.98, 480.15, 1871.8, 656.35, 296.3, 351.13, 535.05,
                       683.05, 437.39, 444.37, 851.91, 424.47, 659.23, 745.1, 885.21, 421.04, 1651.13,
                       1098.42, 442.49, 278.21, 273.69, 225.8, 365.48, 498.46, 665.39, 876.37, 532.05,
                       660.22, 892.74, 778.44, 871.3, 1651.13, 445.0, 546.74]
        for i in range(len(self.network.lines) - len(self.line_limits)):
            self.line_limits.append(800)

    # @profile
    def plot(self, gen_p, gen_q, load_p, load_q, gen_status, save_path, step, line_status):
        self.network.generators.p_set = gen_p
        self.network.generators.q_set = gen_q
        self.network.loads.p_set = load_p
        self.network.loads.q_set = load_q
        self.network.status = gen_status
        self.network.pf()
        gen = self.network.generators.assign(g=self.network.generators.p_set).groupby(["bus", "carrier"]).g.sum()
        load = self.network.loads.assign(l=self.network.loads.p_set).groupby(["bus", "carrier"]).l.sum()

        # Active Power
        self.network.plot(
            bus_sizes=gen * 5, margin=0.0001,
            bus_colors={'renewable': 'green', 'thermal': 'red'},
            flow='mean',
            line_widths=5, link_widths=0,
            title=f'Active Power Generation, Step={step}'
        )
        if not os.path.exists(os.path.join(save_path, 'active_power')):
            os.makedirs(os.path.join(save_path, 'active_power'))
        path = os.path.join(save_path, f'active_power/gen_step_{step}.png')
        plt.savefig(path, dpi=400)
        plt.clf()
        # img_a = plt.imread(path)
        # img_a = (img_a * self.map)
        # plt.imshow(img_a)
        # plt.axis('off')
        # plt.savefig(path, dpi=400)
        # plt.clf()

        # Line Loading
        line_status = line_status + [1 for _ in range(len(self.line_limits) - len(line_status))]
        line_status = np.asarray(line_status)
        # line_status[0] = 0
        line_p = (self.network.lines_t.p0.mean().abs()) #+ 1e6 * (1 - line_status)
        # line_p = np.log(line_p) * line_status
        line_p = line_p / np.asarray(self.line_limits)
        # line_p = line_p * line_status
        collection = self.network.plot(
            bus_sizes=10, margin=0.0001,
            flow='mean',
            line_widths=3,
            link_widths=0,
            # jitter=0.1,
            # line_colors=self.network.lines_t.p0.mean().abs()/np.array(self.line_limits).clip(0, 1),
            line_colors=line_p * line_status + (1 - line_status) * np.clip((max(line_p) + 0.1), 0, 1),
            # link_colors=line_p,
            # bus_alpha=(step * 0.1) % 0.5 + 0.5
            bus_alpha=0.8,
            title=f'Line Loading, Step={step}'
        )
        if not os.path.exists(os.path.join(save_path, 'line_loading')):
            os.makedirs(os.path.join(save_path, 'line_loading'))
        path = os.path.join(save_path, f'line_loading/line_step_{step}.png')

        # collection = self.network.plot(
        #     bus_sizes=10, margin=0.0001,
        #     flow='mean',
        #     line_widths=5,
        #     link_widths=0,
        #     # jitter=0.1,
        #     line_alpha=0.2,
        #     # line_colors='red',
        #     line_colors=np.ones_like(line_p) * (1 - line_status),
        #     # link_colors=line_p,
        #     # bus_alpha=(step * 0.1) % 0.5 + 0.5
        #     bus_alpha=0.8,
        #     title=f'Line Loading, Step={step}'
        # )
        # import ipdb
        # ipdb.set_trace()
        # plt.show()
        plt.savefig(path, dpi=400)
        plt.clf()
        # img_a = plt.imread(path)
        # img_a = (img_a * self.map)
        # obj = plt.imshow(img_a)
        # colbar = plt.colorbar(obj, fraction=0.04, pad=0.004, label="Line Loading Rate")
        # obj.set_clim(0, 1)
        # plt.axis('off')
        # plt.savefig(path, dpi=400)
        # plt.clf()

        # Reactive Power
        q = self.network.buses_t.q.loc['now']
        bus_colors = pd.Series('r', self.network.buses.index)
        bus_colors[q < 0.0] = 'b'
        self.network.plot(
            bus_sizes=5*abs(q),
            bus_colors=bus_colors,
            title=f'Reactive Power Feed-in (red=+ve, blue=-ve), Step={step}'
        )
        if not os.path.exists(os.path.join(save_path, 'reactive_power')):
            os.makedirs(os.path.join(save_path, 'reactive_power'))
        path = os.path.join(save_path, f'reactive_power/reactive_step_{step}.png')
        plt.savefig(path, dpi=400)
        plt.clf()
        # img_a = plt.imread(path)
        # img_a = (img_a * self.map)
        # plt.imshow(img_a)
        # plt.axis('off')
        # plt.savefig(path, dpi=400)
        # plt.clf()

    def close(self):
        plt.close()



