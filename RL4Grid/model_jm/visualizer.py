import pandas as pd
import numpy as np
import math
import pypsa
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# import cartopy.crs as ccrs
import os
import pypower
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *



class Visualizer:

    def __init__(self, ppc):
        self.ppc = ppc
        self.network = pypsa.Network()
        self.network.import_from_pypower_ppc(self.ppc)
        T_fbus = self.network.transformers.bus0.values.astype(int).tolist()
        T_tbus = self.network.transformers.bus1.values.astype(int).tolist()
        F_bus = self.ppc['branch'][:, F_BUS].astype(int).tolist()
        T_bus = self.ppc['branch'][:, T_BUS].astype(int).tolist()
        self.T_idxs = []
        for t_fbus, t_tbus in zip(T_fbus, T_tbus):
            for i, (fbus, tbus) in enumerate(zip(F_bus, T_bus)):
                if t_fbus == fbus and t_tbus == tbus:
                    self.T_idxs.append(i)
        self.L_idxs = list(set([i for i in range(self.ppc['branch'].shape[0])]) - set(self.T_idxs))

        x, y = [], []
        for i in range(len(self.network.buses)):
            x.append(ppc['coordinates'][i][1])
            y.append(ppc['coordinates'][i][0])
        x = np.array(x)
        y = np.array(y)
        self.network.buses['x'] = x
        self.network.buses['y'] = y

        carriers = []
        gen_type = ppc['gen_type']
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

        self.line_limits = ppc['branch'][:, RATE_A]

    # @profile
    def plot(self, results, save_path, step):
    # def plot(self, gen_p, gen_q, load_p, load_q, gen_status, save_path, step, line_status):
        self.network.generators.p_set = results['gen'][:, PG]
        self.network.generators.q_set = results['gen'][:, QG]
        self.network.loads.p_set = results['bus'][self.ppc['load_bus'], PD]
        self.network.loads.q_set =  results['bus'][self.ppc['load_bus'], QD]
        self.network.status = results['gen'][:, GEN_STATUS]
        line_status = results['branch'][:, BR_STATUS]
        self.network.pf()
        gen = self.network.generators.assign(g=self.network.generators.p_set).groupby(["bus", "carrier"]).g.sum()
        load = self.network.loads.assign(l=self.network.loads.p_set).groupby(["bus", "carrier"]).l.sum()

        # Active Power
        gen = np.clip((results['gen'][:, PG] - results['gen'][:, PMIN]) / (results['gen'][:, PMAX] - results['gen'][:, PMIN] + 1e-3), 0, 1)

        reds = plt.get_cmap('Reds')
        greens = plt.get_cmap('Greens')
        norm = mcolors.Normalize(vmin=0, vmax=1.0)
        bus_colors = pd.Series(index=self.network.buses.index, dtype=object)
        for i, idx in enumerate(self.network.buses.index):
            if i in results['gen_bus']:
                gen_idx = results['gen_bus'].index(i)
                if results['gen_type'][gen_idx] == 1:
                    color = reds(norm(gen[gen_idx]))
                elif results['gen_type'][gen_idx] == 5:
                    color = greens(norm(gen[gen_idx]))
                else:
                    color = '#999999'  # fallback color for unknown types
            else:
                color = '#999999'  # fallback color for unknown types
            bus_colors[idx] = mcolors.to_hex(color)

        bus_sizes = np.zeros(results['num_bus'])
        bus_sizes[results['thermal_ids'] + results['renewable_ids'] + [results['balanced_id']]] = 0.001

        self.network.plot(
            bus_sizes=bus_sizes,
            # margin=0.0001,
            bus_colors=bus_colors,
            # flow='mean',
            line_widths=0.5,
            link_widths=0.5,
            bus_alpha=0.8,
            title=f'{self.ppc["network"]} Active Power Generation, Step={step}',
            geomap=True
        )
        if not os.path.exists(os.path.join(save_path, f'active_power')):
            os.makedirs(os.path.join(save_path, f'active_power'))
        path = os.path.join(save_path, f'active_power/gen_step_{step}.png')
        plt.savefig(path, dpi=400)
        plt.clf()

        # Line Loading
        line_p = np.abs(results['branch'][:, PF])
        line_p = line_p / (np.asarray(self.line_limits)+1e-3)
        line_p = np.clip(line_p + 0.01, 0, 1)
        cmap = plt.get_cmap('coolwarm')
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        line_colors = pd.Series([
            mcolors.to_hex(cmap(norm(val))) for val in line_p[self.L_idxs]
        ], index=self.network.lines.index)
        # line_p = line_p * line_status + (1 - line_status) * np.clip((max(line_p) + 0.1), 0, 1)
        # line_p = line_p * line_status
        collection = self.network.plot(
            bus_sizes=0.001,
            # margin=0.0001,
            # flow='None',
            line_widths=0.5,
            link_widths=0.5,
            line_colors=line_colors,
            bus_alpha=0.8,
            title=f'{self.ppc["network"]} Line Loading, Step={step}',
            geomap=True
        )
        if not os.path.exists(os.path.join(save_path, f'line_loading')):
            os.makedirs(os.path.join(save_path, f'line_loading'))
        path = os.path.join(save_path, f'line_loading/line_step_{step}.png')

        plt.savefig(path, dpi=400)
        plt.clf()


        # LMP
        lmp = results['bus'][:, LAM_P]
        cmap = plt.get_cmap('coolwarm')
        # norm = mcolors.Normalize(vmin=lmp.min(), vmax=lmp.max())
        norm = mcolors.Normalize(vmin=0.00099999997, vmax=0.001000000028)
        bus_colors = pd.Series([
            mcolors.to_hex(cmap(norm(val))) for val in lmp
        ], index=self.network.buses.index)
        self.network.plot(
            bus_sizes=0.001,
            bus_colors=bus_colors,
            line_widths=0.5,
            title=f'{self.ppc["network"]} LMP at Buses, step={step}',
            geomap=True
        )
        if not os.path.exists(os.path.join(save_path, f'lmp')):
            os.makedirs(os.path.join(save_path, f'lmp'))
        path = os.path.join(save_path, f'lmp/lmp_{step}.png')
        plt.savefig(path, dpi=400)
        plt.clf()

        # Reactive Power
        q = self.network.buses_t.q.loc['now']
        q /= np.abs(q).max()
        bus_colors = pd.Series('r', self.network.buses.index)
        bus_colors[q < 0.0] = 'b'
        self.network.plot(
            bus_sizes=abs(q) * 0.01,
            bus_colors=bus_colors,
            line_widths=0.5,
            title=f'{self.ppc["network"]} Reactive Power Feed-in (red=+ve, blue=-ve), Step={step}',
            geomap=True
        )
        if not os.path.exists(os.path.join(save_path, f'reactive_power')):
            os.makedirs(os.path.join(save_path, f'reactive_power'))
        path = os.path.join(save_path, f'reactive_power/reactive_step_{step}.png')
        plt.savefig(path, dpi=400)
        plt.clf()

    def close(self):
        plt.close()



