import numpy as np

with open("C:/Users/sh-li/Downloads/case_ACTIVSg2000.m", "r", encoding="utf-8") as file:
    line = file.readline()
    bus_array = []
    gen_array = []
    branch_array = []
    gencost_array = []
    while line:
        if 'mpc.bus =' in line:
            import ipdb
            ipdb.set_trace()
            line = file.readline()
            while '];' not in line:
                floats = line.split(';\n')[0].split('\t')[1:]
                floats = [float(f) for f in floats]
                bus_array.append(floats)
                line = file.readline()
        elif 'mpc.gen =' in line:
            import ipdb
            ipdb.set_trace()
            line = file.readline()
            while '];' not in line:
                floats = line.split(';\n')[0].split('\t')[1:]
                floats = [float(f) for f in floats]
                gen_array.append(floats)
                line = file.readline()
        elif 'mpc.branch =' in line:
            import ipdb
            ipdb.set_trace()
            line = file.readline()
            while '];' not in line:
                floats = line.split(';\n')[0].split('\t')[1:]
                floats = [float(f) for f in floats]
                branch_array.append(floats)
                line = file.readline()
        elif 'mpc.gencost =' in line:
            line = file.readline()
            while '];' not in line:
                floats = line.split(';\n')[0].split('\t')[1:]
                floats = [float(f) for f in floats]
                gencost_array.append(floats)
                line = file.readline()
        else:
            line = file.readline()
    import ipdb
    ipdb.set_trace()
    bus_array = np.asarray(bus_array)
    np.save("C:/Users/sh-li/Downloads/AdversarialGridZero/model_jm/TX2000_bus.npy", bus_array)
    gen_array = np.asarray(gen_array)
    np.save("C:/Users/sh-li/Downloads/AdversarialGridZero/model_jm/TX2000_gen.npy", gen_array)
    branch_array = np.asarray(branch_array)
    np.save("C:/Users/sh-li/Downloads/AdversarialGridZero/model_jm/TX2000_branch.npy", branch_array)
    gencost_array = np.asarray(gencost_array)
    np.save("C:/Users/sh-li/Downloads/AdversarialGridZero/model_jm/TX2000_gencost.npy", gencost_array)