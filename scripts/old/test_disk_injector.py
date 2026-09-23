import numpy as np
from datasets.transforms.disk_simulator import DiskSimulator, DiskInjector

def run_test():
    sim = DiskSimulator(sim_params={'r_frac':0.25,'width_frac':0.03,'amp':1.0})
    injector = DiskInjector(simulator=sim, scale=1.0)

    C, T, H, W = 2, 4, 64, 64
    frame = np.zeros((C, T, H, W), dtype=np.float32)
    sample = {'frame': frame.copy(), 'rot': np.zeros(T)}

    out = injector(sample)
    y = out.get('y', None)
    if y is None:
        y = out.get('frame', None)
    print('output keys:', list(out.keys()))
    print('frame shape:', None if y is None else y.shape)
    print('max value after injection:', None if y is None else float(np.max(np.asarray(y))))

if __name__ == '__main__':
    run_test()
