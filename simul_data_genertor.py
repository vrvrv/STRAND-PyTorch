import os
import json
import pickle
import argparse
from tqdm import tqdm
from src.generator import SimulationGenerator


def gen_single_simulation_data(rank, n, m, nbinom, disp_param):
    g = SimulationGenerator(
        rank=rank,
        dim=10,
        e_dim=12,
        n_dim=4,
        c_dim=2,
        mutation=96,
        total_sample=n,
        mutations_per_sample=m,
        train_ratio=0.8,
        bt=[0.5, 0.5],
        epi=[0.4, 0.001, 0.001, 0.001,
             0.001, 0.002, 0.001, 0.003,
             0.020, 0.040, 0.020, 0.500],
        nuc=[0.75, 0.08, 0.01, 0.16],
        clu=[0.9, 0.1],
        nbinom=nbinom,
        disp_param=disp_param
    )

    return g.sample()


def generate_data(args):
    r_range = [5, 10, 20, 30]
    n_range = [50, 100, 1000, 2000]
    m_range = [50, 100, 1000, 2000]

    SAVE_DIR = f"data/simulation_{args.id}"

    if os.path.isdir(SAVE_DIR):
        raise NameError(f"Already exists the directory, {SAVE_DIR}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    with tqdm(total=len(r_range) * len(n_range) * len(m_range),
              desc=f'Simulation data generation id: {args.id}') as pbar:
        for row, n in enumerate(n_range):
            for col, m in enumerate(m_range):
                for rank in r_range:
                    data = gen_single_simulation_data(rank, n, m, args.nbinom, args.disp_param)

                    with open(os.path.join(SAVE_DIR, f'rank_{rank}_m_{m}_n_{n}.pkl'), 'wb') as f:
                        pickle.dump(data, f)

                    pbar.update(1)

    meta = {
        'r_range': r_range,
        'n_range': n_range,
        'm_range': m_range,
        'distribution': {
            'family': 'nbinom' if args.nbinom else 'poisson',
            'disp_param': args.disp_param if args.nbinom else None
        }
    }

    with open(os.path.join(SAVE_DIR, 'meta.json'), "w") as f:
        json.dump(meta, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data prepare')
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--nbinom', action='store_true')
    parser.add_argument('--disp_param', type=float, default=100)
    args = parser.parse_args()

    generate_data(args)
