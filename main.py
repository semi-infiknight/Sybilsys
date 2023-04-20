import os
import argparse
import Qgans_molGen as Solver
from torch.backends import cudnn

def main(config):
      if config.mode == 'train':
        Solver.TrainNtest().train()
        # Solver.train()
    elif config.mode == 'test':
        Solver.TrainNtest().test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()
    print("Config: ", config)
    main(config)