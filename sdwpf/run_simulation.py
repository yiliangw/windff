import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from lib.sdwpf.sim import SDWPFSimulation


def main():
  sim = SDWPFSimulation()

  sim.setup(SDWPFSimulation.Config(
      time_start=np.datetime64('2024-11-10T17:00:00'),
      time_interval=np.timedelta64(10, 'm'),
      time_duration=np.timedelta64(1, 'h')
  ))

  sim.run()


if __name__ == '__main__':
  main()
