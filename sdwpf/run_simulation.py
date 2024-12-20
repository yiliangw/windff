import numpy as np


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from lib.sdwpf.sim import SDWPFSimulation
from lib.windff.config import InfluxDBConfig

import logging
logging.basicConfig(level=logging.INFO)


def main():
  sim = SDWPFSimulation()

  sim.setup(SDWPFSimulation.Config(
      influx_db=InfluxDBConfig(
          url='http://localhost:8086',
          org='windff',
          user='windff',
          token='ILWXcDGuMhLBxr0LaeJodwyzFN3dIJhSM6jBZJ8ph37LtxdpevpBFu1ypD-MEmnSrHPyh6a_SSzFAl6JTUxhbA==',

          raw_data_bucket='windff_raw',
          raw_turb_ts_measurement='turb_ts',

          preprocessed_data_bucket='windff_preprocessed',
          preprocessed_turb_ts_measurement='turb_ts',

          predicted_data_bucket='windff_predicted',
          predicted_turb_ts_measurement='turb_ts'
      ),
      time_start=np.datetime64('2024-11-10T17:00:00', 's'),
      time_interval=np.timedelta64(10, 'm'),
      time_duration=np.timedelta64(1, 'h'),

      collector_port=17000,
      broadcaster_port=17000
  ))

  sim.run()


if __name__ == '__main__':
  main()
