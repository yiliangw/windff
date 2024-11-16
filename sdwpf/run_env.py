import os
import sys
import argparse
import logging
import signal

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lib.windff.components import Component, Controller, Collector, Preprocessor, Predictor, Broadcaster
from lib.windff.config import Config as WindffConfig, TypeConfig, InfluxDBConfig, ModelConfig
from lib.windff.env import Env

from lib.sdwpf.data.raw import SDWPFRawTurbData
from lib.sdwpf.data.dataset import SDWPFDataset


influx_config = InfluxDBConfig(
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
)

time_start = np.datetime64('2024-11-10T17:00:00', 's')
time_interval = np.timedelta64(10, 'm')
time_duration = np.timedelta64(1, 'h')

port = 18181

dataset = SDWPFDataset()

config = WindffConfig(
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
    model=ModelConfig(
        feat_dim=dataset.get_feat_dim(),
        target_dim=dataset.get_target_dim(),
        hidden_dim=64,
        input_win_sz=6 * 24,  # 1 day
        output_win_sz=6 * 2,  # 2 hours
        hidden_win_sz=6 * 12,
        adj_weight_threshold=0.8
    ),
    type=TypeConfig(
        raw_turb_data_type=SDWPFRawTurbData
    ),
    time_interval=time_interval,
    preprocessor_url=None,
    predictor_url=None
)

env = Env(config)
controller = env.spawn(Controller.get_type())
collector = env.spawn(Collector.get_type())
preprocessor = env.spawn(Preprocessor.get_type())
predictor = env.spawn(Predictor.get_type())
broadcaster = env.spawn(Broadcaster.get_type())

current_time = time_start + time_interval


def tick(sig, frame):
  global current_time

  print(f"Processing time: {current_time}")
  controller.process(current_time)
  print(f"Processed time: {current_time}")

  current_time += time_interval
  if (current_time - time_duration) >= time_start:
    print("Simulation finished")
    exit(0)


tick(None, None)


signal.signal(signal.SIGINT, tick)

env.start_services(port)
