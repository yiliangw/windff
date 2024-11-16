import numpy as np
from dataclasses import dataclass

import logging

from ...windff.components import Component, Controller, Collector, Preprocessor, Predictor, Broadcaster
from ...windff.config import Config as WindffConfig, TypeConfig, InfluxDBConfig, ModelConfig
from ...windff.env import Env

from .turbine_edge import SDWPFTurbineEdge
from .client import SDWPFClient
from ..data.raw import SDWPFRawTurbData
from ..data.dataset import SDWPFDataset


class SDWPFSimulation:

  @dataclass
  class Config:
    influx_db: InfluxDBConfig
    time_start: np.datetime64
    time_interval: np.timedelta64
    time_duration: np.timedelta64

    collector_port: int
    broadcaster_port: int

  def __init__(self):

    self.time: np.datetime64 = None

    self.config: self.Config = None
    self.windff_config: WindffConfig = None

    self.controller: Controller = None
    self.collector: Collector = None
    self.preprocessor: Preprocessor = None
    self.predictor: Predictor = None
    self.broadcaster: Broadcaster = None
    self.turbine_edges: list[SDWPFTurbineEdge] = None
    self.client: SDWPFClient = None

  def setup(self, config: Config):

    dataset = SDWPFDataset()

    self.config = config

    self.windff_config = WindffConfig(
        influx_db=config.influx_db,
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

        time_interval=config.time_interval,

        preprocessor_url=None,
        predictor_url=None
    )
    self.env = Env(self.windff_config)

    self.controller = self.env.spawn(Controller.get_type())
    self.collector = self.env.spawn(Collector.get_type())
    self.preprocessor = self.env.spawn(Preprocessor.get_type())
    self.predictor = self.env.spawn(Predictor.get_type())
    self.broadcaster = self.env.spawn(Broadcaster.get_type())

    self.turbine_edges = SDWPFTurbineEdge.create_all(
        self.config.time_start, self.config.time_interval)
    self.client = SDWPFClient()

  def run(self):
    self.time = self.config.time_start

    time_stop = self.config.time_start + self.config.time_duration

    interval_nb = 1
    self.time += self.config.time_interval

    while self.time <= time_stop:
      for edge in self.turbine_edges:
        raw_data = edge.get_raw_data(interval_nb)
        self.collector.handle_raw_turb_data(raw_data)

      self.controller.process(self.time)

      query = self.client.create_query(
          self.time, self.time + self.config.time_interval * self.windff_config.model.output_win_sz)

      response = self.broadcaster.handle_query(query)

      self.time += self.config.time_interval
      interval_nb += 1

      return
