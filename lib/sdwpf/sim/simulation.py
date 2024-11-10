import numpy as np
from dataclasses import dataclass

from ...windff.components import Component, Controller, Collector, Preprocessor, Predictor, Broadcaster
from ...windff import Config as WindFFConfig, Env

from .turbine_edge import SDWPFTurbineEdge
from .client import SDWPFClient


class SDWPFSimulation:

  @dataclass
  class Config:
    time_start: np.datetime64
    time_interval: np.timedelta64
    time_duration: np.timedelta64

  def __init__(self):

    self.time: np.datetime64 = None

    self.controller: Controller = None
    self.collector: Collector = None
    self.preprocessor: Preprocessor = None
    self.predictor: Predictor = None
    self.broadcaster: Broadcaster = None
    self.turbine_edges: list[SDWPFTurbineEdge] = None
    self.client: SDWPFClient = None

  def setup(self, config: Config):

    self.config = config
    self.env = Env(config)

    self.contorller = self.env.spawn(Controller.get_type())
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
    interval_nb = 0

    while self.time <= time_stop:
      for edge in self.turbine_edges:
        raw_data_json = edge.get_raw_data_json(interval_nb)
        self.collector.handle_raw_turb_data_json(raw_data_json)

    self.controller.process(self.time)
