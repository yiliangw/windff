from dataclasses import dataclass
import numpy as np
import xmlrpc.client
from ..config import Config
import logging


class Controller(object):

  def __init__(self, config: Config):
    self.config = config
    self.preprocessor_proxy = None
    self.predictor_proxy = None
    self.trainer_proxy = None
    self.broadcaster_proxy = None
    self.nodes: list[str] = []

  def run(self):
    self.data_preprocessor_proxy = xmlrpc.client.ServerProxy(
        self.config.preprocessor_url)

  def preprocess_data(self, time_start: np.datetime64, time_end: np.datetime64, interval: np.timedelta64):
    try:
      return self.data_preprocessor_proxy.preprocess_data(self.nodes,
                                                          time_start, time_end, interval)
    except xmlrpc.client.Fault as err:
      logging.error("XML-RPC fault (preprocess_data): %d %s",
                    err.faultCode, err.faultString)
      return None
