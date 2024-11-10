import xmlrpc.server
import numpy as np
import pandas as pd

from .component import Component
from ..model import ModelManager
from ..data.dataset import Graph
from ..errors import Errno, DBQueryError, DBWriteError


class Predictor(Component):

  @classmethod
  def get_type(cls):
    return "predictor"

  def __init__(self):
    from ..env import Env
    self.env: Env = None
    self.manager: ModelManager = None

  def initialize(self, env):
    self.env = env
    # TODO: Initialize model

  def __prepare_data(self, time: np.datetime64):
    '''Prepare the data for prediction
    @param time: The time for prediction
    '''
    time_stop = self.env.align_time(time)
    time_start = time_stop - self.env.time_interval * self.env.time_win_sz

    turb_ts_df = self.env.query_preprocessed_turb_data_df(
        time_start, time_stop)

    data = self.manager.pre

    graph = Graph(len(self.env.turb_list), self.env.edges, turb_ts_df, None)
    return graph

  def __do_predict(self, g: Graph) -> pd.DataFrame:
    res = self.manager.infer(g)

  def predict(self, time: np.datetime64):
    '''Predict with output window starting from the given time
    '''
    try:
      g = self.__prepare_data(time)
      res = self.__do_predict(g)

      return Errno.OK, "Ok"
    except DBQueryError as e:
      return Errno.DBQueryErr, str(e.raw)
    except DBWriteError as e:
      return Errno.DBWriteErr, str(e.raw)
