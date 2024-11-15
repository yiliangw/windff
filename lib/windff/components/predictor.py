import xmlrpc.server
import numpy as np
import pandas as pd

from .component import Component
from ..model import ModelManager
from ..data.dataset import Graph, Dataset
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
        time_start, self.env.time_interval, self.env.time_win_sz)

    turb_ts_df = turb_ts_df.ffill().bfill().fillna(0.0)

    raw_data = Dataset.RawData(
        turb_id_col=self.env.turb_col,
        time_col=self.env.time_col,

        turb_location_df=self.env.turb_loc_df,
        turb_timeseries_df=turb_ts_df,
        turb_timeseries_target_cols=self.env.config.type.raw_turb_data_type.get_target_col_names(),
    )

    graph = Dataset.process_raw_data(raw_data)

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

  def start(self):
    pass
