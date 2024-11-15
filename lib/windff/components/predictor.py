import numpy as np
import pandas as pd
import torch

from .component import Component
from ..model import ModelManager
from ..data.dataset import Graph, Dataset
from ..data.database import DatabaseID
from ..errors import Errno, DBQueryError, DBWriteError
from ..utils import Utils


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
    self.manager = ModelManager(self.env.config.model)
    self.env.connect_db(DatabaseID.PREDICTED)

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

    graph = Dataset.process_raw_data(
        raw_data, dtype=self.env.config.model.dtype)

    return graph

  def predict(self, time: np.datetime64):
    '''Predict with output window starting from the given time
    '''
    time = self.env.align_time(time)
    try:

      g = self.__prepare_data(time)
      res = self.manager.infer(g)

      assert res.shape[0] == len(g.nodes)
      assert res.shape[2] == len(
          self.env.config.type.raw_turb_data_type.get_target_col_names())

      turb_dfs = []
      for i in range(len(g.nodes)):
        df = pd.DataFrame()
        df[self.env.time_col] = pd.date_range(time, time + self.env.time_interval *
                                              (res.shape[1] - 1), freq=f'{Utils.td_to_sec(self.env.time_interval)}s')
        df[self.env.turb_col] = g.nodes[i]
        for j, col in enumerate(self.env.config.type.raw_turb_data_type.get_target_col_names()):
          df[col] = res[i, :, j]

        turb_dfs.append(df)

      df = pd.concat(turb_dfs).reset_index(drop=True)
      self.env.write_predicted_data_df(df)

      return Errno.OK, "Ok"
    except DBQueryError as e:
      return Errno.DBQueryErr, str(e.raw)
    except DBWriteError as e:
      return Errno.DBWriteErr, str(e.raw)

  def start(self):
    pass
