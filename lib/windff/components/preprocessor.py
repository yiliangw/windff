import pandas as pd
import numpy as np

import xmlrpc.server
import logging

from .component import Component
from ..errors import Errno, DBQueryError, DBWriteError


class Preprocessor(Component):

  PADDING_INTERVAL_NB = 3

  """Windff data preprocessor

  Altogether there are two separate data preprocessing steps in the pipeline, and the data preprocessor only handle the first step. In this step, the data are processed so that the timestamps are aligned and the missing data are interpolated. The second step is handled by the model, where the transformations are applied to the data for the model input.
  """

  @classmethod
  def get_type(cls):
    return 'preprocessor'

  def __init__(self):
    from ..env import Env
    self.env: Env = None
    self.rpc_server = None

  def initialize(self, env):
    self.env = env

  def __do_preprocess_turb_data(self, turbs: list[str], df: pd.DataFrame, time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int) -> pd.DataFrame:
    time_end = time_start + time_interval * interval_nb
    processed_dfs = []

    g = df.groupby(self.env.turb_col)
    for k in g.groups.keys():
      gdf = g.get_group(k).reset_index(drop=True)

      gdf = gdf.set_index("timestamp").interpolate(
          method="time").resample(time_interval).mean().fillna(0)
      gdf = gdf.reindex(pd.date_range(
          time_start, time_end, freq=time_interval)).fillna(method="ffill").fillna(method="bfill")
      gdf = gdf.reset_index(drop=False)
      processed_dfs.append(gdf)

    for t in set(turbs) - set(g.groups.keys()):
      gdf = pd.DataFrame(columns=df.columns)
      # Change the column type to match the original data
      gdf["timestamp"] = pd.date_range(
          time_start, time_end, freq=time_interval)
      gdf["turb_id"] = t
      gdf = gdf.astype(df.dtypes.to_dict())
      gdf = gdf.fillna(0)
      processed_dfs.append(gdf)

    df = pd.concat(processed_dfs).reset_index(drop=True)
    return df

  def preprocess_turb_data(self, turbs: list[str], time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int):
    try:
      raw_df = self.env.query_raw_turb_data_df(
          time_start, time_interval, interval_nb)
      processed_df = self.__do_preprocess_turb_data(
          turbs, raw_df, time_start, time_interval, interval_nb)
      self.env.write_preprocessed_turb_data_df(processed_df)
      return Errno.OK, "Ok"
    except DBQueryError as e:
      return Errno.DBQueryErr, str(e.raw)
    except DBWriteError as e:
      return Errno.DBWriteErr, str(e.raw)

  def preprocess(self, turbs: set[str], time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int):
    return self.preprocess_turb_data(turbs, time_start, time_interval, interval_nb)

  def start(self):
    self.__start_xmlrpc_server()

  def __start_xmlrpc_server(self):
    self.rpc_server = xmlrpc.server.SimpleXMLRPCServer(
        (self.config.server_config.listen_addr, self.config.server_config.listen_port))
    self.rpc_server.register_function(self.preprocess_data, "preprocess_data")
    self.rpc_server.serve_forever()
