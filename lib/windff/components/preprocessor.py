import pandas as pd
import numpy as np

import xmlrpc.server
import logging

from .component import Component
from ..errors import Errno, DBQueryError, DBWriteError
from ..data.database import DatabaseID


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

    self.env.connect_db(DatabaseID.RAW)
    self.env.connect_db(DatabaseID.PREPROCESSED)

  def __do_preprocess_turb_data(self, turbs: list[str], df: pd.DataFrame, time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int) -> pd.DataFrame:

    df[self.env.turb_col] = df[self.env.turb_col].astype(str)

    time_end = time_start + time_interval * (interval_nb - 1)

    interval_str = f"{time_interval.astype('timedelta64[s]').astype(int)}s"

    processed_dfs = []

    g = df.groupby(self.env.turb_col)
    for k in g.groups.keys():
      gdf = g.get_group(k).reset_index(drop=True)
      gdf = gdf.drop(columns=[self.env.turb_col])

      gdf = gdf.set_index(self.env.time_col).interpolate(
          method="linear")
      gdf = gdf.resample(interval_str).mean()
      gdf = gdf.reindex(pd.date_range(time_start, time_end,
                        freq=interval_str)).ffill().bfill().fillna(0.0)

      gdf.index.name = self.env.time_col
      gdf = gdf.reset_index()
      gdf[self.env.turb_col] = k
      processed_dfs.append(gdf)

    for t in turbs:
      if t in g.groups.keys():
        continue
      gdf = pd.DataFrame(columns=df.columns)
      gdf = gdf.astype(df.dtypes.to_dict())
      # Change the column type to match the original data
      gdf[self.env.time_col] = pd.date_range(
          time_start, time_end, freq=interval_str)
      gdf[self.env.turb_col] = t
      processed_dfs.append(gdf)

    df = pd.concat(processed_dfs).reset_index(drop=True)
    df = df.fillna(0.0)
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

  def preprocess(self, time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int):
    return self.preprocess_turb_data(self.env.turb_list, time_start, time_interval, interval_nb)

  def start(self):
    self.__start_xmlrpc_server()

  def __start_xmlrpc_server(self):
    self.rpc_server = xmlrpc.server.SimpleXMLRPCServer(
        (self.config.server_config.listen_addr, self.config.server_config.listen_port))
    self.rpc_server.register_function(self.preprocess_data, "preprocess_data")
    self.rpc_server.serve_forever()
