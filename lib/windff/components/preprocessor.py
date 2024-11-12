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
    from ..env import WindffEnv
    self.env: WindffEnv = None
    self.rpc_server = None

  def initialize(self, env):
    self.env: Env = env

  def __do_preprocess_turb_data(self, turbs: list[str], df: pd.DataFrame, time_start: np.datetime64, time_end: np.datetime64, time_interval: np.timedelta64) -> pd.DataFrame:
    processed_dfs = []
    processed_turbs = set()
    for t, group in df.groupby("turb_id"):
      group = group.set_index("timestamp").interpolate(
          method="time").resample(time_interval).mean().fillna(0)
      group = group.reindex(pd.date_range(
          time_start, time_end, freq=time_interval)).fillna(method="ffill").fillna(method="bfill")
      group = group.reset_index(drop=False)
      processed_dfs.append(group)
      processed_turbs.add(t)

    for t in set(turbs) - processed_turbs:
      group = pd.DataFrame(columns=df.columns)
      # Change the column type to match the original data
      group["timestamp"] = pd.date_range(
          time_start, time_end, freq=time_interval)
      group["turb_id"] = t
      group = group.astype(df.dtypes.to_dict())
      group = group.fillna(0)
      processed_dfs.append(group)

    df = pd.concat(processed_dfs).reset_index(drop=True)
    return df

  def preprocess_turb_data(self, turbs: list[str], time_start: np.datetime64, time_end: np.datetime64, time_interval: np.timedelta64):
    try:
      raw_df = self.env.query_raw_turb_data_df(time_start, time_end)
      processed_df = self.__do_preprocess_turb_data(raw_df)
      self.env.write_preprocessed_turb_data_df(processed_df)
      return Errno.OK, "Ok"
    except DBQueryError as e:
      return Errno.DBQueryErr, str(e.raw)
    except DBWriteError as e:
      return Errno.DBWriteErr, str(e.raw)

  def preprocess(self, turbs: set[str], time_start: np.datetime64, time_end: np.datetime64, time_interval: np.timedelta64):
    return self.preprocess_turb_data(turbs, time_start, time_end, time_interval)

  def start(self):
    self.__start_xmlrpc_server()

  def __start_xmlrpc_server(self):
    self.rpc_server = xmlrpc.server.SimpleXMLRPCServer(
        (self.config.server_config.listen_addr, self.config.server_config.listen_port))
    self.rpc_server.register_function(self.preprocess_data, "preprocess_data")
    self.rpc_server.serve_forever()
