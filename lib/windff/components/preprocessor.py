from dataclasses import dataclass
from ..config import Config
import influxdb_client.client
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import WriteApi, SYNCHRONOUS
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
import logging

import xmlrpc.server


class Preprocessor(object):

  PADDING_INTERVAL_NB = 3

  """WindFF data preprocessor

  Altogether there are two separate data preprocessing steps in the pipeline, and the data preprocessor only handle the first step. In this step, the data are processed so that the timestamps are aligned and the missing data are interpolated. The second step is handled by the model, where the transformations are applied to the data for the model input.
  """

  def __init__(self, config: Config):
    self.config: Config = config

  def do_preprocess_turb_data(self, turbs: list[str], df: pd.DataFrame, time_start: np.datetime64, time_end: np.datetime64, time_interval: np.timedelta64) -> pd.DataFrame:
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
    query = f'''
    SELECT * FROM "{self.config.raw_db_config.turb_data_bucket}"
    WHERE time >= '{time_start - time_interval * self.PADDING_INTERVAL_NB}' \
    AND time <= '{time_end + time_interval * self.PADDING_INTERVAL_NB}'
    '''
    raw_df = self.__raw_db_query_api.query_data_frame(query)
    raw_df["timestamp"] = raw_df.index
    raw_df = raw_df.reset_index(drop=True)

    processed_df = self.do_preprocess_turb_data(raw_df)
    # Save the preprocessed data to the preprocessed database
    try:
      self.__processed_db_write_api.write(
          bucket=self.config.preprocessed_db.bucket,
          record=processed_df,
          data_frame_measurement_name=self.config.preprocessed_db.turb_ts_measurement,
          data_frame_tag_columns=["turb_id"],
          data_frame_time_column="timestamp"
      )
      return True
    except influxdb_client.client.InfluxDBError as err:
      logging.error("InfluxDB error (write): %s", err)
      return False

  def preprocess_data(self, turbs: set[str], time_start: np.datetime64, time_end: np.datetime64, time_interval: np.timedelta64):
    return self.preprocess_turb_data(turbs, time_start, time_end, time_interval)

  def run(self):
    self.__start_influxdb_clients()
    self.__start_xmlrpc_server()

  def __start_influxdb_clients(self):
    self.__raw_db_client = InfluxDBClient(url=self.config.raw_db_config.url,
                                          token=self.config.raw_db_config.token,
                                          org=self.config.raw_db_config.org)
    self.__raw_db_query_api = QueryApi(client=self.__raw_db_client)
    self.__preprocessed_db_client = InfluxDBClient(url=self.config.preprocessed_db_config.url,
                                                   token=self.config.preprocessed_db_config.token,
                                                   org=self.config.preprocessed_db_config.org)
    self.__processed_db_write_api = WriteApi(
        client=self.__preprocessed_db_client, write_options=SYNCHRONOUS)

  def __start_xmlrpc_server(self):
    self.__server = xmlrpc.server.SimpleXMLRPCServer(
        (self.config.server_config.listen_addr, self.config.server_config.listen_port))
    self.__server.register_function(self.preprocess_data, "preprocess_data")
    self.__server.serve_forever()
