from enum import Enum

import influxdb_client.client
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import WriteApi, SYNCHRONOUS
from influxdb_client import InfluxDBClient


import numpy as np
import pandas as pd

import logging
import inspect

from .config import Config
from .components import Component, Controller, Collector, Preprocessor, Predictor, Broadcaster
from .errors import DBConnectionError, DBWriteError, DBQueryError
from .data import RawTurbData
from .data.database import DatabaseID


class Env:

  logger = logging.getLogger(__qualname__)

  component_types = {
      t.get_type(): t for t in [Controller, Collector, Preprocessor, Predictor, Broadcaster]
  }

  def __init__(self, config: Config):
    self.config = config
    self.components = {}

    self.time_col: str = 'timestamp'
    self.turb_col: str = 'turb_id'

    self.turb_list: list[str]
    self.edges: list[tuple[int, int, float]]

    self.preprocess_retry_nb: int = 3
    self.predict_retry_nb: int = 3

    self.__influx_client: InfluxDBClient = None
    self.__influx_query_api: QueryApi = None
    self.__influx_write_api: WriteApi = None
    # TODO: Save model parameters with MongoDB?

    self.__connected_dbs = set()

  def spawn(self, type: str) -> Component:
    '''Spawn a component of the given type
    '''

    t = self.component_types.get(type)
    if t is None:
      raise ValueError(f"Invalid component type: {type}")

    comp = t()
    comp.initialize(self)

    type_list = self.components.get(type, [])
    type_list.append(comp)
    self.components[type] = type_list

    return comp

  @property
  def time_interval(self):
    return self.config.time_interval

  @property
  def time_win_sz(self):
    return self.config.model.input_win_sz

  def get_components(self, type):
    '''Get the components of the given type
    '''
    return self.components.get(type, [])

  def call_preprocess(self, turbs: set[str], time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int):
    comp_l = self.get_components(Preprocessor.get_type())
    assert len(comp_l) > 0
    preprocessor: Preprocessor = comp_l[0]
    return preprocessor.preprocess(turbs, time_start, time_interval, interval_nb)

  def call_predict(self, time: np.datetime64):
    comp_l = self.get_components(Component.Type.PREDICTOR)
    assert len(comp_l) > 0
    predictor: Predictor = comp_l[0]
    return predictor.predict(time)

  def connect_db(self, dbid: DatabaseID):
    if self.__influx_client is None:
      self.__influx_client = InfluxDBClient(
          url=self.config.influx_db.url,
          token=self.config.influx_db.token,
          org=self.config.influx_db.org
      )
      self.__influx_query_api = QueryApi(self.__influx_client)
      self.__influx_write_api = WriteApi(
          self.__influx_client, write_options=SYNCHRONOUS
      )

    self.__connected_dbs.add(dbid)

  def write_raw_turb_data(self, data: RawTurbData):

    self.logger.info(f"Writing raw turbine data: {data.to_json()}")

    self.__check_db_connection(DatabaseID.RAW)

    p = data.to_influxdb_point(
        self.config.influx_db.raw_turb_ts_measurement)

    try:
      res = self.__influx_write_api.write(
          bucket=self.config.influx_db.raw_data_bucket,
          org=self.config.influx_db.org,
          record=p,
      )
    except InfluxDBError as err:
      logging.error(f"InfluxDB error ({str(inspect.currentframe())}): %s", err)
      raise DBWriteError(str(inspect.currentframe), err)

    return res

  def query_raw_turb_data_df(self, time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int):
    self.__check_db_connection(DatabaseID.RAW)

    time_end = time_start + time_interval * interval_nb

    start_timestamp_s = time_start.astype('datetime64[s]').astype('int')
    end_timestamp_s = time_end.astype('datetime64[s]').astype('int')
    interval_s = time_interval.astype('timedelta64[s]').astype('int')

    query = f'from (bucket: "{self.config.influx_db.raw_data_bucket}")\
    |> range(start: {start_timestamp_s}, stop: {end_timestamp_s})\
    |> filter(fn: (r)=> r["_measurement"] == "{self.config.influx_db.raw_turb_ts_measurement}")\
    |> aggregateWindow(every: {interval_s}s, fn: mean)\
    |> pivot(rowKey:["_time", "{self.turb_col}"], columnKey: ["_field"], valueColumn: "_value")'

    # Change time column to timestamp
    try:
      df = self.__influx_query_api.query_data_frame(query)
    except InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBQueryError(inspect.currentframe(), err)
    df["timestamp"] = df.index
    df = df.reset_index(drop=True)
    return df

  def write_preprocessed_turb_data_df(self, df: pd.DataFrame):
    self.__check_db_connection(DatabaseID.PREPROCESSED)
    try:
      self.__influx_write_api.write(
          bucket=self.config.preprocessed_db.bucket,
          record=df,
          data_frame_measurement_name=self.config.preprocessed_db.turb_ts_measurement,
          data_frame_tag_columns=[self.turb_col],
          data_frame_time_column=self.time_col
      )
    except influxdb_client.client.InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBWriteError(inspect.currentframe(), err)

  def query_preprocessed_turb_data_df(self, time_start: np.datetime64, time_end: np.datetime64):
    self.__check_db_connection(DatabaseID.PREPROCESSED)
    query = f'''
    from(bucket: "{self.config.preprocessed_db.bucket}")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "{self.config.preprocessed_db.turb_ts_measurement}")
      |> yield()
    '''
    # Change time column to timestamp
    try:
      df = self.__influx_query_api.query_data_frame(query)
    except influxdb_client.client.InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBQueryError(inspect.currentframe(), err)
    df["timestamp"] = df.index
    df = df.reset_index(drop=True)
    return df

  def write_predicted_data_df(self, df: pd.DataFrame):
    self.__check_db_connection(DatabaseID.PREDICTED)
    try:
      self.__influx_write_api.write(
          bucket=self.config.predicted.bucket,
          record=df,
          data_frame_measurement_name=self.config.predicted_db.turb_ts_measurement,
          data_frame_tag_columns=[self.turb_col],
          data_frame_time_column=self.time_col
      )
    except influxdb_client.client.InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBWriteError(inspect.currentframe(), err)

  def query_predicted_data_df(self, time_start: np.datetime64, time_end: np.datetime64):
    self.__check_db_connection(DatabaseID.PREDICTED)
    query = f'''
    from(bucket: "{self.config.predicted_db.bucket}")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "{self.config.predicted_db.turb_ts_measurement}")
      |> yield()
    '''
    # Change time column to timestamp
    try:
      df = self.__influx_query_api.query_data_frame(query)
    except influxdb_client.client.InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBQueryError(inspect.currentframe(), err)
    df["timestamp"] = df.index
    df = df.reset_index(drop=True)
    return df

  def parse_raw_turb_data(self, data_json: str) -> RawTurbData:
    return self.config.type.raw_turb_data_type.from_json(data_json)

  def __check_db_connection(self, db_id: DatabaseID):
    if db_id not in self.__connected_dbs:
      logging.error(f"Database {db_id} not connected")
      raise DBConnectionError(f"Database {db_id} not connected")

  def load_model_state_dict(self, time: np.datetime64) -> dict:
    '''Get the latest model state dict before the given time
    '''
    # TODO:
    pass

  def save_model_state_dict(self, time: np.datetime64, state_dict: dict):
    # TODO:
    pass

  def align_time(self, time: np.datetime64) -> np.datetime64:
    return np.datetime64(time, np.datetime_data(self.time_interval))
