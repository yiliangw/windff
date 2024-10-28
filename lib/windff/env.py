from enum import Enum

import influxdb_client.client
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import WriteApi, SYNCHRONOUS
from influxdb_client import InfluxDBClient


import numpy as np
import pandas as pd

import logging
import inspect

from .config import Config
from .components import Component, Controller, Collector, Preprocessor, Predictor
from .errors import DBConnectionError, DBWriteError, DBQueryError
from .data import RawTurbData


class DatabaseID(Enum):
  RAW = 1
  PREPROCESSED = 2
  PREDICTED = 3


class Env:

  def __init__(self, config: Config):
    self.config = config
    self.components = {}

    self.time_col: str = 'timestamp'
    self.turb_col: str = 'turb_id'

    self.time_interval: np.timedelta64
    self.time_retry_interval: np.timedelta64
    self.time_win_sz: int
    self.time_guard: np.timedelta64

    self.preprocess_retry_nb: int = 3
    self.predict_retry_nb: int = 3

    self.raw_turb_data_dtype: type

    self.__influx_client: InfluxDBClient = None
    self.__influx_query_api: QueryApi = None
    self.__influx_write_api: WriteApi = None
    # TODO: Save model parameters with MongoDB?

    self.__connected_dbs = set()

  def spawn(self, type: Component.Type) -> Component:
    '''Spawn a component of the given type
    '''
    comp = Component.create(type)
    comp.initialize(self)

    type_list = self.components.get(type, [])
    type_list.append(comp)
    self.components[type] = type_list

    return comp

  def get_components(self, type: Component.Type):
    '''Get the components of the given type
    '''
    return self.components.get(type, [])

  def call_preprocess(self, turbs: set[str], time_start: np.datetime64, time_end: np.datetime64, time_interval: np.timedelta64):
    comp_l = self.get_components(Component.Type.PREPROCESSOR)
    assert len(comp_l) > 0
    preprocessor: Preprocessor = comp_l[0]
    return preprocessor.preprocess(turbs, time_start, time_end, time_interval)

  def call_predict(self, time: np.datetime64):
    comp_l = self.get_components(Component.Type.PREDICTOR)
    assert len(comp_l) > 0
    predictor: Predictor = comp_l[0]
    return predictor.predict(time)

  def connect_db(self, id_list: list[DatabaseID]):
    if self.__influx_client is None:
      self.__influx_client = InfluxDBClient(
          url=self.config.influxdb_config.url,
          token=self.config.influxdb_config.token,
          org=self.config.influxdb_config.org
      )
      self.__influx_query_api = QueryApi(client=self.__influx_client)
      self.__influx_write_api = WriteApi(
          client=self.__influx_client, write_options=SYNCHRONOUS
      )

    self.__connected_dbs += set(id_list)

  def write_raw_turb_data(self, dfdata: RawTurbData):
    self.__check_db_connection(DatabaseID.RAW)
    try:
      self.__influx_write_api.write(
          bucket=self.config.raw_db.bucket,
          record=dfdata.to_influxdb_point(),
          data_frame_measurement_name=self.config.raw_db.turb_ts_measurement
      )
    except influxdb_client.client.InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBWriteError(inspect.currentframe(), err)

  def query_raw_turb_data_df(self, time_start: np.datetime64, time_end: np.datetime64):
    self.__check_db_connection(DatabaseID.RAW)
    query = f'''
    from(bucket: "{self.config.raw_db.bucket}")
      |> range(start: {time_start}, stop: {time_end})
      |> filter(fn: (r) => r["_measurement"] == "{self.config.raw_db.turb_ts_measurement}")
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
    return self.raw_turb_data_dtype.from_json(data_json)

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
