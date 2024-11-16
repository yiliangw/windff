import numpy as np
import pandas as pd

import logging
import inspect
import influxdb_client.client
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import WriteApi, SYNCHRONOUS
from influxdb_client import InfluxDBClient

from flask import Flask, request, jsonify

from .config import Config
from .components import Component, Controller, Collector, Preprocessor, Predictor, Broadcaster
from .errors import DBError, DBConnectionError, DBWriteError, DBQueryError, RawDataParsingError
from .data import RawTurbData
from .data.database import DatabaseID
from .utils import Utils


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

    self.config.type.raw_turb_data_type.init(self.time_col, self.turb_col)

    from ..sdwpf.data.dataset import SDWPFDataset

    # WARNING: Hardcoded for now
    sdwpf = SDWPFDataset()[1]
    self.turb_list: list[str] = sdwpf.nodes
    self.turb_loc_df: pd.DataFrame = pd.read_csv(SDWPFDataset.LOCATION_CSV)
    self.turb_loc_df[self.turb_col] = self.turb_loc_df['TurbID'].astype(str)
    self.turb_loc_df = self.turb_loc_df.drop(columns=['TurbID'])

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

  def start_services(self, port: int):
    self.flask_server = Flask(f'{__name__}')

    @self.flask_server.route("/raw_turb_data", methods=["POST"])
    def handle_raw_turbine_data():
      try:
        data_json = request.json
        data = self.parse_raw_turb_data(data_json)
      except RawDataParsingError as e:
        logging.warning(
            f"Discarded faulty raw turbine data: {data_json}, err: {str(e)}")
        return jsonify({"error": str(e)}), 400

      try:
        collector: Collector = self.get_component(Collector.get_type())
        assert collector is not None
        collector.handle_raw_turb_data(data)
      except DBError as e:
        logging.warning(f"Ignored raw turbine data due to DB error: {str(e)}")
        return jsonify({"error": "DB error"}), 500

      return jsonify({"status": "OK"}), 200

    @self.flask_server.route("/query", methods=["GET"])
    def handle_query():
      query_json = request.json
      try:
        broadcaster: Broadcaster = self.get_component(Broadcaster.get_type())
        assert broadcaster is not None
        resp = broadcaster.handle_query(query_json)
      except ValueError as e:
        return jsonify({"error": str(e)}), 400

      return jsonify(resp), 200

    self.flask_server.run(host='0.0.0.0', port=port)

  def get_component(self, type: str) -> Component:
    comp_l = self.get_components(type)
    if comp_l is None or len(comp_l) == 0:
      return None
    return comp_l[0]

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

  def call_preprocess(self, time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int):
    comp_l = self.get_components(Preprocessor.get_type())
    assert len(comp_l) > 0
    preprocessor: Preprocessor = comp_l[0]
    return preprocessor.preprocess(time_start, time_interval, interval_nb)

  def call_predict(self, time: np.datetime64):
    comp_l = self.get_components(Predictor.get_type())
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

    query = f'from (bucket: "{self.config.influx_db.raw_data_bucket}")\
    |> range(start: {Utils.dt_to_sec(time_start)}, stop: {Utils.dt_to_sec(time_end)})\
    |> filter(fn: (r)=> r["_measurement"] == "{self.config.influx_db.raw_turb_ts_measurement}")\
    |> aggregateWindow(every: {Utils.td_to_sec(time_interval)}s, fn: mean)\
    |> pivot(rowKey:["_time", "{self.turb_col}"], columnKey: ["_field"], valueColumn: "_value")\
    |> keep(columns: ["_time", "{self.turb_col}"'

    for col in self.config.type.raw_turb_data_type.get_col_names():
      query += f', "{col}"'

    query += ']) |> yield()'

    # Change time column to timestamp
    try:
      df = self.__influx_query_api.query_data_frame(query)
    except InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBQueryError(inspect.currentframe(), err)

    df = df.rename(columns={'_time': self.time_col})
    df[self.time_col] = pd.to_datetime(df[self.time_col], unit='s')
    df[self.turb_col] = df[self.turb_col].astype(str)

    df = df.drop(columns=['result', 'table'])
    df = df.reset_index(drop=True)

    for col in df.columns:
      if col != self.time_col and col != self.turb_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

  def query_preprocessed_turb_data_df(self, time_start: np.datetime64, time_interval: np.timedelta64, interval_nb: int):
    time_end = time_start + time_interval * interval_nb

    query = f'from (bucket: "{self.config.influx_db.preprocessed_data_bucket}")\
    |> range(start: {Utils.dt_to_sec(time_start)}, stop: {Utils.dt_to_sec(time_end)})\
    |> filter(fn: (r)=> r["_measurement"] == "{self.config.influx_db.preprocessed_turb_ts_measurement}")\
    |> aggregateWindow(every: {Utils.td_to_sec(time_interval)}s, fn: mean)\
    |> pivot(rowKey:["_time", "{self.turb_col}"], columnKey: ["_field"], valueColumn: "_value")\
    |> keep(columns: ["_time", "{self.turb_col}"'

    for col in self.config.type.raw_turb_data_type.get_col_names():
      query += f', "{col}"'

    query += ']) |> yield()'

    try:
      df = self.__influx_query_api.query_data_frame(query)
    except InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBQueryError(inspect.currentframe(), err)

    df = df.rename(columns={'_time': self.time_col})
    df[self.time_col] = pd.to_datetime(df[self.time_col], unit='s')
    df[self.turb_col] = df[self.turb_col].astype(str)

    df = df.drop(columns=['result', 'table'])
    df = df.reset_index(drop=True)

    for col in df.columns:
      if col != self.time_col and col != self.turb_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

  def query_predicted_data_df(self, time_start: np.datetime64, time_stop: np.datetime64) -> pd.DataFrame:

    query = f'from (bucket: "{self.config.influx_db.predicted_data_bucket}")\
    |> range(start: {Utils.dt_to_sec(time_start)}, stop: {Utils.dt_to_sec(time_stop)})\
    |> filter(fn: (r)=> r["_measurement"] == "{self.config.influx_db.predicted_turb_ts_measurement}")\
    |> pivot(rowKey:["_time", "{self.turb_col}"], columnKey: ["_field"], valueColumn: "_value")\
    |> keep(columns: ["_time", "{self.turb_col}"'

    for col in self.config.type.raw_turb_data_type.get_target_col_names():
      query += f', "{col}"'

    query += ']) |> yield()'

    try:
      df = self.__influx_query_api.query_data_frame(query)
    except InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBQueryError(inspect.currentframe(), err)

    df = df.rename(columns={'_time': self.time_col})
    df[self.time_col] = pd.to_datetime(df[self.time_col], unit='s')
    df[self.turb_col] = df[self.turb_col].astype(str)

    df = df.drop(columns=['result', 'table'])
    df = df.reset_index(drop=True)

    for col in df.columns:
      if col != self.time_col and col != self.turb_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

  def write_preprocessed_turb_data_df(self, df: pd.DataFrame):
    self.__check_db_connection(DatabaseID.PREPROCESSED)
    df = df.set_index(self.time_col)
    try:
      self.__influx_write_api.write(
          bucket=self.config.influx_db.preprocessed_data_bucket,
          record=df,
          data_frame_measurement_name=self.config.influx_db.preprocessed_turb_ts_measurement,
          data_frame_tag_columns=[self.turb_col],
      )
    except InfluxDBClient as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBWriteError(inspect.currentframe(), err)

  def write_predicted_data_df(self, df: pd.DataFrame):
    self.__check_db_connection(DatabaseID.PREDICTED)
    df = df.set_index(self.time_col)
    try:
      self.__influx_write_api.write(
          bucket=self.config.influx_db.predicted_data_bucket,
          record=df,
          data_frame_measurement_name=self.config.influx_db.predicted_turb_ts_measurement,
          data_frame_tag_columns=[self.turb_col],
      )
    except influxdb_client.client.InfluxDBError as err:
      logging.error(f"InfluxDB error ({inspect.currentframe()}): %s", err)
      raise DBWriteError(inspect.currentframe(), err)

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
