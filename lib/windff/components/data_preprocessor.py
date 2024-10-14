from flask import Flask, request, jsonify
from dataclasses import dataclass
from ..config import InfluxDBClientConfig, FlaskServerConfig
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import WriteApi, SYNCHRONOUS
from influxdb_client import InfluxDBClient


class DataPreprocessor(object):

  """WindFF data preprocessor

  Altogether there are two separate data preprocessing steps in the pipeline, and the data preprocessor only handle the first step. In this step, the data are processed so that the timestamps are aligned and the missing data are interpolated. The second step is handled by the model, where the transformations are applied to the data for the model input.
  """

  @dataclass
  class Config:
    raw_db_config: InfluxDBClientConfig
    preprocessed_db_config: InfluxDBClientConfig
    server_config: FlaskServerConfig

  def __init__(self, config: Config):
    self.config = config

  def preprocess_turb_data(self):
    pass

  def start(self):
    self.__start_influxdb_clients()
    self.__start_flask_server()

  def __start_influxdb_clients(self):
    self.__raw_db_client = InfluxDBClient(url=self.config.raw_db_config.url,
                                          token=self.config.raw_db_config.token,
                                          org=self.config.raw_db_config.org)
    self.__raw_db_query_api = QueryApi(client=self.__raw_db_client)
    self.__preprocessed_db_client = InfluxDBClient(url=self.config.preprocessed_db_config.url,
                                                   token=self.config.preprocessed_db_config.token,
                                                   org=self.config.preprocessed_db_config.org)
    self.__preprocessed_db_write_api = WriteApi(
        client=self.__preprocessed_db_client, write_options=SYNCHRONOUS)

  def __start_flask_server(self):
    self.__server = Flask(__name__)

    @self.__server.route("/preprocess", methods=["POST"])
    def handle_preprocess():
      pass

    self.__server.run(host=self.config.server_config.listen_addr,
                      port=self.config.server_config.listen_port)
