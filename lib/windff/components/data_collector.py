from ..data import TurbRawData, DataParsingError
from dataclasses import dataclass
from flask import Flask, request, jsonify
import logging

import influxdb_client as influxc
from influxdb_client.client.write_api import SYNCHRONOUS


class DataCollector(object):

  @dataclass
  class Config:
    turb_raw_dtype: type
    listen_addr: str
    listen_port: int
    raw_db_url: str
    raw_db_toekn: str
    raw_db_org: str
    turb_bucket: str

  def __init__(self, config: Config):
    self.config = config

    self.server: Flask = None

    self.influxdb_client = None
    self.raw_db_write_api = None

  def __receive_turb_raw_data(self, data: TurbRawData):
    try:
      self.raw_db_write_api.write(
          bucket=self.config.turb_bucket, record=data.to_influxdb_point())
      return True
    except influxc.client.InfluxDBError as e:
      logging.error(
          f"Error writing to InfluxDB: data({str(data)}) err({str(e)})")
      return False

  def __start_flask_server(self):
    self.server = Flask(__name__)

    @self.server.route("/turb_raw_data", methods=["POST"])
    def handle_turb_raw_data():
      try:
        data = self.config.turb_raw_dtype.from_json(request.data)
        res = self.__receive_turb_raw_data(data)
        if not res:
          return jsonify({"error": "Error writing to InfluxDB"}), 500
      except DataParsingError as e:
        logging.warning(f"Error parsing data: {str(e)}")
        return jsonify({"error": str(e)}), 400

    self.server.run(host=self.config.listen_addr, port=self.config.listen_port)

  def __start_influxdb_client(self):
    self.influxdb_client = influxc.InfluxDBClient(
        url=self.config.raw_db_url, token=self.config.raw_db_token, org=self.config.raw_db_org)
    self.raw_db_write_api = self.influxdb_client.write_api(
        write_options=SYNCHRONOUS)

  def start(self):
    self.__start_influxdb_client()
    self.__start_flask_server()
