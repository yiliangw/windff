from flask import Flask, request, jsonify

import logging

from ..env import Env
from .component import Component
from ..data.raw import RawTurbData
from ..errors import WindffDBError, WindffRawDataParsingError


class Collector(Component):

  def get_type(cls):
    return Component.Type.COLLECTOR

  def __init__(self):
    self.env: Env = None

    self.server: Flask = None

  def initialize(self, env):
    logging.info("Initializing collector...")
    self.env = env
    logging.info("Collector initialized.")

  def __receive_turb_raw_data(self, data: RawTurbData):
    try:
      self.env.write_raw_turb_data(data)
    except WindffDBError as e:
      logging.warning("Ignored raw turbine data due to DB error.")

  def __start_flask_server(self):
    self.server = Flask(__name__)

    @self.server.route("/raw_turb_data", methods=["POST"])
    def handle_turb_raw_data():
      try:
        data = self.config.turb_raw_dtype.from_json(request.data)
        res = self.__receive_turb_raw_data(data)
        if not res:
          return jsonify({"error": "Error writing to InfluxDB"}), 500
      except WindffRawDataParsingError as e:
        logging.warning(f"Error parsing data: {str(e)}")
        return jsonify({"error": str(e)}), 400

    self.server.run(host=self.config.listen_addr, port=self.config.listen_port)

  def start(self):
    self.__start_flask_server()
