from flask import Flask, request, jsonify

import logging

from .component import Component
from ..data.raw import RawTurbData
from ..errors import DBError, RawDataParsingError


class Collector(Component):

  @classmethod
  def get_type(cls):
    return 'collector'

  def __init__(self):
    from ..env import WindffEnv
    self.env: WindffEnv = None

    self.server: Flask = None

  def initialize(self, env):
    logging.info("Initializing collector...")
    self.env = env
    logging.info("Collector initialized.")

  def handle_raw_turb_data_json(self, data_json: str):
    '''
    @throws RawDataParsingError: If the raw turbine data JSON is faulty
    @throws DBError: If there is a DB error
    '''
    data = self.env.parse_raw_turb_data(data_json)
    self.env.write_raw_turb_data(data)

  def __start_flask_server(self):
    self.server = Flask(__name__)

    @self.server.route("/raw_turb_data", methods=["POST"])
    def flask_handle_raw_turb_data():
      try:
        data_json = request.json
        self.handle_raw_turb_data_json(data_json)
        return jsonify({"status": "OK"}), 200
      except RawDataParsingError as e:
        logging.warning(
            f"Discarded faulty raw turbine data: {data_json}, err: {str(e)}")
        return jsonify({"error": str(e)}), 400
      except DBError as e:
        logging.warning(f"Ignored raw turbine data due to DB error: {str(e)}")
        return jsonify({"error": "DB error"}), 500

    self.server.run(host=self.config.listen_addr, port=self.config.listen_port)

  def start(self):
    self.__start_flask_server()
