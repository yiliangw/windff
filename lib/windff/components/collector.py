import logging

from .component import Component
from ..data.raw import RawTurbData
from ..data.database import DatabaseID
from ..errors import DBError, RawDataParsingError


class Collector(Component):

  @classmethod
  def get_type(cls):
    return 'collector'

  def __init__(self):
    from ..env import Env
    self.env: Env = None

  def initialize(self, env):
    logging.info("Initializing collector...")
    self.env = env
    self.env.connect_db(DatabaseID.RAW)
    logging.info("Collector initialized.")

  def handle_raw_turb_data(self, data: RawTurbData):
    '''
    @throws RawDataParsingError: If the raw turbine data JSON is faulty
    @throws DBError: If there is a DB error
    '''
    self.env.write_raw_turb_data(data)
