import logging
import json
import numpy as np
from datetime import datetime

from .. import Config
from .component import Component
from ..data.database import DatabaseID
from ..utils import Utils


class Broadcaster(Component):

  logger = logging.getLogger(__qualname__)

  @classmethod
  def get_type(cls) -> str:
    return 'broadcaster'

  def __init__(self):
    from ..env import Env
    self.env: Env = None

  def initialize(self, env):
    self.logger.info("Initializing broadcaster...")
    self.env = env
    self.env.connect_db(DatabaseID.PREDICTED)
    self.logger.info("Broadcaster initialized.")

  def handle_query(self, query: dict) -> dict:
    '''
    @throw ValueError: If the query JSON is invalid
    '''
    if query['time_start'] is None:
      raise ValueError("Query JSON missing field: time_start")
    if query['time_stop'] is None:
      raise ValueError("Query JSON missing field: time_stop")

    self.logger.info("Handling query: %s", json.dumps(query))
    time_start = np.datetime64(query['time_start'])
    time_stop = np.datetime64(query['time_stop'])

    df = self.env.query_predicted_data_df(time_start, time_stop)

    df[self.env.time_col] = df[self.env.time_col].dt.tz_localize(None).apply(
        lambda x: x.isoformat())

    resp = df.to_dict(orient='records')
    self.logger.info("Respond to query: %s", json.dumps(resp))

    return resp
