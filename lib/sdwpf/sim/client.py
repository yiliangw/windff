import numpy as np

import logging


class SDWPFClient:

  logger = logging.getLogger(__qualname__)

  def __init__(self):
    pass

  def create_query(self, time_start: np.datetime64, time_stop: np.datetime64) -> dict:
    query = {
        'time_start': np.datetime_as_string(time_start, unit='s'),
        'time_stop': np.datetime_as_string(time_stop, unit='s')
    }
    return query
