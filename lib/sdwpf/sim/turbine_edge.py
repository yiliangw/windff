import pandas as pd
import numpy as np
import random

from ..data import SDWPFRawTurbData


class SDWPFTurbineEdge:

  @classmethod
  def create_all(cls) -> list['SDWPFTurbineEdge']:
    all = []
    return all

  def __init__(self, id: str, data: pd.DataFrame, time_start: np.datetime64, time_interval: np.timedelta64):
    self.id = id
    self.data = data
    self.time_start = time_start
    self.time_interval = time_interval
    self.random = random.Random(self.id)

  def get_raw_data_json(self, idx: int) -> str:
    if idx >= len(self.data):
      return None
    row = self.data.iloc[idx]
    time = self.time_start + idx * self.time_interval + \
        self.time_interval * self.random.random()
    return SDWPFRawTurbData(time, self.id, row['wspd'], row['wdir'], row['etmp'], row['itmp'], row['ndir'], row['pab1'], row['pab2'], row['pab3'], row['prtv'], row['patv']).to_json()
