import pandas as pd
import numpy as np
import random
import os

import logging

from ..data.raw import SDWPFRawTurbData


class SDWPFTurbineEdge:

  logger = logging.getLogger(__qualname__)

  CSV_PATH = os.path.join(os.path.dirname(
      __file__), '../data/assets/timeseries_2.csv')
  TURB_NB = 134

  @classmethod
  def create_all(cls, time_start: np.datetime64, time_interval: np.timedelta64) -> list['SDWPFTurbineEdge']:

    df = pd.read_csv(cls.CSV_PATH)
    df['TurbID'] = df['TurbID'].astype(str)
    group = df.groupby('TurbID')

    edges = []
    for i in range(1, cls.TURB_NB + 1):
      turb_df = group.get_group(str(i))
      turb_df = turb_df.reset_index(drop=True)
      turb_df = turb_df.drop(columns=['TurbID', 'Day', 'Tmstamp'])
      turb_df = turb_df.apply(pd.to_numeric, errors='coerce')
      edges.append(SDWPFTurbineEdge(
          str(i), turb_df, time_start, time_interval))

    return edges

  @classmethod
  def get_turb_nb(cls) -> int:
    return cls.TURB_NB

  def __init__(self, id: str, df: pd.DataFrame, time_start: np.datetime64, time_interval: np.timedelta64):
    self.id = id
    self.df = df
    self.time_start = time_start
    self.time_interval = time_interval
    self.random = random.Random(self.id)

  def get_raw_data_json(self, idx: int) -> str:
    if idx >= len(self.df):
      return None
    row = self.df.iloc[idx]
    time = self.time_start + idx * self.time_interval + \
        self.time_interval * self.random.random()

    json_str = SDWPFRawTurbData(time, self.id, row['Wspd'], row['Wdir'], row['Etmp'], row['Itmp'],
                                row['Ndir'], row['Pab1'], row['Pab2'], row['Pab3'], row['Prtv'], row['Patv']).to_json()

    self.logger.info(f"[Turbine {self.id}]: {json_str}")

    return json_str
