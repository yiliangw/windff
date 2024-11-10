import pandas as pd
import numpy as np
import random
import os

from ..data.raw import SDWPFRawTurbData


class SDWPFTurbineEdge:

  CSV_PATH = os.path.join(os.path.dirname(
      __file__), '../data/assets/timeseries_0.csv')
  TURB_NB = 134

  @classmethod
  def create_all(cls, time_start: np.datetime64, time_interval: np.timedelta64) -> list['SDWPFTurbineEdge']:

    df = pd.read_csv(cls.CSV_PATH)
    df['TurbID'] = df['TurbID'].astype(str)
    group = df.groupby('TurbID')

    edges = []
    for i in range(1, cls.TURB_NB + 1):
      edf = group.get_group(str(i)).reset_index(drop=True)
      edf = edf.drop(columns=['TurbID', 'Day', 'Tmstamp'])
      edf = edf.apply(pd.to_numeric, errors='coerce')
      edges.append(SDWPFTurbineEdge(str(i), group, time_start, time_interval))
    
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
    if idx >= len(self.data):
      return None
    row = self.df.iloc[idx]
    time = self.time_start + idx * self.time_interval + \
        self.time_interval * self.random.random()
    return SDWPFRawTurbData(time, self.id, row['wspd'], row['wdir'], row['etmp'], row['itmp'], row['ndir'], row['pab1'], row['pab2'], row['pab3'], row['prtv'], row['patv']).to_json()
