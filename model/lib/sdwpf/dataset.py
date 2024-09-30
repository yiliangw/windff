from ..windff import WindFFDataset
import os
import pandas as pd
from dataclasses import dataclass


class SDWPFDataset(WindFFDataset):

  COMPRESSED_ASSETS = os.path.join(os.path.dirname(__file__), "assets.tar.bz2")
  ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
  LOCATION_CSV = os.path.join(ASSETS_DIR, "location.csv")

  TIMESERIES_CSV_LIST = [os.path.join(
      SDWPFDataset.ASSETS_DIR, f"timeseries_{i}.csv") for i in range(3)]

  TARGETS = ['Patv']

  # {"00:00": 0, "00:10": 1, ..., "23:50": ...}
  time_dict = {}

  @ dataclass
  class Config:
    discard_features: list[str] = []

  @ classmethod
  def initialize(cls):
    for i in range(0, 24):
      for j in range(0, 60, 10):
        cls.time_dict[f'{i:02d}:{j:02d}'] = len(cls.time_dict)

  def __init__(self, config: Config = None):
    super().__init__("SDWPF")
    if config is None:
      self.config = SDWPFDataset.Config()
    else:
      self.config = config.copy()

  def __len__(self):
    return len(self.TIMESERIES_CSV_LIST)

  def get_data(self, idx):
    if not os.path.exists(self.ASSETS_DIR):
      import tarfile
      with tarfile.open(self.COMPRESSED_ASSETS, "r:bz2") as tar:
        tar.extractall(os.path.dirname(__file__))

    data = self.__preprocess(
        raw_loc_df=pd.read_csv(self.LOCATION_CSV),
        raw_ts_df=pd.read_csv(self.TIMESERIES_CSV_LIST[idx])
    )
    return data

  def __preprocess(self, raw_loc_df: pd.DataFrame, raw_ts_df: pd.DataFrame) -> WindFFDataset.Data:

    # Location
    loc_df = pd.DataFrame()
    loc_df['TurbID'] = raw_loc_df['TurbID'].astype(
        pd.UInt32Dtype) - 1
    loc_df['x'] = raw_loc_df['x'].astype(pd.Float32Dtype)
    loc_df['y'] = raw_loc_df['y'].astype(pd.Float32Dtype)

    # Timeseries
    ts_df = pd.DataFrame()

    # Turbine ID
    ts_df['TurbID'] = raw_ts_df['TurbID'].astype(pd.UInt32Dtype) - 1

    # Time
    days = raw_ts_df['Day'].astype(pd.UInt32Dtype)
    tmstamp = raw_ts_df['Tmstamp'].astype(pd.StringDtype)
    ts_df['Time'] = tmstamp.map(self.time_dict) + \
        (days - days.min()) * len(self.time_dict)

    # Targets and features
    for t in self.TARGETS:
      ts_df[t] = raw_ts_df[t].apply(pd.to_numeric, errors='coerce')
      # Also copy the targets features and do preprocessing later
      ts_df[f'{t}_feat'] = ts_df[t]

    for f in raw_ts_df.columns:
      if f not in {'TurbID', 'Day', 'Tmstamp', *self.config.discard_features, *self.TARGETS}:
        ts_df[f] = raw_ts_df[f].apply(pd.to_numeric, errors='coerce')

    # Interpolate missing values
    ts_df.interpolate(method='linear', inplace=True)

    return WindFFDataset.Data(
        turb_id_col='TurbID',
        time_col='Time',

        turb_location_df=loc_df,
        turb_timeseries_df=ts_df,
        turb_timeseries_target_cols=self.TARGETS.copy()
    )


SDWPFDataset.initialize()
