from ...windff.data import Dataset
import os
import pandas as pd
from dataclasses import dataclass
import torch


class SDWPFDataset(Dataset):

  COMPRESSED_ASSETS = os.path.join(os.path.dirname(__file__), "assets.tar.bz2")
  ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
  LOCATION_CSV = os.path.join(ASSETS_DIR, "location.csv")

  TIMESERIES_CSV_LIST: list[str] = []

  TARGETS = ['Patv']

  # {"00:00": 0, "00:10": 1, ..., "23:50": ...}
  time_dict = {}

  @ dataclass
  class Config:
    discard_features: list[str] = None

  @ classmethod
  def initialize(cls):
    for i in range(0, 24):
      for j in range(0, 60, 10):
        cls.time_dict[f'{i:02d}:{j:02d}'] = len(cls.time_dict)

    cls.TIMESERIES_CSV_LIST = [os.path.join(
        cls.ASSETS_DIR, f"timeseries_{i}.csv") for i in range(1, 3)]

  def __init__(self, config: Config = None):
    if config is None:
      self.config = SDWPFDataset.Config()
    else:
      self.config = config.copy()

    if self.config.discard_features is None:
      self.config.discard_features = []

    super().__init__("SDWPF")

  def __len__(self):
    return len(self.TIMESERIES_CSV_LIST)

  def _get_raw_data(self, idx):
    if not os.path.exists(self.ASSETS_DIR):
      import tarfile
      with tarfile.open(self.COMPRESSED_ASSETS, "r:bz2") as tar:
        tar.extractall(os.path.dirname(__file__))

    data = self.__preprocess(
        raw_loc_df=pd.read_csv(self.LOCATION_CSV),
        raw_ts_df=pd.read_csv(self.TIMESERIES_CSV_LIST[idx])
    )
    return data

  def _get_data_cache_path(self, idx):
    return os.path.join(self.ASSETS_DIR, f"cache/timeseries_{idx}.pkl")

  def _get_metadata_cache_path(self) -> str:
    return os.path.join(self.ASSETS_DIR, "cache/metadata.pkl")

  def _get_tensor_dtype(self):
    return torch.float64

  def __preprocess(self, raw_loc_df: pd.DataFrame, raw_ts_df: pd.DataFrame) -> Dataset.RawData:

    # Location
    loc_df = pd.DataFrame()
    loc_df['TurbID'] = raw_loc_df['TurbID'].astype(str)
    loc_df['x'] = raw_loc_df['x'].astype(pd.Float32Dtype())
    loc_df['y'] = raw_loc_df['y'].astype(pd.Float32Dtype())

    # Timeseries
    ts_df = pd.DataFrame()

    # Turbine ID
    ts_df['TurbID'] = raw_ts_df['TurbID'].astype(str)

    # Time
    days = raw_ts_df['Day'].astype(pd.UInt32Dtype())
    tmstamp = raw_ts_df['Tmstamp'].astype(pd.StringDtype())
    ts_df['Time'] = tmstamp.map(self.time_dict) + \
        (days - days.min()) * len(self.time_dict)

    for f in raw_ts_df.columns:
      if f not in {'TurbID', 'Day', 'Tmstamp', *self.config.discard_features}:
        ts_df[f] = raw_ts_df[f].apply(pd.to_numeric, errors='coerce')

    # Interpolate missing values
    ts_df.infer_objects(copy=False).interpolate(method='linear', inplace=True)
    ts_df.ffill(inplace=True)
    ts_df.bfill(inplace=True)

    return Dataset.RawData(
        turb_id_col='TurbID',
        time_col='Time',

        turb_location_df=loc_df,
        turb_timeseries_df=ts_df,
        turb_timeseries_target_cols=self.TARGETS.copy()
    )


SDWPFDataset.initialize()
