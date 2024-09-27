from ..windff import WindFFDataset
import os
import pandas as pd
from dataclasses import dataclass


class SDWPFDataset(WindFFDataset):

  COMPRESSED_ASSETS = os.path.join(os.path.dirname(__file__), "assets.tar.bz2")
  ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
  LOCATION_CSV = os.path.join(ASSETS_DIR, "location.csv")
  TIMESERIES_CSV = os.path.join(ASSETS_DIR, "train.csv")

  TEST_X_CSV = os.path.join(ASSETS_DIR, "test_x.csv")
  TEST_Y_CSV = os.path.join(ASSETS_DIR, "test_y.csv")

  TARGETS = ['Patv']

  # {"00:00": 0, "00:10": 1, ..., "23:50": ...}
  time_dict = {}

  @dataclass
  class Config:
    input_win_sz: int
    output_win_sz: int
    adj_weight_threshold: float
    discard_features: list[str] = None

  @dataclass
  class Data:
    location_df: pd.DataFrame
    timeseries_df: pd.DataFrame

  @classmethod
  def initialize(cls):
    for i in range(0, 24):
      for j in range(0, 60, 10):
        cls.time_dict[f'{i:02d}:{j:02d}'] = len(cls.time_dict)

  def __init__(self, config: Config):
    super(SDWPFDataset, self).__init__("SDWPF")
    self.config = config.copy()
    if self.config.discard_features is None:
      self.config.discard_features = []

  def process(self):

    if not os.path.exists(self.ASSETS_DIR):
      import tarfile
      with tarfile.open(self.COMPRESSED_ASSETS, "r:bz2") as tar:
        tar.extractall(os.path.dirname(__file__))

    data = self.Data(
        location_df=pd.read_csv(self.LOCATION_CSV),
        timeseries_df=pd.read_csv(self.TIMESERIES_CSV)
    )

    data = self.__preprocess(data)

    args = WindFFDataset.Arguments(
        turbine_id_col='TurbID',
        time_col='Time',

        turbine_location_df=data.turbine_location_df,
        turbine_timeseries_df=data.timeseries_df,
        turbine_timeseries_target_cols=self.TARGETS,

        adj_weight_threshold=self.config.adj_weight_threshold,
        input_win_sz=self.config.input_win_sz,
        output_win_sz=self.config.output_win_sz
    )

    self._do_process(args)

  def __preprocess(self, dfs: Data) -> Data:
    # Location
    loc_df = pd.DataFrame()
    loc_df['TurbID'] = dfs.location_df['TurbID'].astype(pd.UInt32Dtype) - 1
    loc_df['x'] = dfs.location_df['x'].astype(pd.Float32Dtype)
    loc_df['y'] = dfs.location_df['y'].astype(pd.Float32Dtype)

    # Timeseries
    ts_df = pd.DataFrame()

    # Turbine ID
    ts_df['TurbID'] = dfs.timeseries_df['TurbID'].astype(pd.UInt32Dtype) - 1

    # Time
    days = dfs.timeseries_df['Day'].astype(pd.UInt32Dtype)
    tmstamp = dfs.timeseries_df['Tmstamp'].astype(pd.StringDtype)
    ts_df['Time'] = tmstamp.map(self.time_dict) + \
        (days - days.min()) * len(self.time_dict)

    # Targets and features
    for t in self.TARGETS:
      ts_df[t] = dfs.timeseries_df[t].apply(pd.to_numeric, errors='coerce')
      # Also copy the targets features and do preprocessing later
      ts_df[f'{t}_feat'] = ts_df[t]

    for f in dfs.timeseries_df.columns:
      if f not in {'TurbID', 'Day', 'Tmstamp', *self.config.discard_features, *self.TARGETS}:
        ts_df[f] = dfs.timeseries_df[f].apply(pd.to_numeric, errors='coerce')

    # Interpolate missing values
    ts_df.interpolate(method='linear', inplace=True)

    # # Normalize features
    # for f in ts_df.columns:
    #   if f not in {'TurbID', 'Time', *self.TARGETS}:
    #     ts_df[f] = (ts_df[f] - ts_df[f].mean()) / ts_df[f].std()

    return self.Data(location_df=loc_df, timeseries_df=ts_df)


SDWPFDataset.initialize()
