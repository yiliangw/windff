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

  # {"00:00": 0, "00:10": 1, ..., "23:50": ...}
  time_dict = {}

  @dataclass
  class Config:
    input_win_sz: int
    output_win_sz: int
    adj_weight_threshold: float

  @classmethod
  def initialize(cls):
    for i in range(0, 24):
      for j in range(0, 60, 10):
        cls.time_dict[f'{i:02d}:{j:02d}'] = len(cls.time_dict)

  def __init__(self, config: Config):
    super(SDWPFDataset, self).__init__("SDWPF")
    self.config = config

  def process(self):

    if not os.path.exists(self.ASSETS_DIR):
      import tarfile
      with tarfile.open(self.COMPRESSED_ASSETS, "r:bz2") as tar:
        tar.extractall(os.path.dirname(__file__))

    turbine_localtion_df = pd.read_csv(self.LOCATION_CSV)
    timeseries_df = pd.read_csv(self.TIMESERIES_CSV)

    days = timeseries_df['Day'].astype(pd.UInt32Dtype)
    tmstamp = timeseries_df['Tmstamp'].astype(pd.StringDtype)
    timeseries_df['Time'] = tmstamp.map(self.time_dict) + \
        (days - days.min()) * len(self.time_dict)
    timeseries_df = timeseries_df.drop(columns=['Day', 'Tmstamp'])

    args = WindFFDataset.Arguments(
        turbine_id_col='TurbID',
        time_col='Time',

        turbine_location_df=turbine_localtion_df,
        turbine_timeseries_df=timeseries_df,
        turbine_timeseries_target_cols=['Patv']

        adj_weight_threshold=self.config.adj_weight_threshold,
        input_win_sz=self.config.input_win_sz,
        output_win_sz=self.config.output_win_sz
    )

    self._do_process(args)


SDWPFDataset.initialize()
