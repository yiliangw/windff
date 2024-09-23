import dgl
from dgl import DGLDataset, DGLGraph
from dataclasses import dataclass
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

time_dict = {}  # {"00:00": 0, "00:10": 1, ..., "23:50": ...}
for i in range(0, 24):
  for j in range(0, 60, 10):
    time_dict[f'{i:02d}:{j:02d}'] = len(time_dict)


class WindFFDataset(DGLDataset, ABC):

  @dataclass
  class Config:
    """Configuration for WindFFDataset

    Attributes:
      turbine_id_col: The values should have been preprocessed to index from 0
      time_col: The values should have been preprocessed to index from 0

      turbine_timeseries_df: Should include time_col, turbine_id_col
      turbine_timeseries_target_cols: Target features to predict in turbine_timeseries_df 

      turbine_prediction_df (optional): Predicted values for some of turbine_timeseries_df.columns

      global_timeseries_df (optional): Global timeseries information 
      global_prediction_df (optional): The columns should be a subset of global_timeseries_df's

      adj_weight_threshold (float): The threshold value in (0, 1) to accept edges in 
        the graph after normalization. A higher value means fewer edges (e^(-1) corresponds to the standard deviation of the distances).
    """
    turbine_id_col: str
    time_col: str
    turbine_location_df: pd.DataFrame
    turbine_timeseries_df: pd.DataFrame
    turbine_timeseries_target_cols: list[str]
    turbine_prediction_df: pd.DataFrame = None
    global_timeseries_df: pd.DataFrame = None
    global_prediction_df: pd.DataFrame = None
    adj_weight_threshold: float
    input_win_sz: int
    output_win_sz: int

  @classmethod
  def __check_config(cls, config: Config):
    turbcol = config.turbine_id_col
    timecol = config.time_col

    # turbine_location_df
    if turbcol not in config.turbine_location_df.columns:
      raise ValueError()
    if len(config.turbine_location_df.columns != 3):
      raise ValueError()

    # turbine_timeseries_df
    if not {turbcol, timecol, *config.turbine_timeseries_target_cols} < set(config.turbine_timeseries_df.columns):
      raise ValueError()

    # turbine_prediction_df
    if config.turbine_prediction_df is not None:
      if not {turbcol, timecol} < set(config.turbine_prediction_df.columns):
        raise ValueError()

    # global_timeseries_df
    if config.global_timeseries_df is not None:
      if not {timecol} < set(config.global_timeseries_df.columns):
        raise ValueError()
      # global_prediction_df
      if config.global_prediction_df is not None:
        if not {timecol} < set(config.global_prediction_df.columns):
          raise ValueError()
        if not set(config.global_prediction_df.columns) < set(config.global_timeseries_df.columns):
          raise ValueError()

    if config.adj_weight_threshold <= 0 or config.adj_weight_threshold >= 1:
      raise ValueError()

  @classmethod
  def __create_raw_graph(cls, config: Config) -> DGLGraph:
    df = config.turbine_location_df
    idcol = config.turbine_id_col
    nodes = df[idcol].unique()
    if len(nodes) != len(df):
      raise ValueError("turbine_location_df")

    coordcols = list(df.columns - {idcol})
    if len(coordcols) != 2:
      raise ValueError("turbine_location_df")

    g = dgl.graph()
    g.add_nodes(nodes)
    for i in range(len(df)):
      for j in range(i, len(df)):
        ni = df.iloc[i][idcol]
        nj = df.iloc[j][idcol]

        coordi = df.iloc[i][coordcols]
        coordj = df.iloc[j][coordcols]
        dist = np.sqrt(np.sum(np.square(coordi - coordj)))

        g.add_edge(ni, nj, {'dist': dist})
        g.add_edge(nj, ni, {'dist': dist})

    return g

  @classmethod
  def __create_graph(cls, config: Config) -> DGLGraph:
    g = cls.__create_raw_graph(config)
    dists = g.edata['dist']
    weights = np.exp(-np.square(dists / np.std(dists)))
    g.edata['w'] = np.where(
        weights > config.adj_weight_threshold, weights, 0)
    g.remove_edges(g.edata['w'] == 0)
    del g.edata['dist']
    return g

  def __init__(self, name: str = None):
    super(WindFFDataset, self).__init__(name=name)

  def _do_process(self, config: Config):
    """_do_process
    The subclass can free the dataframes in args after calling this function
    """
    self.__check_config(config)
    g = self.__create_graph(config)

  @abstractmethod
  def process(self):
    pass

  def __getitem__(self, idx):
    return self.graph

  def __len__(self):
    return 1
