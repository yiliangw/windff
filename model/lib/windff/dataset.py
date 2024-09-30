import dgl
from dgl import DGLGraph
from dgl.data import DGLDataset
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod


class WindFFGraph(DGLGraph):
  def __init__(self):
    super().__init__()

  def get_windowed_node_data_all(self, input_win_sz: int, output_win_sz: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = self
    feat = g.ndata['feat']
    target = g.ndata['target']
    data_nb = feat.shape[1] - input_win_sz - output_win_sz + 1

    # (node_nb, datapoint_nb, input_win_sz, feature_dim)
    win_feat = torch.stack([self.get_windowed_node_feature(
        g, i, input_win_sz) for i in range(data_nb)], dim=1)
    # (node_nb, datapoint_nb, output_win_sz, target_dim)
    win_target = torch.stack(
        [self.get_windowed_node_target(self, g, i + input_win_sz, output_win_sz) for i in range(data_nb)], dim=1)

    return win_feat, win_target

  def get_windowed_node_feature(self, win_start: int, input_win_sz: int) -> torch.Tensor:
    """Get the windowed node feature starting from win_start
    """
    g = self
    feat = g.ndata['feat']
    if win_start + input_win_sz > feat.shape[1]:
      raise ValueError("The requested window is out of bounds")
    win_feat = feat[:, win_start:win_start + input_win_sz]
    return win_feat

  def get_windowed_node_target(self, win_start: int, output_win_sz: int) -> torch.Tensor:
    """Get the windowed node target starting from win_start
    """
    g = self
    target = g.ndata['target']
    if win_start + output_win_sz > target.shape[1]:
      raise ValueError("The requested window is out of bounds")
    win_target = target[:, win_start:win_start + output_win_sz]
    return win_target

  def get_normalized_edge_weight(self, adj_weight_threshold: float = 0) -> torch.Tensor:
    """
    Parameters:
      adj_weight_threshold (float): The threshold value in (0, 1) to accept edges in 
        the graph after normalization. A higher value means fewer edges (e^(-1) corresponds to the standard deviation of the distances).
    """
    g = self
    dists = g.edata['dist']
    weights = np.exp(-np.square(dists / np.std(dists)))
    if adj_weight_threshold > 0:
      weights = np.where(weights > adj_weight_threshold, weights, 0)
    return torch.tensor(weights)


class WindFFDataset(DGLDataset, ABC):
  """General dataset for WindFF

  Attributes:
    g (WindFFGraph): The graph of the dataset. 
      g.ndata['feat'] stores the original node feature time series.
      g.ndata['target'] stores the original node target time series.
      g.edata['w'] store the edge weights.

  """

  @dataclass
  class Data:
    """Preprocessed data for WindFFDataset

    Attributes:
      turb_id_col: The values should have been preprocessed to index from 0
      time_col: The values should have been preprocessed to index from 0

      turb_timeseries_df: Should include time_col, turb_id_col and turb_timeseries_target_cols. If targets 
        are also used as features, they should be copied as other columns and preprocessed.
      turb_timeseries_target_cols: The target columns in the timeseries dataframe.

    """
    turb_id_col: str
    time_col: str

    turb_location_df: pd.DataFrame
    turb_timeseries_df: pd.DataFrame
    turb_timeseries_target_cols: list[str]

  def __init__(self, name: str = None):
    super().__init__(name=name)
    self.g: WindFFGraph = None

  def process(self):
    data = self.get_data()
    self.__check_config(data)
    g = self.__create_raw_graph(data)

    grouped_ts_df = data.turb_timeseries_df.groupby(data.turb_id_col)

    node_ts_dfs = [grouped_ts_df.get_group(node).sort_values(
        by=data.time_col, ascending=True) for node in g.nodes()]

    feat_cols = list(set(data.turb_timeseries_df.columns) -
                     {data.turb_id_col, data.time_col, *data.turb_timeseries_target_cols})

    feat_series_tensor = torch.tensor(
        [n_df[feat_cols].values for n_df in node_ts_dfs])
    target_series_tensor = torch.tensor([
        n_df[data.turb_timeseries_target_cols].values for n_df in node_ts_dfs])

    # (node_nb, time, feature_dim)
    g.ndata['feat'] = feat_series_tensor
    # (node_nb, time, target_dim)
    g.ndata['target'] = target_series_tensor

    self.graph = g

  def __getitem__(self, idx):
    return self.graph

  def __len__(self):
    return 1

  @abstractmethod
  def get_data(self) -> Data:
    pass

  @classmethod
  def __check_data(cls, data: Data):

    turbcol = data.turb_id_col
    timecol = data.time_col

    # turbine_location_df
    if turbcol not in data.turb_location_df.columns:
      raise ValueError()
    if len(data.turb_location_df.columns != 3):
      raise ValueError()

    # turbine_timeseries_df
    if not {turbcol, timecol, *data.turb_timeseries_target_cols} < set(data.turbine_timeseries_df.columns):
      raise ValueError()

  @classmethod
  def __create_raw_graph(cls, data: Data) -> WindFFGraph:
    df = data.turb_location_df
    idcol = data.turb_id_col
    nodes = df[idcol].unique()
    if len(nodes) != len(df):
      raise ValueError("turbine_location_df")

    coordcols = list(set(df.columns) - {idcol})
    if len(coordcols) != 2:
      raise ValueError("turbine_location_df")

    g = WindFFGraph()
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
