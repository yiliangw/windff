import dgl
from dgl import DGLGraph
from dgl.data import DGLDataset
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod


class WindFFDataset(DGLDataset, ABC):

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

  @dataclass
  class Config:
    """

    Attributes:
      adj_weight_threshold (float): The threshold value in (0, 1) to accept edges in 
        the graph after normalization. A higher value means fewer edges (e^(-1) corresponds to the standard deviation of the distances).

    """
    data: 'WindFFDataset.Data'
    adj_weight_threshold: float
    input_win_sz: int
    output_win_sz: int

  def __init__(self, name: str = None):
    super(WindFFDataset, self).__init__(name=name)
    self.g: DGLGraph = None

  def process(self):
    config = self.get_config()
    self.__check_config(config)
    g = self.__create_graph(config)

    data = config.data

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
    g.ndata['feat_series'] = feat_series_tensor
    # (node_nb, time, target_dim)
    g.ndata['target_series'] = target_series_tensor

    datapoint_nb = feat_series_tensor.shape[1] - \
        config.input_win_sz - config.output_win_sz + 1

    # (node_nb, datapoint_nb, input_win_sz, feature_dim)
    g.ndata['feat'] = torch.stack([feat_series_tensor[:, i:i + config.input_win_sz]
                                   for i in range(datapoint_nb)], dim=1)
    # (node_nb, datapoint_nb, output_win_sz, target_dim)
    g.ndata['target'] = torch.stack([target_series_tensor[:, i + config.input_win_sz:i +
                                    config.input_win_sz + config.output_win_sz] for i in range(datapoint_nb)], dim=1)
    self.graph = g

  def __getitem__(self, idx):
    return self.graph

  def __len__(self):
    return 1

  @abstractmethod
  def get_config(self) -> Config:
    pass

  @classmethod
  def __check_config(cls, config: Config):

    data = config.data

    turbcol = data.turb_id_col
    timecol = data.time_col

    # turbine_location_df
    if turbcol not in data.turb_location_df.columns:
      raise ValueError()
    if len(data.turb_location_df.columns != 3):
      raise ValueError()

    # turbine_timeseries_df
    if not {turbcol, timecol, *data.turb_timeseries_target_cols} < set(config.turbine_timeseries_df.columns):
      raise ValueError()

    if config.adj_weight_threshold <= 0 or config.adj_weight_threshold >= 1:
      raise ValueError()

  @classmethod
  def __create_raw_graph(cls, config: Config) -> DGLGraph:
    df = config.data.turb_location_df
    idcol = config.data.turb_id_col
    nodes = df[idcol].unique()
    if len(nodes) != len(df):
      raise ValueError("turbine_location_df")

    coordcols = list(set(df.columns) - {idcol})
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
