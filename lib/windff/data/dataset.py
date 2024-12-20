import dgl
from dgl import DGLGraph
from dgl.data import DGLDataset
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
import logging
from abc import ABC, abstractmethod


class Graph(object):
  def __init__(self, nodes: list[str], edges: list[tuple[int, int, float]], feat: torch.Tensor, target: torch.Tensor, dtype: torch.dtype):
    """
    Args:
        edges (list[tuple[int, int, float]]): (u, v, dist) for every edge

    """
    if torch.isnan(feat).any():
      raise ValueError("Input feature contains NaN")

    self._edges = edges
    self._nodes = nodes

    u_list = torch.tensor([u for u, _, _ in edges], dtype=torch.int64)
    v_list = torch.tensor([v for _, v, _ in edges], dtype=torch.int64)
    dist_list = torch.tensor(
        [dist for _, _, dist in edges])

    if torch.isnan(dist_list).any():
      raise ValueError("Edge distance contains NaN")

    if max(u_list) >= len(nodes) or max(v_list) >= len(nodes) or min(u_list) < 0 or min(v_list) < 0:
      raise ValueError("The edge contains nodes that are out of bounds")
    if len(feat) != len(nodes):
      raise ValueError(
          "The number of nodes in the feature tensor is not consistent")
    if len(target) != len(nodes):
      raise ValueError(
          "The number of nodes in the target tensor is not consistent")

    self.g = dgl.graph((u_list, v_list), num_nodes=len(nodes))

    self.__node_feat = feat.to(dtype)
    self.__node_target = target.to(dtype)
    self.__edge_dist = dist_list.to(dtype)

  def get_windowed_node_data_all(self, input_win_sz: int, output_win_sz: int) -> tuple[torch.Tensor, torch.Tensor]:
    feat = self.get_node_feat()
    target = self.get_node_target()
    data_nb = feat.shape[1] - input_win_sz - output_win_sz + 1

    # (datapoint_nb, node_nb, input_win_sz, feat_dim)
    win_feat = torch.stack([self.get_windowed_node_feat(
        i, input_win_sz) for i in range(data_nb)], dim=0)
    # (datapoint_nb, node_nb, output_win_sz, target_dim)
    win_target = torch.stack(
        [self.get_windowed_node_target(i + input_win_sz, output_win_sz) for i in range(data_nb)], dim=0)

    return win_feat, win_target

  def get_last_windowed_node_feat(self, input_win_sz: int) -> torch.Tensor:
    feat = self.get_node_feat()
    if input_win_sz > feat.shape[1]:
      raise ValueError("The requested window is out of bounds")
    # (node_nb, input_win_sz, feat_dim)
    win_feat = feat[:, -input_win_sz:]
    return win_feat

  def get_windowed_node_feat(self, win_start: int, input_win_sz: int) -> torch.Tensor:
    feat = self.get_node_feat()
    if win_start + input_win_sz > feat.shape[1]:
      raise ValueError("The requested window is out of bounds")
    # (node_nb, input_win_sz, feat_dim)
    win_feat = feat[:, win_start:win_start + input_win_sz]
    return win_feat

  def get_windowed_node_target(self, win_start: int, output_win_sz: int) -> torch.Tensor:
    """Get the windowed node target starting from win_start
    """
    target = self.get_node_target()
    if win_start + output_win_sz > target.shape[1]:
      raise ValueError("The requested window is out of bounds")
    # (node_nb, output_win_sz, target_dim)
    win_target = target[:, win_start:win_start + output_win_sz]
    return win_target

  def get_normalized_edge_weight(self, adj_weight_threshold: float = 0) -> torch.Tensor:
    """
    Parameters:
      adj_weight_threshold (float): The threshold value in (0, 1) to accept edges in
        the graph after normalization. A higher value means fewer edges (e^(-1) corresponds to the standard deviation of the distances).
    """
    dists = self.get_edge_dist().numpy()
    weights = np.exp(-np.square(dists / np.std(dists)))
    if adj_weight_threshold > 0:
      weights = np.where(weights > adj_weight_threshold, weights, 0)
    return torch.tensor(weights)

  @property
  def dgl_graph(self) -> DGLGraph:
    return self.g

  def get_node_feat(self) -> torch.Tensor:
    return self.__node_feat

  def get_node_target(self) -> torch.Tensor:
    return self.__node_target

  def get_edge_dist(self) -> torch.Tensor:
    return self.__edge_dist

  @property
  def nodes(self):
    return self._nodes

  @property
  def edges(self):
    return self._edges


class Dataset(DGLDataset, ABC):
  """General dataset for Windff

  Attributes:
    g (WindffGraph): The graph of the dataset.
      g.ndata['feat'] stores the original node feature time series.
      g.ndata['target'] stores the original node target time series.
      g.edata['w'] store the edge weights.

  For efficiency, we do processing, loading and saving in __getitem__().

  """

  DEFAULT_TENSOR_DTYPE = torch.float64

  @dataclass
  class RawData:
    """Preprocessed data for WindffDataset

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
    self.graph_list: list[Graph] = []
    self.meta = {
        "feat_dim": 0,
        "target_dim": 0
    }
    super().__init__(name=name)

  def has_cache(self):
    meta_cache = self._get_metadata_cache_path()
    if meta_cache is None:
      return False
    else:
      import os
      return os.path.exists(meta_cache)

  def process(self):
    g = self[0]
    self.meta["feat_dim"] = g.get_node_feat().shape[2]
    self.meta["target_dim"] = g.get_node_target().shape[2]

  def save(self):
    meta_cache = self._get_metadata_cache_path()
    if meta_cache is None:
      return
    logging.info(f"Saving metadata to {meta_cache}")
    import os
    import pickle
    os.makedirs(os.path.dirname(
        self._get_metadata_cache_path()), exist_ok=True)
    with open(self._get_metadata_cache_path(), "wb") as f:
      pickle.dump(self.meta, f)

  def load(self):
    meta_cache = self._get_metadata_cache_path()
    logging.info(f"Loading metadata from {meta_cache}")
    import pickle
    self.meta = pickle.load(open(meta_cache, "rb"))

  def __getitem__(self, idx) -> Graph:
    if (idx >= len(self)):
      raise IndexError("Index out of bounds")

    cache_pkl = self._get_data_cache_path(idx)
    if cache_pkl is None:
      return self.__process_idx(idx)

    import os
    import pickle
    if os.path.exists(cache_pkl):
      logging.info(f"Loading data [{idx}] from {cache_pkl}")
      with open(cache_pkl, "rb") as f:
        return pickle.load(f)
    else:
      g = self.__process_idx(idx)
      logging.info(f"Saving data [{idx}] to {cache_pkl}")
      os.makedirs(os.path.dirname(cache_pkl), exist_ok=True)
      with open(cache_pkl, "wb") as f:
        pickle.dump(g, f)
      return g

  @abstractmethod
  def __len__(self):
    pass

  @classmethod
  def _get_tensor_dtype(cls) -> torch.dtype:
    return cls.DEFAULT_TENSOR_DTYPE

  @abstractmethod
  def _get_raw_data(self, idx) -> RawData:
    pass

  def _get_data_cache_path(self, idx) -> str:
    return None

  def _get_metadata_cache_path(self) -> str:
    return None

  def get_feat_dim(self) -> int:
    return self.meta["feat_dim"]

  def get_target_dim(self) -> int:
    return self.meta["target_dim"]

  def __process_idx(self, idx: int) -> Graph:
    data = self._get_raw_data(idx)
    return self.process_raw_data(data, dtype=self._get_tensor_dtype())

  @classmethod
  def process_raw_data(cls, data: RawData, dtype=DEFAULT_TENSOR_DTYPE) -> Graph:

    cls.__check_raw_data(data)
    nodes, edges = cls.__get_topology(data)

    turb_ts_df = data.turb_timeseries_df
    # Also copy targets as features for later preprocessing
    for col in data.turb_timeseries_target_cols:
      turb_ts_df[f'{col}_feat'] = turb_ts_df[col]

    grouped_ts_df = data.turb_timeseries_df.groupby(data.turb_id_col)

    node_ts_dfs = [grouped_ts_df.get_group(node).sort_values(
        by=data.time_col, ascending=True) for node in nodes]

    feat_cols = list(set(data.turb_timeseries_df.columns) -
                     {data.turb_id_col, data.time_col, *data.turb_timeseries_target_cols})

    # (node_nb, time, feat_dim)
    feat_series_tensor = torch.tensor(
        np.array([n_df[feat_cols].values for n_df in node_ts_dfs]))
    # (node_nb, time, target_dim)
    target_series_tensor = torch.tensor(
        np.array([
            n_df[data.turb_timeseries_target_cols].values for n_df in node_ts_dfs]))

    return Graph(nodes, edges, feat_series_tensor, target_series_tensor, dtype=dtype)

  @classmethod
  def __check_raw_data(cls, data: RawData):

    turbcol = data.turb_id_col
    timecol = data.time_col

    nodes = data.turb_location_df[turbcol].unique()
    if len(nodes) != len(data.turb_location_df):
      raise ValueError("Node ID")

    # turbine_location_df
    if turbcol not in data.turb_location_df.columns:
      raise ValueError()
    if len(data.turb_location_df.columns) != 3:
      raise ValueError()
    if data.turb_location_df.isnull().values.any():
      raise ValueError("turbine_location_df has NaN values")

    # turbine_timeseries_df
    if not {turbcol, timecol, *data.turb_timeseries_target_cols} < set(data.turb_timeseries_df.columns):
      raise ValueError()
    if data.turb_timeseries_df.isnull().values.any():
      raise ValueError("turbine_timeseries_df has NaN values")

  @classmethod
  def __get_topology(cls, data: RawData) -> tuple[int, list[tuple[int, int, float]]]:
    df = data.turb_location_df
    idcol = data.turb_id_col
    if len(df[idcol].unique()) != len(df):
      raise ValueError("turbine_location_df")

    coordcols = list(set(df.columns) - {idcol})
    if len(coordcols) != 2:
      raise ValueError("turbine_location_df")

    edges = []
    nodes = [df[idcol][i] for i in range(len(df))]

    for i in range(len(df)):
      for j in range(i + 1, len(df)):

        coordi = df.iloc[i][coordcols]
        coordj = df.iloc[j][coordcols]
        dist = np.sqrt(np.sum(np.square(coordi - coordj)))

        edges.append((i, j, dist))
        edges.append((j, i, dist))

    return nodes, edges
