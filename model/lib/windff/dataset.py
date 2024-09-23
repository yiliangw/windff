import dgl
from dgl import DGLDataset, DGLGraph
from dataclasses import dataclass
import pandas as pd
import numpy as np

time_dict = {}  # {"00:00": 0, "00:10": 1, ..., "23:50": ...}
for i in range(0, 24):
  for j in range(0, 60, 10):
    time_dict[f'{i:02d}:{j:02d}'] = len(time_dict)


class WindFFDataset(DGLDataset):

  @dataclass
  class CreateArgs:
    """Arguments for WindFFDataset

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
    turbine_prediction_df: pd.DataFrame
    global_timeseries_df: pd.DataFrame
    global_prediction_df: pd.DataFrame
    adj_weight_threshold: float

  @classmethod
  def __check_create_args(cls, args: CreateArgs):
    turbcol = args.turbine_id_col
    timecol = args.time_col

    # turbine_location_df
    if turbcol not in args.turbine_location_df.columns:
      raise ValueError()
    if len(args.turbine_location_df.columns != 3):
      raise ValueError()

    # turbine_timeseries_df
    if not {turbcol, timecol, *args.turbine_timeseries_target_cols} < set(args.turbine_timeseries_df.columns):
      raise ValueError()

    # turbine_prediction_df
    if args.turbine_prediction_df is not None:
      if not {turbcol, timecol} < set(args.turbine_prediction_df.columns):
        raise ValueError()

    # global_timeseries_df
    if args.global_timeseries_df is not None:
      if not {timecol} < set(args.global_timeseries_df.columns):
        raise ValueError()
      # global_prediction_df
      if args.global_prediction_df is not None:
        if not {timecol} < set(args.global_prediction_df.columns):
          raise ValueError()
        if not set(args.global_prediction_df.columns) < set(args.global_timeseries_df.columns):
          raise ValueError()

    if args.adj_weight_threshold <= 0 or args.adj_weight_threshold >= 1:
      raise ValueError()

  @classmethod
  def __create_raw_graph(cls, args: CreateArgs) -> DGLGraph:
    df = args.turbine_location_df
    idcol = args.turbine_id_col
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
  def create(cls, args: CreateArgs) -> 'WindFFDataset':
    """Create a WindFFDataset from dataframes
    The user can free the dataframes after calling this function
    """
    cls.__check_create_args(args)

    graph = cls.__create_raw_graph(args)
    dists = graph.edata['dist']
    weights = np.exp(-np.square(dists / np.std(dists)))
    weights = np.where(weights > args.adj_weight_threshold, weights, 0)
    graph.edata['w'] = weights
    del graph.edata['dist']

  def __init__(self):
    super(WindFFDataset, self).__init__(name='windff')

  def process(self):
    pass

  def __getitem__(self, idx):
    return self.graph

  def __len__(self):
    return 1
