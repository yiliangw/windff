from dataclasses import dataclass

from .dataset import WindFFDataset
from .model import WindFFModel


class WindFFModelManager:
  def __init__(self, config: WindFFModel.Config):
    self.config = config
    self.model: WindFFModel = WindFFModel(config)

  @dataclass
  class TrainArgs:
    dataset: WindFFDataset
    epochs: int
    lr: float
    batch_sz: int

  def train(self, dataset: WindFFDataset):
    '''Train a model from scratch
    '''
    self.__check_dataset(dataset)

  def update(self):
    pass

  def infer(self):
    pass

  def __check_dataset(self, dataset: WindFFDataset):

    g = dataset[0]
    feat = g.ndata['feat']  # (N, DATAPOINT_NB, INPUT_WIN_SZ, FEAT_DIM)
    target = g.ndata['target']  # (N, DATAPOINT_NB, OUTPUT_WIN_SZ, TARGET_DIM)
    w = g.edata['w']

    if feat.shape[0] != target.shape[0] or feat.shape[0] != w.shape[0]:
      raise ValueError("Node number inconsistent")

    if feat.shape[1] != target.shape[1]:
      raise ValueError("Data point number inconsistent")

    if feat.shape[2] != self.config.input_win_sz:
      raise ValueError("Input window size mismatch")

    if feat.shape[3] != self.config.feat_dim:
      raise ValueError("Feature dimension mismatch")

    if target.shape[2] != self.config.output_win_sz:
      raise ValueError("Output window size mismatch")

    if target.shape[3] != self.config.target_dim:
      raise ValueError("Target dimension mismatch")
