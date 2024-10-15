from dataclasses import dataclass
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from ..data.dataset import WindFFGraph, WindFFDataset
from .model import WindFFModel, WindFFModelConfig
import logging


class WindFFModelManager:

  DEFAULT_BATCH_SZ = 32

  def __init__(self, config: WindFFModelConfig):
    self.config = config
    self.model: WindFFModel = WindFFModel(config)

  @dataclass
  class TrainConfig:
    adj_weight_threshold: float

    epochs: int
    early_stop: bool
    patience: int
    lr: float
    batch_sz: int
    val_ratio: float

  def train(self, graph_list: list[WindFFGraph], config: TrainConfig):
    '''Train a model from scratch
    '''

    # TODO: Train the model based on multiple graphs
    if len(graph_list) != 1:
      logging.warning(
          "Training on multiple graphs is not supported yet, only the first graph is used")

    g = graph_list[0]
    dglg = g.get_dgl_graph()

    self.__check_graph(g)

    self.model = WindFFModel(self.config).to(self.config.dtype)
    self.model = self.__init_model_params(self.model)

    feat, target = g.get_windowed_node_data_all(
        input_win_sz=self.config.input_win_sz,
        output_win_sz=self.config.output_win_sz
    )

    w = g.get_normalized_edge_weight(
        adj_weight_threshold=config.adj_weight_threshold
    )

    train_sz = int((1 - config.val_ratio) * len(feat))
    val_sz = len(feat) - train_sz

    torch_ds = TensorDataset(feat, target)
    train_ds, val_ds = random_split(torch_ds, [train_sz, val_sz])

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_sz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_sz, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    patience_cnt = 0

    for epoch in range(config.epochs):
      # Train
      self.model.train()
      train_loss = 0.0
      for batch in train_loader:
        batch_feat, batch_target = batch

        optimizer.zero_grad()
        pred = self.model(dglg, batch_feat, w)

        loss = criterion(pred, batch_target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader) / batch_feat.shape[0]

      # Validation
      self.model.eval()
      val_loss = 0.0
      with torch.no_grad():
        for batch in val_loader:
          batch_feat, batch_target = batch
          pred = self.model(dglg, batch_feat, w)
          loss = criterion(pred, batch_target)
          val_loss += loss.item() / len(val_loader) / batch_feat.shape[0]

      logging.info(
          f"Epoch {epoch+1}/{config.epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

      if config.early_stop:
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          patience_cnt = 0
        else:
          patience_cnt += 1

        if patience_cnt >= config.patience:
          logging.info(f"Early stopping at epoch {epoch+1}")
          break

  def update(self):
    pass

  def infer(self, g: WindFFGraph):
    self.model.eval()
    with self.model.no_grad():
      self.__check_graph(g)
      # Get the last feature window
      win_feat = g.get_windowed_node_feat(
          -self.config.input_win_sz, self.config.input_win_sz)
      w = g.get_normalized_edge_weight(self.config.adj_weight_threshold)
      result = self.model(win_feat, w)
      return result

  def evaluate(self, g: WindFFGraph) -> float:
    self.__check_graph(g)
    feat, target = g.get_windowed_node_data_all(
        input_win_sz=self.config.input_win_sz,
        output_win_sz=self.config.output_win_sz
    )
    torch_ds = TensorDataset(feat, target)
    loader = DataLoader(
        torch_ds, batch_size=self.DEFAULT_BATCH_SZ, shuffle=False)

    self.model.eval()
    with torch.no_grad():
      loss = 0.0
      criterion = nn.MSELoss()
      for batch in loader:
        batch_feat, batch_target = batch
        w = g.get_normalized_edge_weight(self.config.adj_weight_threshold)
        pred = self.model(batch_feat, w)
        loss += criterion(pred, batch_target).item()
      loss = loss / len(loader)
      return loss

  def __check_graph(self, g: WindFFGraph):

    feat = g.get_node_feat()  # (N, T, FEAT_DIM)
    target = g.get_node_target()  # (N, T, TARGET_DIM)

    if feat.shape[1] != target.shape[1]:
      raise ValueError("Data point number inconsistent")

    if feat.shape[2] != self.config.feat_dim:
      raise ValueError("Feature dimension mismatch")

    if target.shape[2] != self.config.target_dim:
      raise ValueError("Target dimension mismatch")

  # TODO: Initialize model parameters in a configurable way
  def __init_model_params(self, model: WindFFModel) -> WindFFModel:
    return model
