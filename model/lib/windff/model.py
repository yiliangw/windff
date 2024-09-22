import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph

from dgl.nn import GraphConv
from dataclasses import dataclass


class WindFFModel(nn.Module):

  @dataclass
  class Config:
    feat_dim: int
    input_win_sz: int
    output_win_sz: int
    hidden_win_sz: int
    out_hidden_dim: int

  class Linear(nn.Module):
    def __init__(self, in_features, out_features):
      super(WindFFModel.Linear, self).__init__()
      self.linear = nn.Linear(in_features, out_features)

    def forward(self, g: DGLGraph, x: torch.Tensor):
      B = x.shape[0]
      N = x.shape[1]
      x = x.view(B * N, -1)
      x = self.linear(x)
      x = x.view(B, N, -1)
      return x

  def __init__(self, cfg: Config):
    super(WindFFModel, self).__init__()
    self.cfg = cfg

    F = cfg.feat_dim
    I = cfg.input_win_sz
    O = cfg.output_win_sz
    H = cfg.hidden_win_sz

    self.conv = GraphConv(in_feats=(I * F), out_feats=(H * F))

    self.l_linear_out_0 = self.Linear(
        in_features=(H * F), out_features=(H * cfg.out_hidden_dim))
    self.l_linear_out_1 = self.Linear(
        in_features=(H * cfg.out_hidden_dim), out_features=(H * F))
    self.l_linear_out_2 = self.Linear(
        in_features=(H * F), out_features=(O * F)
    )

  def forward(self, g: DGLGraph, feat: torch.Tensor):
    '''
    @param feat: torch.tensor shape=(batch_sz/B, node_nb/N, input_win/I, feat_dim/D)
    '''

    # Check input shape
    if len(feat.shape) == 4:
      batched = True
      B, N, I, F = feat.shape
    elif len(feat.shape) == 3:
      batched = False
      N, I, F = feat.shape
      feat = feat.unsqueeze(0)
      B = 1
    else:
      raise ValueError(
          f"Input shape must be (B, N, I, F) or (N, I, F)")

    assert feat.shape[1] == g.number_of_nodes(
    ) and feat.shape[2] == self.cfg.input_win_sz and feat.shape[3] == self.cfg.feat_dim

    B, N, I, F = feat.shape
    O = self.cfg.outpu_win_sz
    H = self.cfg.hidden_win_sz

    # Graph convolution
    x = feat.view(B, N, I * F)
    x = self.conv(g, x)  # (B, N, H * F)
    x = nn.ReLU(x)

    # Output linear layers
    x = self.l_linear_out_0(x)  # (B, N, H * out_hidden_dim)
    x = F.relu(x)
    x = self.l_linear_out_1(x)  # (B, N, H * F)
    x = F.relu(x)
    x = self.l_linear_out_2(x)  # (B, N, O * F)

    x = x.view(B, N, O, F)

    if not batched:
      x = x.squeeze(0)

    return x
