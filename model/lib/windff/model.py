import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph

from dgl.nn import GraphConv
from dataclasses import dataclass


@dataclass
class WindFFModelConfig:
  feat_dim: int
  target_dim: int
  hidden_dim: int
  input_win_sz: int
  output_win_sz: int
  hidden_win_sz: int


class WindFFModel(nn.Module):

  class TimeLinear(nn.Module):
    def __init__(self, in_win_sz, out_win_sz):
      super(WindFFModel.TimeLinear, self).__init__()
      self.linear = nn.Linear(in_win_sz, out_win_sz)

    def forward(self, g: DGLGraph, x: torch.Tensor):
      B, N, T, F = x.shape
      x = x.permute((0, 1, 3, 2))
      x = x.view(B * N * F, T)
      x = self.linear(x)
      x = x.view(B, N, F, -1)
      x = x.permute((0, 1, 3, 2)).contiguous()
      return x

  class FeatureLinear(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim):
      super(WindFFModel.FeatureLinear, self).__init__()
      self.linear = nn.Linear(in_feat_dim, out_feat_dim)

    def forward(self, g: DGLGraph, x: torch.Tensor):
      B, N, T, F = x.shape
      x = x.view(B * N, T * F)
      x = self.linear(x)
      x = x.view(B, N, T, -1)
      return x

  class GraphConv(nn.Module):
    def __init__(self):
      super(WindFFModel.GraphConv, self).__init__()

    def forward(self, g: DGLGraph, x: torch.Tensor, w: torch.Tensor):
      B, N, T, F = x.shape
      x = x.view(B, N, x.shape[2] * x.shape[3])

      with g.local_scope():
        g.ndata['x'] = x
        g.edata['w'] = w
        g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'),
                     reduce_func=fn.mean('m', 'h'))
        return g.ndata['h']

  def __init__(self, config: WindFFModelConfig):
    super(WindFFModel, self).__init__()
    self.config = config

    idim = config.feat_dim
    odim = config.target_dim
    hdim = config.hidden_dim

    iwin = config.input_win_sz
    owin = config.output_win_sz
    hwin = config.hidden_win_sz

    # Input linear layers map (iwin, idim) features to (hwin, hdim) features
    self.l_linear_in_0 = self.FeatureLinear(
        in_feat_dim=idim,
        out_feat_dim=hdim
    )
    self.l_linear_in_1 = self.TimeLinear(
        in_win_sz=iwin,
        out_win_sz=hwin
    )

    # Graph convolution layers
    self.conv = self.GraphConv()

    # Output linear layers map (hwin, hdim) features to (owin, odim) features
    self.l_linear_out_0 = self.TimeLinear(
        in_win_sz=hwin,
        out_win_sz=owin
    )
    self.l_linear_out_1 = self.FeatureLinear(
        in_feat_dim=hdim,
        out_feat_dim=odim
    )

  def forward(self, g: DGLGraph, feat: torch.Tensor, w: torch.Tensor):
    '''
    @param feat: torch.tensor shape=(batch_sz/B, node_nb/N, input_win/I, feat_dim/D)
    '''

    # Check input shape
    if len(feat.shape) == 4:
      batched = True
      B, N, I_WIN, I_DIM = feat.shape
    elif len(feat.shape) == 3:
      batched = False
      N, I_WIN, I_DIM = feat.shape
      feat = feat.unsqueeze(0)
      B = 1
    else:
      raise ValueError(
          f"Input shape must be (B, N, I_WIN, I_DIM) or (N, I_WIN, I_DIM)")

    if (I_WIN != self.config.input_win_sz):
      raise ValueError(
          f"Input window size mismatch: {I_WIN} != {self.config.input_win_sz}")
    if (I_DIM != self.config.feat_dim):
      raise ValueError(
          f"Feature dimension mismatch: {I_DIM} != {self.config.feat_dim}")

    assert feat.shape[1] == g.number_of_nodes(
    ) and feat.shape[2] == self.config.input_win_sz and feat.shape[3] == self.config.feat_dim

    x = feat
    x = self.l_linear_in_0(g, x)  # (B, N, I_WIN, H_DIM)
    x = F.relu(x)
    x = self.l_linear_in_1(g, x)  # (B, N, H_WIN, H_DIM)
    x = F.relu(x)

    x = self.conv(g, x, w)
    x = F.relu(x)

    # Output linear layers
    x = self.l_linear_out_0(x)  # (B, N, O_WIN, H_DIM)
    x = F.relu(x)
    x = self.l_linear_out_1(x)  # (B, N, O_WIN, O_DIM)

    if not batched:
      x = x.squeeze(0)

    return x
