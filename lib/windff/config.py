from dataclasses import dataclass
from typing import Type

import torch
import numpy as np

from .data.raw import RawTurbData


@dataclass
class InfluxDBConfig:
  url: str
  org: str
  user: str
  token: str

  # Raw data
  raw_data_bucket: str
  raw_turb_ts_measurement: str

  # Preprocessed data
  preprocessed_data_bucket: str
  preprocessed_turb_ts_measurement: str

  # Predicted data
  predicted_data_bucket: str
  predicted_turb_ts_measurement: str


@dataclass
class ModelConfig:
  feat_dim: int
  target_dim: int
  hidden_dim: int
  input_win_sz: int
  output_win_sz: int
  hidden_win_sz: int
  adj_weight_threshold: float
  dtype: torch.dtype = torch.float64


@dataclass
class TypeConfig:
  raw_turb_data_type: Type[RawTurbData]


@dataclass
class Config:
  influx_db: InfluxDBConfig
  model: ModelConfig
  type: TypeConfig

  time_interval: np.timedelta64

  preprocessor_url: str
  predictor_url: str


@dataclass
class FlaskServerConfig:
  listen_addr: str
  listen_port: int


@dataclass
class XMLRPCServerConfig:
  listen_addr: str
  listen_port: int


@dataclass
class XMLRPCClientConfig:
  url: str
