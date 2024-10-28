from dataclasses import dataclass
import torch


@dataclass
class RawDBConfig:
  bucket: str
  turb_measurement: str


@dataclass
class PreprocessedDBConfig:
  bucket: str
  turb_ts_measurement: str  # Turbine time series


@dataclass
class PredictedDBConfig:
  bucket: str
  turb_ts_measurement: str


@dataclass
class ModelConfig:
  feat_dim: int
  target_dim: int
  hidden_dim: int
  input_win_sz: int
  output_win_sz: int
  hidden_win_sz: int
  dtype: torch.dtype = torch.float64


@dataclass
class Config:
  raw_db: RawDBConfig
  preprocessed_db: PreprocessedDBConfig
  predicted_db: PredictedDBConfig

  model: ModelConfig

  preprocessor_url: str
  predictor_url: str
  trainer_url: str


@dataclass
class InfluxDBClientConfig:
  url: str
  token: str
  turb_data_bucket: str
  org: str


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
