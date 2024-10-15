from dataclasses import dataclass


@dataclass
class RawDBConfig:
  bucket: str
  turb_measurement: str


@dataclass
class PreprocessedDBConfig:
  bucket: str
  turb_ts_measurement: str  # Turbine time series


@dataclass
class Config:
  raw_db: RawDBConfig
  preprocessed_db: PreprocessedDBConfig

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
