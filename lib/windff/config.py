from dataclasses import dataclass

@dataclass
class InfluxDBClientConfig:
  url: str
  token: str
  bucket: str
  org: str

@dataclass
class FlaskServerConfig:
  listen_addr: str
  listen_port: int
