from abc import ABC, abstractmethod
from datetime import datetime
from influxdb_client import Point as InfluxDBPoint


class RawDataParsingError(Exception):
  def __init__(self, str):
    super().__init__(str)


class TurbRawData(ABC):

  @abstractmethod
  def timestamp(self) -> datetime:
    pass

  @abstractmethod
  def turb_id(self) -> str:
    pass

  @abstractmethod
  @classmethod
  def from_json(json_str: str) -> 'TurbRawData':
    """_summary_

    Raises:
        DataParsingError: Error parsing JSON string
    """
    pass

  @abstractmethod
  def to_json(self) -> str:
    pass

  @abstractmethod
  def to_influxdb_point(self) -> InfluxDBPoint:
    pass
