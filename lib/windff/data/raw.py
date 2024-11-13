from abc import ABC, abstractmethod
from datetime import datetime
from influxdb_client import Point as InfluxDBPoint
from ..errors import RawDataParsingError


class RawTurbData(ABC):

  @abstractmethod
  def timestamp(self) -> datetime:
    pass

  @abstractmethod
  def turb_id(self) -> str:
    pass

  @classmethod
  @abstractmethod
  def from_json(json_str: str) -> 'RawTurbData':
    """_summary_

    Raises:
        WindffRawDataParsingError: Error parsing JSON string
    """
    pass

  @abstractmethod
  def to_json(self) -> str:
    pass

  @abstractmethod
  def to_influxdb_point(self, measurement: str) -> InfluxDBPoint:
    pass
