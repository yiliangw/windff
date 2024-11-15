from abc import ABC, abstractmethod
from datetime import datetime
from influxdb_client import Point as InfluxDBPoint
from ..errors import RawDataParsingError


class RawTurbData(ABC):

  TIME_COL: str = None
  TURB_COL: str = None

  @classmethod
  def init(cls, time_col: str, turb_col: str):
    cls.TIME_COL = time_col
    cls.TURB_COL = turb_col

  @classmethod
  @abstractmethod
  def from_json(json_str: str) -> 'RawTurbData':
    """_summary_

    Raises:
        WindffRawDataParsingError: Error parsing JSON string
    """
    pass

  @classmethod
  @abstractmethod
  def get_col_names(self) -> list[str]:
    '''
    @return List of column names except for the timestamp and turbine ID columns
    '''
    pass

  @classmethod
  def get_target_col_names(self):
    '''
    @return List of target column names
    '''
    pass

  @abstractmethod
  def to_json(self) -> str:
    pass

  @abstractmethod
  def to_influxdb_point(self, measurement: str) -> InfluxDBPoint:
    pass
