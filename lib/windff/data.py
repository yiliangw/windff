from abc import ABC, abstractmethod
from datetime import datetime


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
    pass

  @abstractmethod
  def to_json(self) -> str:
    pass
