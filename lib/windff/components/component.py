from abc import ABC, abstractmethod


class Component(ABC):

  @classmethod
  @abstractmethod
  def get_type(cls) -> str:
    pass

  @abstractmethod
  def initialize(self, env):
    pass

  @abstractmethod
  def start(self):
    pass
