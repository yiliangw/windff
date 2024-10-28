from abc import ABC, abstractmethod
from enum import Enum
from ..env import Env

from .controller import Controller
from .collector import Collector
from .preprocessor import Preprocessor
from .predictor import Predictor
from .broadcaster import Broadcaster


class Component(ABC):

  class Type(Enum):
    CONTROLLER = 1
    COLLECTOR = 2
    PREPROCESSOR = 3
    PREDICTOR = 4
    BROADCASTER = 5

  @classmethod
  def create(cls, type: Type):
    match type:
      case Component.Type.CONTROLLER:
        return Controller()
      case Component.Type.COLLECTOR:
        return Collector()
      case Component.Type.PREPROCESSOR:
        return Preprocessor()
      case Component.Type.PREDICTOR:
        return Predictor()
      case Component.Type.BROADCASTER:
        return Broadcaster()
      case _:
        raise ValueError(f"Unknown component type: {type}")

  @abstractmethod
  def initialize(self, env: Env):
    pass

  @abstractmethod
  def start(self):
    pass
