from .. import Config
from .component import Component


class Broadcaster(Component):

  @classmethod
  def get_type(cls) -> str:
    return 'broadcaster'

  def __init__(self, config: Config):
    self.config = config
