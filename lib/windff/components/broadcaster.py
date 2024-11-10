from .. import Config
from .component import Component


class Broadcaster(Component):
  def __init__(self, config: Config):
    self.config = config
