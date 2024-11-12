from .. import WinffConfig
from .component import Component


class Broadcaster(Component):

  @classmethod
  def get_type(cls) -> str:
    return 'broadcaster'

  def __init__(self):
    from ..env import WindffEnv
    self.env: WindffEnv = None

  def initialize(self, env):
    self.env = env

  def start(self):
    pass
