from .. import Config
from .component import Component


class Broadcaster(Component):

  @classmethod
  def get_type(cls) -> str:
    return 'broadcaster'

  def __init__(self):
    from ..env import Env
    self.env: Env = None

  def initialize(self, env):
    self.env = env

  def start(self):
    pass
