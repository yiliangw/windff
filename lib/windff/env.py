from .config import Config
from .components import Component


class Env:

  def __init__(self, config: Config):
    self.config = config
    self.components = {}

  def spawn(self, type: Component.Type) -> Component:
    '''Spawn a component of the given type
    '''
    comp = Component.create(type)
    comp.initialize(self)

    type_list = self.components.get(type, [])
    type_list.append(comp)
    self.components[type] = type_list

    return comp
