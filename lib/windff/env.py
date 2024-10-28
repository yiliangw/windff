from .config import Config
from .components import Component, Controller, Collector, Preprocessor, Predictor

import numpy as np


class Env:

  def __init__(self, config: Config):
    self.config = config
    self.components = {}

    self.time_interval: np.timedelta64
    self.time_retry_interval: np.timedelta64
    self.time_win_sz: int
    self.time_guard: np.timedelta64

    self.preprocess_retry_nb: int = 3
    self.predict_retry_nb: int = 3

  def spawn(self, type: Component.Type) -> Component:
    '''Spawn a component of the given type
    '''
    comp = Component.create(type)
    comp.initialize(self)

    type_list = self.components.get(type, [])
    type_list.append(comp)
    self.components[type] = type_list

    return comp

  def get_components(self, type: Component.Type):
    '''Get the components of the given type
    '''
    return self.components.get(type, [])

  def call_preprocess(self, turbs: set[str], time_start: np.datetime64, time_end: np.datetime64, time_interval: np.timedelta64):
    comp_l = self.get_components(Component.Type.PREPROCESSOR)
    assert len(comp_l) > 0
    preprocessor: Preprocessor = comp_l[0]
    return preprocessor.preprocess(turbs, time_start, time_end, time_interval)

  def call_predict(self, time: np.datetime64):
    comp_l = self.get_components(Component.Type.PREDICTOR)
    assert len(comp_l) > 0
    predictor: Predictor = comp_l[0]
    return predictor.predict(time)
