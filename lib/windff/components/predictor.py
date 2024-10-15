from ..model import Model, ModelManager
from ..data.dataset import Graph
from ..config import Config
import numpy as np


class Predictor(object):

  def __init__(self, config: Config):
    self.config: Config = config
    self.manager: ModelManager = None

  def __prepare_data(self, time: np.datetime64):
    '''Prepare the data for prediction
    @param time: The time for prediction
    TODO:
    '''
    pass

  def do_predict(self, g: Graph):
    return self.manager.infer(g)

  def predict(self, time: np.datetime64):
    '''Predict with output window starting from the given time
    '''
    g = self.__prepare_data(time)
    res = self.do_predict(g)
    return res
