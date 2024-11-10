import numpy as np
import threading
import logging

from .component import Component


class Controller(Component):

  def __init__(self):
    from ..env import Env
    self.env: Env = None
    self.nodes: list[str] = []
    self.timer: threading.Timer = None

  @classmethod
  def get_type(cls):
    return 'controller'

  def initialize(self, env):
    logging.info("Initializing controller...")
    self.env = env
    logging.info("Controller initialized.")

  def start(self):
    """Periodically call preprocess and predict"""
    logging.info("Starting controller...")
    self.__tick()
    logging.info("Controller started.")

  def __tick(self):
    now = np.datetime64('now')
    target = np.datetime64(now + self.env.time_interval, np.datetime_data(
        self.env.time_interval))
    res = self.process(target)

    if res == 0:
      next_tick = target + self.env.time_guard
    else:
      next_tick = now + self.env.time_retry_interval
    self.__schedule_tick(next_tick - now)

  def __schedule_tick(self, delay: np.timedelta64):
    self.timer = threading.Timer(delay.astype(
        'timedelta[s]').astype(float), self.__tick)
    logging.info("Scheduled next tick in %s seconds", delay)

  def process(self, target: np.datetime64):
    # Ensure the time is aligned with interval
    time = np.datetime64(time, np.datetime_data(self.env.time_interval))
    time_start = time - self.env.time_interval * self.env.time_win_sz

    for _ in range(self.env.preprocess_retry_nb):
      res = self.__preprocess(time_start, target, self.env.time_interval)
      if res == 0:
        break
    if res != 0:
      logging.error(f"Preprocess failed for {target.isoformat()}")
      return res

    for _ in range(self.env.predict_retry_nb):
      res = self.__predict(target)
      if res == 0:
        break
    if res != 0:
      logging.error(f"Predict failed for {target.isoformat()}")
      return res

    return 0

  def __preprocess(self, time_start: np.datetime64, time_end: np.datetime64, interval: np.timedelta64):
    self.env.call_preprocess(
        self.nodes, time_start, time_end, interval)

  def __predict(self, time: np.datetime64):
    self.env.call_predict(time)
