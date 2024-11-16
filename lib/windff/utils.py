import numpy as np


class Utils:
  @staticmethod
  def td_to_sec(td: np.timedelta64) -> int:
    return int(td.astype('timedelta64[s]').astype(int))

  @staticmethod
  def dt_to_sec(dt: np.datetime64) -> int:
    return int(dt.astype('datetime64[s]').astype(int))

  @staticmethod
  def dt_to_isoformat(dt: np.datetime64) -> str:
    return np.datetime_as_string(dt, unit='s')
