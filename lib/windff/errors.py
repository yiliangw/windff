from enum import Enum


class Errno(Enum):
  OK = 0
  DBQueryErr = 1
  DBWriteErr = 2


class WindffError(Exception):
  def __init__(self, msg: str = None):
    super().__init__(msg)


class DBError(WindffError):
  def __init__(self, msg: str = None):
    super().__init__(msg)


class DBQueryError(DBError):
  def __init__(self, msg: str = None, raw: Exception = None):
    super().__init__(msg)
    self.raw = raw


class DBWriteError(DBError):
  def __init__(self, msg: str = None, raw: Exception = None):
    super().__init__(msg)
    self.raw = raw


class DBConnectionError(DBError):
  def __init__(self, msg: str = None):
    super().__init__(msg)


class RawDataParsingError(WindffError):
  def __init__(self, msg: str = None):
    super().__init__(msg)
