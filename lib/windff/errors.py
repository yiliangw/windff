from enum import Enum


class Errno(Enum):
  OK = 0
  DBQueryErr = 1
  DBWriteErr = 2


class WindffError(Exception):
  def __init__(self, msg: str = None):
    super(msg)


class DBError(WindffError):
  def __init__(self, msg: str = None):
    super(msg)


class DBQueryError(DBError):
  def __init__(self, msg: str = None, raw: Exception = None):
    super(msg)
    self.raw = raw


class DBWriteError(DBError):
  def __init__(self, msg: str = None, raw: Exception = None):
    super(msg)
    self.raw = raw


class DBConnectionError(DBError):
  def __init__(self, msg: str = None):
    super(msg)


class RawDataParsingError(WindffError):
  def __init__(self, msg: str = None):
    super(msg)
