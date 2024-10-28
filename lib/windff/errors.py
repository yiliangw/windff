class Error(Exception):
  def __init__(self, msg: str = None):
    super(msg)


class DBError(Error):
  def __init__(self, msg: str = None, raw: Exception = None):
    super(msg)
    self.raw = raw


class DBConnectionError(Error):
  def __init__(self, msg: str = None):
    super(msg)


class RawDataParsingError(Error):
  def __init__(self, msg: str = None):
    super(msg)
