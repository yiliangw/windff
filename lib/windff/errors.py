
class WindffError(Exception):
  def __init__(self, msg: str = None):
    super(msg)


class WindffDBError(WindffError):
  def __init__(self, msg: str = None, raw: Exception = None):
    super(msg)
    self.raw = raw


class WindffDBConnectionError(WindffError):
  def __init__(self, msg: str = None):
    super(msg)


class WindffRawDataParsingError(WindffError):
  def __init__(self, msg: str = None):
    super(msg)
