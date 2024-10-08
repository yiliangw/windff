import json
import datetime


class TurbineData(object):

  """
  Properties:
    wspd (float): The wind speed recorded by the anemometer
    wdir (float): The angle (degree) between the wind direction and the position of turbine nacellel
    etmp (float): Temperature (Celcius degree) of the environment
    itmp (float): Temperature (Celcius degree) inside the turbine nacelle
    ndir (float): Nacelle direction, i.e., the yaw angle of the nacelle
    pab1 (float): Pitch angle (degree) of blade 1
    pab2 (float): Pitch angle (degree) of blade 2
    pab3 (float): Pitch angle (degree) of blade 3
    prtv (float): Reactive power (kW)
    patv (float): Active power (kW)
  """
  PROPERTIES = {
      'wspd': float,
      'wdir': float,
      'etmp': float,
      'itmp': float,
      'ndir': float,
      'pab1': float,
      'pab2': float,
      'pab3': float,
      'prtv': float,
      'patv': float,
  }

  def __init__(self, timestamp: datetime.datetime, turbine_id: int, properties: dict):
    self.timestamp = timestamp
    self.turbine_id = turbine_id

    for prop in self.PROPERTIES:
      type = self.PROPERTIES[prop]
      try:
        val = type(properties[prop])
      except KeyError:
        raise Exception(f"Missing property: {prop}")
      except ValueError:
        raise Exception(
            f"Invalid value for property: {prop}: {properties[prop]}")

      setattr(self, prop, val)

  @classmethod
  def from_json(cls, json_str: str) -> 'TurbineData':
    data = json.loads(json_str)
    return cls(datetime.strptime(data['timestamp']), int(data['turbine_id']), data)

  def to_json(self) -> str:
    obj = {}
    obj['timestamp'] = self.timestamp.isoformat()
    obj['turbine_id'] = self.turbine_id
    for prop in self.PROPERTIES:
      obj[prop] = getattr(self, prop)
    return json.dumps(obj)
