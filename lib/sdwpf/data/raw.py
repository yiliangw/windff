import numpy as np
import math

from ...windff.data.raw import RawTurbData
from ...windff.errors import RawDataParsingError
import json
from influxdb_client import Point as InfluxDBPoint
from influxdb_client.domain.write_precision import WritePrecision


class SDWPFRawTurbData(RawTurbData):
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

  dtype_map = {
      "timestamp": np.datetime64,
      "turb_id": str,
      "wspd": np.float64,
      "wdir": np.float64,
      "etmp": np.float64,
      "itmp": np.float64,
      "ndir": np.float64,
      "pab1": np.float64,
      "pab2": np.float64,
      "pab3": np.float64,
      "prtv": np.float64,
      "patv": np.float64
  }

  def __init__(self, timestamp: np.datetime64, turb_id: str, wspd: float, wdir: float, etmp: float, itmp: float, ndir: float, pab1: float, pab2: float, pab3: float, prtv: float, patv: float):
    self.timestamp: np.datetime64 = timestamp
    self.turb_id = turb_id
    self.wspd = wspd
    self.wdir = wdir
    self.etmp = etmp
    self.itmp = itmp
    self.ndir = ndir
    self.pab1 = pab1
    self.pab2 = pab2
    self.pab3 = pab3
    self.prtv = prtv
    self.patv = patv

  def timestamp(self):
    return self.timestamp

  def turb_id(self):
    return self.turb_id

  @classmethod
  def from_json(cls, json_str: str):
    try:
      d = json.loads(json_str)
      timestamp = np.datetime64(d["timestamp"], 's')
      turb_id = d["turb_id"]
      wspd = float(d["wspd"])
      wdir = float(d["wdir"])
      etmp = float(d["etmp"])
      itmp = float(d["itmp"])
      ndir = float(d["ndir"])
      pab1 = float(d["pab1"])
      pab2 = float(d["pab2"])
      pab3 = float(d["pab3"])
      prtv = float(d["prtv"])
      patv = float(d["patv"])
      return cls(timestamp, turb_id, wspd, wdir, etmp, itmp, ndir, pab1, pab2, pab3, prtv, patv)
    except Exception as e:
      raise RawDataParsingError(str(e))

  def to_json(self):
    return json.dumps({
        self.TIME_COL: int(self.timestamp.astype('datetime64[s]').astype(int)),
        self.TURB_COL: self.turb_id,
        "wspd": float(self.wspd),
        "wdir": float(self.wdir),
        "etmp": float(self.etmp),
        "itmp": float(self.itmp),
        "ndir": float(self.ndir),
        "pab1": float(self.pab1),
        "pab2": float(self.pab2),
        "pab3": float(self.pab3),
        "prtv": float(self.prtv),
        "patv": float(self.patv)
    })

  def to_dict(self):
    return json.loads(self.to_json())

  def to_influxdb_point(self, measurement: str) -> InfluxDBPoint:
    return InfluxDBPoint(measurement) \
        .tag(self.TURB_COL, self.turb_id) \
        .field("wspd", self.wspd) \
        .field("wdir", self.wdir) \
        .field("etmp", self.etmp) \
        .field("itmp", self.itmp) \
        .field("ndir", self.ndir) \
        .field("pab1", self.pab1) \
        .field("pab2", self.pab2) \
        .field("pab3", self.pab3) \
        .field("prtv", self.prtv) \
        .field("patv", self.patv) \
        .time(int(self.timestamp.astype('datetime64[s]').astype(int)), write_precision=WritePrecision.S)

  @classmethod
  def get_col_names(self):
    return ["wspd", "wdir", "etmp", "itmp", "ndir", "pab1", "pab2", "pab3", "prtv", "patv"]

  @classmethod
  def get_target_col_names(self):
    return ['patv']
