
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lib.sdwpf.sim import SDWPFTurbineEdge
from lib.sdwpf.data import SDWPFRawTurbData

port = 18181

time_start = np.datetime64('2024-11-10T17:00:00', 's')
time_interval = np.timedelta64(10, 'm')
time_duration = np.timedelta64(1, 'h')

turbine_edges = SDWPFTurbineEdge.create_all(time_start, time_interval)

url = f'http://localhost:{port}/raw_turb_data'

interval_cnt = 1
current_time = time_start + time_interval

SDWPFRawTurbData.init('timestamp', 'turb_id')


def handler(signum, frame):
  global interval_cnt, current_time
  print(f'Sending raw data at {current_time}')
  for e in turbine_edges:
    e.send_raw_data(interval_cnt, url)
  print(f'Sent raw data at {current_time}')
  interval_cnt += 1
  current_time += time_interval

  if current_time >= time_start + time_duration:
    print('Simulation finished')
    sys.exit(0)


import signal
signal.signal(signal.SIGINT, handler)

while True:
  signal.pause()
