
import os
import sys
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lib.sdwpf.sim import SDWPFClient

port = 18181

time_start = np.datetime64('2024-11-10T17:00:00', 's')
time_interval = np.timedelta64(10, 'm')
time_duration = np.timedelta64(1, 'h')

SDWPFClient.logger.setLevel(logging.INFO)
client = SDWPFClient()

url = f'http://localhost:{port}/query'

interval_cnt = 1
current_time = time_start + time_interval


def handler(signum, frame):
  global interval_cnt, current_time
  print(f'Sending query at {current_time}')
  client.send_query(current_time, current_time + time_interval, url)
  print(f'Sent query at {current_time}')
  interval_cnt += 1
  current_time += time_interval

  if current_time >= time_start + time_duration:
    print('Simulation finished')
    sys.exit(0)


import signal
signal.signal(signal.SIGINT, handler)

while True:
  signal.pause()
