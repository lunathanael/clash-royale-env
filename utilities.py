import time
import numpy as np
import math

def wait_until(somepredicate, timeout, period=0.01, *args, **kwargs):
  mustend = time.time() + timeout
  while time.time() < mustend:
    if somepredicate(*args, **kwargs): return True
    time.sleep(period)
  return False

def wait_until_value(someeval, value, timeout, period=0.25, *args, **kwargs):
  mustend = time.time() + timeout
  while time.time() < mustend:
    if someeval(*args, **kwargs) == value: return True
    time.sleep(period)
  return False

def wait_until_nvalue(someeval, nvalue, timeout, period=0.25, *args, **kwargs):
  mustend = time.time() + timeout
  while time.time() < mustend:
    if someeval(*args, **kwargs) != nvalue: return True
    time.sleep(period)
  return False

def redmean(bgr1, bgr2):
  rm = (bgr1[2] + bgr2[2]) // 2
  delta = (bgr1 - bgr2) ** 2
  return math.sqrt((2 + rm/256) * delta[2] + 4 * delta[1] + (2 + (255-rm) / 256) * delta[0])

def vredmean(bgr1, bgr2):
    rm = (bgr1[:, 2] + bgr2[:, 2]) // 2.0
    delta = (bgr1 - bgr2) ** 2
    return np.sqrt((2 + rm/256) * delta[:, 2] + 4 * delta[:, 1] + (2 + (255-rm) / 256) * delta[:, 0])