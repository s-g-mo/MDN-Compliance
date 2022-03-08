'''
FUNCTION SET misc.py

A set of miscellaneous helper functions.

Stephen Mosher, Mar. 2022
'''
#################################### IMPORTS ###################################

# The usual suspects.
import numpy as np

################################### FUNCTIONS ################################## 

def idx_of_closest(target_value, array):
  '''
  Given an array and some target value, find the index of the element in the
  array that most closely matches the target value.
  '''
  idx = np.argmin(np.abs(target_value - array))
  return idx