'''
FUNCTION sliding_window.py

Description below.

Borrowed from OBStools (https://github.com/nfsi-canada/OBStools). 
'''

#################################### IMPORTS ###################################

# The usual.
import numpy as np

################################### FUNCTION ###################################

def sliding_window(a, ws, ss=None, hann=True):
  """
  Function to split a data array into overlapping, possibly tapered sub-windows

  Parameters
  ----------
  a : :class:`~numpy.ndarray`
      1D array of data to split
  ws : int
      Window size in samples
  ss : int
      Step size in samples. If not provided, window and step size
       are equal.

  Returns
  -------
  out : :class:`~numpy.ndarray`
      1D array of windowed data
  nd : int
      Number of windows

  """

  if ss is None:
      # no step size was provided. Return non-overlapping windows
      ss = ws

  # Calculate the number of windows to return, ignoring leftover samples, and
  # allocate memory to contain the samples
  valid = len(a) - ss
  nd = (valid) // ss
  out = np.ndarray((nd, ws), dtype=a.dtype)

  if nd == 0:
      if hann:
          out = a * np.hanning(ws)
      else:
          out = a

  for i in range(nd):
      # "slide" the window along the samples
      start = i * ss
      stop = start + ws
      if hann:
          out[i] = a[start: stop] * np.hanning(ws)
      else:
          out[i] = a[start: stop]

  return out, nd