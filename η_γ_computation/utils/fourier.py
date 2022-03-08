'''
FUNCTION fourier.py

Helper functions to calculate a windowed FFT from an ObsPy Trace obj. as input.

Borrowed from OBStools (https://github.com/nfsi-canada/OBStools). 
'''

#################################### IMPORTS ###################################

import numpy as np
from utils import sliding_window

################################### FUNCTIONS ##################################

def calculate_windowed_fft(trace, ws, ss=None, hann=True):
  """
  Calculates windowed Fourier transform
  Parameters
  ----------
  trace : :class:`~obspy.core.Trace`
      Input trace data
  ws : int
      Window size, in number of samples
  ss : int
      Step size, or number of samples until next window
  han : bool
      Whether or not to apply a Hanning taper to data
  Returns
  -------
  ft : :class:`~numpy.ndarray`
      Fourier transform of trace
  f : :class:`~numpy.ndarray`
      Frequency axis in Hz
  """

  n2 = _npow2(ws)
  f = trace.stats.sampling_rate/2. * np.linspace(0., 1., int(n2/2) + 1)
  
  # Extract sliding windows
  tr, nd = sliding_window.sliding_window(trace.data, ws, ss)
  
  # Fourier transform
  ft = np.fft.fft(tr, n=n2)
  
  return ft, f

def _npow2(x):
  return 1 if x == 0 else 2**(x-1).bit_length()