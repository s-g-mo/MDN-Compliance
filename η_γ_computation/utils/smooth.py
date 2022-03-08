'''
FUNCTION smooth.py

Description below.

Borrowed from OBStools (https://github.com/nfsi-canada/OBStools). 
'''

#################################### IMPORTS ###################################

import numpy as np

################################### FUNCTION ###################################

def smooth(data, nd, axis=0):
  """
  Function to smooth power spectral density functions from the convolution
  of a boxcar function with the PSD

  Parameters
  ----------
  data : :class:`~numpy.ndarray`
      Real-valued array to smooth (PSD)
  nd : int
      Number of samples over which to smooth
  axis : int
      axis over which to perform the smoothing

  Returns
  -------
  filt : :class:`~numpy.ndarray`, optional
      Filtered data

  """
  if np.any(data):
      if data.ndim > 1:
          filt = np.zeros(data.shape)
          for i in range(data.shape[::-1][axis]):
              if axis == 0:
                  filt[:, i] = np.convolve(
                      data[:, i], np.ones((nd,))/nd, mode='same')
              elif axis == 1:
                  filt[i, :] = np.convolve(
                      data[i, :], np.ones((nd,))/nd, mode='same')
      else:
          filt = np.convolve(data, np.ones((nd,))/nd, mode='same')
      return filt
  else:
      return None