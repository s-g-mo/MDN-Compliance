'''
FUNCTION SET spectral.py

Functions to compute auto- and cross- spectral densities b/w components of an
OBS.

This code borrowed from OBStools (https://github.com/nfsi-canada/OBStools).
'''
#################################### IMPORTS ###################################

# The usual.
import numpy as np

################################### FUNCTIONS ##################################

def coherence(Gxy, Gxx, Gyy):
  """
  Calculates coherence between two components
  Parameters
  ---------
  Gxy : :class:`~numpy.ndarray`
      Cross spectral density function of `x` and `y`
  Gxx : :class:`~numpy.ndarray`
      Power spectral density function of `x`
  Gyy : :class:`~numpy.ndarray`
      Power spectral density function of `y`
  Returns
  -------
  : :class:`~numpy.ndarray`, optional
      Coherence between `x` and `y`
  """
  if np.any(Gxy) and np.any(Gxx) and np.any(Gxx):
      return np.abs(Gxy)**2/(Gxx*Gyy)
  else:
      return None

def phase(Gxy):
  """
  Calculates phase angle between two components

  Parameters
  ---------
  Gxy : :class:`~numpy.ndarray`
      Cross spectral density function of `x` and `y`

  Returns
  -------
  : :class:`~numpy.ndarray`, optional
      Phase angle between `x` and `y`

  """

  if np.any(Gxy):
      return np.angle(Gxy)
  else:
      return None

def admittance(Gxy, Gxx):
  """
  Calculates admittance between two components
  Parameters
  ---------
  Gxy : :class:`~numpy.ndarray`
      Cross spectral density function of `x` and `y`
  Gxx : :class:`~numpy.ndarray`
      Power spectral density function of `x`
  Returns
  -------
  : :class:`~numpy.ndarray`, optional
      Admittance between `x` and `y`
  """
  if np.any(Gxy) and np.any(Gxx):
      return np.abs(Gxy)/Gxx
  else:
      return None