'''
FUNCTION ncomp_fortran.py

Translated from Wayne Crawford's MATLAB code for forward calculating normalized
compliance from synthetic Earth structure  

Original source can be found at (http://www.ipgp.fr/~crawford/Homepage/Software.html).

Stephen Mosher, Feb 2020. 
'''

#################################### IMPORTS ###################################

# The usual.
import numpy as np

# Import helper functions - raydep_ft does the heavy lifting. It's an F95
# file and it has to be compiled using F2PY from NumPy.
from forward_funcs import gravd, raydep_ft

################################### FUNCTION ###################################

def ncomp_fortran(depth, freqs, model):
  # Compute wavenumber of infragravity waves and slowness vectors.
  ω = 2 * np.pi * freqs
  k = gravd.gravd(ω, depth)
  p = k / ω

  # Compute normalized compliance.
  ncomp = np.zeros(len(ω))
  for i in range(len(p)):
    v, u, sigzz, sigzx = raydep_ft.raydep_ft(p[i], ω[i], model)
    ncomp[i] = -k[i] * v[0] / (ω[i] * sigzz[0])
  return ncomp