'''
FUNCTION ftest.py

An F-test to determine whether data contained in a given PSD is beyond the 
standard deviation obtained from a set of observations of several PSDs.

Borrowed from OBStools (https://github.com/nfsi-canada/OBStools). 
'''

#################################### IMPORTS ###################################

import numpy as np
from scipy.stats import f as f_dist

################################### FUNCTION ###################################

def ftest(res1, pars1, res2, pars2):

  N1 = len(res1)
  N2 = len(res2)

  dof1 = N1 - pars1
  dof2 = N2 - pars2

  Ea_1 = np.sum(res1**2)
  Ea_2 = np.sum(res2**2)

  Fobs = (Ea_1/dof1)/(Ea_2/dof2)

  P = 1. - (f_dist.cdf(Fobs, dof1, dof2) - f_dist.cdf(1./Fobs, dof1, dof2))

  return P