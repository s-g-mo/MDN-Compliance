'''
FUNCTION SET misc.py

A set of miscellaneous helper functions.

Stephen Mosher, Mar. 2022
'''
#################################### IMPORTS ###################################

# The usual suspects.
import numpy as np

# Useful for percentile_levels() 
from scipy.stats import gaussian_kde

################################### FUNCTIONS ################################## 

def idx_of_closest(target_value, array):
  '''
  Given an array and some target value, find the index of the element in the
  array that most closely matches the target value.
  '''
  idx = np.argmin(np.abs(target_value - array))
  return idx

def percentile_levels(data):
  '''
  Given a 2D numpy array of data, of shape (number_observations, length of
  signals), compute the 95% limits for the signals across their domain.
  '''

  # Initialize arrays to hold the lower bounds L at 2.5% and the upper bounds U
  # at 97.5%
  L = np.zeros(data.shape[1])
  U = np.zeros(data.shape[1])
  
  print('Computing 95% levels...')

  # For each value in the domain of the individual signals...
  for i in range(data.shape[1]):

    # Copy/slice into observations from all signals at that value.
    samples = data[:,i].copy()

    # Create an axis covering the distribution of samples at current value.
    distribution_axis = np.linspace(np.amin(samples), np.amax(samples), len(samples))
    
    # Sort the samples.
    samples.sort()

    # Estimate the distribution of the samples using a Gaussian KDE.
    kernel = gaussian_kde(samples, bw_method='scott')

    # Derive the PDF of the samples from the KDE, normalize, compute the CDF.
    pdf = kernel.pdf(samples)
    pdf = pdf/np.sum(pdf)
    cdf = np.cumsum(pdf)

    # Then use the CDF to determine the 2.5% and 97.5% limits of the samples for
    # the current domain value.
    L[i] = distribution_axis[np.argmin(np.abs(cdf - 0.025))]
    U[i] = distribution_axis[np.argmin(np.abs(cdf - 0.975))]

  return (L,U)