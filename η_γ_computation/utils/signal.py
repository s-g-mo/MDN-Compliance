'''
FUNCTION statistics.py

A function to compute 95% statistics from a set of η and γ curves recorded by an
OBS

Stephen Mosher, Mar. 2022
'''
#################################### IMPORTS ###################################

import numpy as np
from scipy.interpolate import interp1d

################################### FUNCTION ###################################

def statistics(μ_η, μ_γ, η_observed, γ_observed):

  # Sort observations of η at each frequency.
  sorted_η = np.sort(η_observed, axis=0)

  # Interpolate and calculate 2.5 and 97.5 % 
  interp_η = interp1d(np.linspace(0, 1, len(η_observed)), sorted_η, axis=0)
  upper_bounds_η = interp_η(0.975)
  lower_bounds_η = interp_η(0.025)

  # Sort observations of γ at each frequency.
  sorted_γ = np.sort(γ_observed, axis=0)

  # Interpolate and calculate 2.5 and 97.5 %
  interp_γ = interp1d(np.linspace(0, 1, len(γ_observed)), sorted_γ, axis=0)
  upper_bounds_γ = interp_γ(0.975)
  lower_bounds_γ = interp_γ(0.025)

  # Ignore the zero frequency term (which will give div by 0 errors), and 
  # scale the confidence intervals at each frequency by the mean.
  upper_bounds_η = upper_bounds_η[1:]/μ_η[1:]
  lower_bounds_η = lower_bounds_η[1:]/μ_η[1:]

  # Put the zero frequency term back in.
  upper_bounds_η = np.insert(upper_bounds_η, 0, 0.0)
  lower_bounds_η = np.insert(lower_bounds_η, 0, 0.0)

  # Ignore the zero frequency term (which will give div by 0 errors), and 
  # scale the confidence intervals at each frequency by the mean.
  upper_bounds_γ = upper_bounds_γ[1:]/μ_γ[1:]
  lower_bounds_γ = lower_bounds_γ[1:]/μ_γ[1:]

  # Put the zero frequency term back in.
  upper_bounds_γ = np.insert(upper_bounds_γ, 0, 0.0)
  lower_bounds_γ = np.insert(lower_bounds_γ, 0, 0.0)

  # Transpose to give the correct shape. Return.
  limits_η = np.vstack([lower_bounds_η, upper_bounds_η]).T
  limits_γ = np.vstack([lower_bounds_γ, upper_bounds_γ]).T

  return (limits_η, limits_γ)