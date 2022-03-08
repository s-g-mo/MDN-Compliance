'''
FUNCTION SET structural.py

A set of functions to aid in building synthetic Earth models from which to
compute normalized compliance.

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# The usual.
import numpy as np

# For constructing Bernstein basis.
from scipy.special import binom

################################### FUNCTIONS ##################################

# Vp as a function of Vs in oceanic crust.
def Vp_fixed_scaling_Hyndman(Vs):
  Vp_layers = 1.87 * Vs
  return Vp_layers

# Vp as a function of z only.
def Vp_Hamilton_1979(z):
  z = np.linspace(0, z/1000, z) # depth in [km]
  Vp = 1.511 + 1.304 * z - 0.741 * z**2 + 0.257 * z**3
  return Vp

# ρ in terms of Vp
def ρ_Christensen_Shaw_1970(Vp):
  ρ = 1.85 + 0.165 * Vp # Vp in [km/s] gives ρ in g/cm^3
  return ρ

# Vs profile determined by Ruan et al. 2014, in Cascadia.
def ruan_limit(z):
  Z = np.linspace(0, z/1000, z)
  a = 0.15608
  b = 1.2198
  c = 0.49473
  ruan_profile = (a * Z**2 + b * Z + c * 0.1)/(c + Z)
  limit = ruan_profile[-1]
  return limit

# Determine Bernstein basis polynomials for a given order.
def bernstein_basis(z, order, j):
  return binom(order, j) * (1 - z)**(order - j) * z**j

# Build a Vs profile from a set of Bernstein basis coefficients.
def bernstein_profile(z, order, coeff):
  assert len(coeff) == order + 1
  profile = np.zeros(len(z))
  for j in range(order+1):
    profile += coeff[j] * bernstein_basis(z, order, j)
  return(profile)