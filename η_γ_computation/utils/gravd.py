'''
FUNCTION gravd.py

A function to compute the wavenumber of infragravity waves from the infragravity
wave dispersion relation.

IG Wave Dispersion Relation: ω^2 = g k(ω) tanh(k(ω)H) 

  ω - angular frequency
  g - acceleration due to gravity
  k(ω) - infragravity wave number as a function of angular frequency
  H - station depth in meters below sea level.

Solivng for k(ω) is non-trivial. Most approachs focus on either shallow or deep
water approximations, for which tanh(k(ω)H) = k(ω)H or 1, respectively. I took
a novel approach and used a rational approximation of tanh(x) to solve for k(ω).
The rational approximation is always better than the shallow approximation. 
However, it's only better than the deep approximation if k*H < 2.96.

For details refer to:

Mosher, S. G., Audet, P., & Gosselin, J. M. (2021).
Shear-wave velocity structure of sediments on Cascadia's continental margin from
probabilistic inversion of seafloor compliance data. Geochemistry, Geophysics, 
Geosystems, 22, e2021GC009720. https://doi. org/10.1029/2021GC009720.

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# The usual.
import numpy as np
import numpy.polynomial as poly

################################### FUNCTION ###################################

def gravd(ω, H):
  
  g = 9.79329 # gravitational accel. for oceanic contexts
  N = len(ω)

  # Deep water approximation for k.
  k_deep = ω**2 / g

  # Initialize values of k to be solved.
  k = np.zeros(len(ω))

  # The rational approximation to tanh(x) leads to a quartic polynomial in the
  # IG dispersion relation. We solve this using poly from numpy.polynomial.
  # Conveniently, the resulting polynomial always has real roots, and now we 
  # aren't restricted to considering deep and shallow cases seperately. Poly 
  # requires we break the resulting polynomial into terms of increasing order.

  a0 = -27 * ω**2 / g                   # constant terms
  a1 = np.zeros(len(ω))                 # no linear terms
  a2 = 27 * H - (9 * ω**2 * H**2)/g     # quadratic terms
  a3 = np.zeros(len(ω))                 # no cubic terms
  a4 = np.ones(len(ω)) * H**3           # quartic terms

  # Loop over angular frequencies and compute wavenumber for each.
  for i in range(len(ω)):

    # Case when ω = 0
    if ω[i] == 0:
      k[i] = 0

    # Othwerise  
    else:
      p = poly.Polynomial([a0[i], a1[i], a2[i], a3[i], a4[i]])
      solu = poly.Polynomial.roots(p)
      positive_roots = solu[solu > 0]

      # By default we get 4 complex numbers as results, so we force the
      # real roots (i.e. those with imaginary comp == 0) to be purely real
      # numbers.
      real_positive_root = positive_roots[positive_roots.imag == 0].real[0]
      k[i] = real_positive_root
  
  # For k*H >= 2.96, prefer the deep approximation.
  for j, wavenumber in enumerate(k_deep):
    if wavenumber * H > 2.96:
      k[j] = k_deep[j]
  
  return k