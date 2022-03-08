'''
FUNCTION compliance_coherence.py

A function to compute η and γ for an OBS from its auto- and cross-spectral
densities.

This code borrowed from OBStools (https://github.com/nfsi-canada/OBStools) with
minor modifications. 

Stephen Mosher, Mar. 2022
'''
#################################### IMPORTS ###################################

# The usual.
import numpy as np

# Additional helper functions.
from utils import gravd, spectral

################################### FUNCTION ###################################

def η_γ(spectral_components):
  
  # Access frequency and depth information from spectral components.
  freqs = spectral_components['freqs']
  depth = spectral_components['depth']

  # Calculate wavenumbers for this station depth.
  ω = 2 * np.pi * freqs
  k = gravd.gravd(ω, depth)

  # Grab auto- and cross-spectral densities from spectral components.
  cPP = spectral_components['cPP']
  c11 = spectral_components['c11']
  c22 = spectral_components['c22']
  cZZ = spectral_components['cZZ']
  c12 = spectral_components['c12']
  c1Z = spectral_components['c1Z']
  c2Z = spectral_components['c2Z']
  c1P = spectral_components['c1P']
  c2P = spectral_components['c2P']
  cZP = spectral_components['cZP']

  # Compute η and γ (compl_ZP_21, and coh_ZP_21). 
  # The remainder taken from OBStools.

  # ZP-21
  lc1c2 = np.conj(c12) / c11
  lc1cP = np.conj(c1P) / c11

  coh_12 = spectral.coherence(c12, c11, c22)
  coh_1P = spectral.coherence(c1P, c11, cPP)
  coh_1Z = spectral.coherence(c1Z, c11, cZZ)

  gc2c2_c1 = c22 * (1. - coh_12)
  gcPcP_c1 = cPP * (1. - coh_1P)
  gcZcZ_c1 = cZZ * (1. - coh_1Z)

  gc2cZ_c1 = np.conj(c2Z) - np.conj(lc1c2 * c1Z)
  
  gcPcZ_c1 = cZP - np.conj(lc1cP * c1Z)

  gc2cP_c1 = np.conj(c2P) - np.conj(lc1c2 * c1P)

  lc2cP_c1 = gc2cP_c1 / gc2c2_c1

  coh_c2cP_c1 = spectral.coherence(gc2cP_c1, gc2c2_c1, gcPcP_c1)
  coh_c2cZ_c1 = spectral.coherence(gc2cZ_c1, gc2c2_c1, gcZcZ_c1)
  
  gcPcP_c1c2 = gcPcP_c1 * (1. - coh_c2cP_c1)
  gcPcZ_c1c2 = gcPcZ_c1 - np.conj(lc2cP_c1) * gc2cZ_c1
  gcZcZ_c1c2 = gcZcZ_c1 * (1. - coh_c2cZ_c1)
  
  compl_ZP_21 = k * spectral.admittance(gcPcZ_c1c2, gcPcP_c1c2)
  coh_ZP_21 = spectral.coherence(gcPcZ_c1c2, gcPcP_c1c2, gcZcZ_c1c2)

  return(compl_ZP_21, coh_ZP_21)