'''
SCRIPT build_stn_db.py

This script builds a station database to use in conjuction with other aspects
of this project. In particular, since the frequency band over which η is
measureable is both depth-dependant, and depends on γ, several quantities need
to be specified for every station/target water depth you wish to work with.

For each station the following should be specified.

  - name
  - deployment depth
  - η frequency band limits
  - frequencies for which η,γ, and the resulting signal statistics were measured
  - γ, in order to weight the synthetic compliance signal
  - σ (measured compliance noise statistics), used as a basis for adding noise
    to synthetic compliance signals

If you want to work in the purely synthetic case, you can leave γ, and σ == 1
if you'd like, but your final inversion results will be artificially good in
that case...

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# The usual.
import pickle
import numpy as np 

# Helper functions.
from utils import fetch

#################################### CASE 1 ####################################

# If you want to use signal statistics computed from real η signals as the 
# basis of noise for the synthetics, then point to the directory containing
# your single station results. 

input_dir = './data/spectral/stn_avg_η_γ/'

#################################### CASE 2 ####################################

# If you want to play around with purely synthetic stations, then you'll have
# to think about how best to add noise to your synthetics, and whether you
# want to model γ for a given hypothetical station depth. Not trivial.

#input_dir = ''

##################################### MAIN #####################################

# Dictionary for station database.
stn_db = {}

# Case 1
if input_dir:

  # Create a list of files of single-station η and γ curves.
  stn_fles = fetch.data_paths(input_dir)

  # Determine the stations from the list of files.
  stns = [fle.split('/')[-1].split('.pkl')[0] for fle in stn_fles]

  # Loop over stations and the corresponding η, γ files to build up a stn db.
  for stn, fle in zip(stns, stn_fles):
  
    data = pickle.load(open(fle, 'rb'))
    
    stn_db[stn] = {}
    stn_db[stn]['stn'] = stn
    stn_db[stn]['depth'] = data['depth']
    stn_db[stn]['η_f_bounds'] = data['η_f_bounds']
    stn_db[stn]['γ'] = data['μ_γ']
    stn_db[stn]['σ'] = data['σ_η']
    stn_db[stn]['freqs'] = data['freqs']

# Case 2 - probably requires more work. I've put some basic values as a baseline
else:
  stns = ['STN1']                        # List of stations
  depths = [2015]                        # List of deployment depths [m]
  f_bounds = [(0.007, 0.024)]            # List of η frequency limits [Hz]
  freqs = [np.linspace(0, 0.5, 2048)]    # Full freq range over which γ, σ exist
  σs = [np.ones(shape=(len(freqs[0]),2))]# List of noise limits for sigs
  γs = [np.ones(len(freqs[0]))]          # List of coherehce functions for sigs

  # Loop over stns and associated properties to build up a stn db.
  for i, stn in enumerate(stns):
    stn_db[stn] = {}
    stn_db[stn]['stn'] = stn
    stn_db[stn]['depth'] = depths[i]
    stn_db[stn]['η_f_bounds'] = f_bounds[i]
    stn_db[stn]['γ'] = γs[i]
    stn_db[stn]['σ'] = σs[i]
    stn_db[stn]['freqs'] = freqs[i]

# Write dictionary to disk.
pickle.dump(stn_db, open('./data/stn_db.pkl', 'wb'))