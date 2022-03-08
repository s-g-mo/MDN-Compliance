'''
SCRIPT compute_daily_η_γ.py

This script is to be run only after running compute_daily_spectral_quantities.py

Given a set of daily-averaged auto- and cross-spectral densities computed for
one or several OBSs, this script computes the daily normalized compliance η and
coherence γ signals for each station, for each day. 

The η and γ data are stored in a Python dictionary and written to disk.

Daily η and γ curves can be plotted and saved to disk if desired. I find this
extremely helpful, as it facilitates a quick check that everything is working
properly. If η and γ look like garbage, something is not working prior to this.

Stephen Mosher, Feb. 2022
'''

#################################### IMPORTS ###################################

# The usual suspects.
import pickle
import numpy as np

# Helper functions.
from utils import compliance_coherence, fetch, plot, setup

##################################### SETUP ####################################

# Input and output directories. Output dirs created if don't already exist.
input_dir = '../data/spectral/daily_spectral_quantities/'
output_dir = '../data/spectral/daily_η_γ/'
plot_dir = '../figs/spectral/daily_η_γ/'
setup.directory(output_dir)
setup.directory(plot_dir)

# Option to create output plots? If so will create plot of η and γ for each day.
output_plots = True

##################################### MAIN #####################################

# Store all paths to files to be processed in a Python list.
fle_paths = fetch.data_paths(input_dir)

# Determine the stations that have data to be processed from the list of files.
stns = np.unique([path.split('/')[-1].split('_')[0] for path in fle_paths])

# Loop over stations.
for stn in stns:

  # Filter files to only those that belong to the current station.
  stn_fles = [fle for fle in fle_paths if stn in fle]

  # Determine the days (time-keys) during which the station has data.
  stn_tks = np.unique([fle.split('_')[-1].split('.pkl')[0] for fle in stn_fles])
  
  # Loop and process data during days/time-keys that the station has data.
  for tk in stn_tks:

    # Handy print statement.
    print(stn, tk)

    # Filter the the stn_fles to only those files during the current day.
    current_spectra = [fle for fle in stn_fles if tk in fle][0]

    # Load dictionary containing current spectral quantities.
    spectral_components = pickle.load(open(current_spectra, 'rb'))

    # Extract frequency information and station depth from spectral dictionary.
    freqs = spectral_components['freqs']
    depth = spectral_components['depth']

    # Compute coherence and normalized compliance from the daily spectral comps.
    η, γ = compliance_coherence.η_γ(spectral_components)

    # Store η and γ in a dictionary. Write to disk.
    data = {'η': η, 'γ': γ, 'freqs': freqs, 'depth': depth, 'stn': stn, 'tk':tk}
    pickle.dump(data, open(output_dir + stn + '_' + tk + '.pkl','wb')) 

    # Plot the current η and γ for the current station if desired.
    if output_plots:
      plot.η_γ_curves(freqs, depth, stn, tk, η, γ, plot_dir)
