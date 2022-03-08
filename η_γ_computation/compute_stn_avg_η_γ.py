'''
SCRIPT compute_stn_avg_η_γ.py

This script is to be run only after running compute_daily_η_γ.py

This takes daily normalized compliance η and coherence γ computed for an OBS and
computes an average single-station result. Importantly, it also computes the
statistics of these signals from the constituent daily signals.

The final η and γ signals and their statistics are stored in a Python dictionary
and written to disk.

Single-station η and γ curves can be plotted and saved to disk if desired.
Highly recommended.

Stephen Mosher, Feb. 2022
'''

#################################### IMPORTS ###################################

# The usual suspects.
import pickle
import numpy as np

# Helper functions.
from utils import fetch, misc, plot, setup, signal

##################################### SETUP ####################################

# Input and output directories. Output dirs created if don't already exist.
input_dir = '../data/spectral/daily_η_γ/'
output_dir = '../data/spectral/stn_avg_η_γ/'
plot_dir = '../figs/spectral/stn_avg_η_γ/'
setup.directory(output_dir)
setup.directory(plot_dir)

# Option to create output plots? If so, will create a plot of the single-station
# η and γ curves and their statistics.
output_plots = True

# Numpy array of compliance frequency bands for stations you're working with.
# There's no pretty way to do this. Since the frequency bandwidth over which η
# is measureable is depth-dependent, you need to determine this for each station
# you want to work with and store it either in another file somwhere, or here.
# the high-frequency cutoff has a theoretical description, but the low freq. 
# limit needs to be determined empirically. 
η_freq_band = np.array([[0.007, 0.024]])

'''
Multi-station example. Need to be sorted alphabetically to match stn sorting.
------------------------------------------------
η_freq_band = np.array([
                        [fmin stn1, fmax stn1],
                        [fmin stn2, fmax stn2],
                                  .
                                  .
                                  .            
                        [fmin stnN, fmax stnN],
                                       ])
'''

##################################### MAIN #####################################

# Store all paths to files to be processed in a Python list.
fle_paths = fetch.data_paths(input_dir)

# Determine the stations that have data to be processed from the list of files.
stns = np.unique([path.split('/')[-1].split('_')[0] for path in fle_paths])

# Loop over stations.
for i,stn in enumerate(stns):

  # Handy print statement.
  print(stn)

  # Extract frequency limits that apply to η given current station. 
  η_fmin = η_freq_band[i,0]
  η_fmax = η_freq_band[i,1]

  # Filter files to only those that belong to the current station.
  stn_fles = [fle for fle in fle_paths if stn in fle]

  # Extract frequency and station depth information from a representative stn
  # file. Choose the first file. This should be safe because the frequencies and 
  # depth for which daily ηs and γs were computed for a given station are const.
  freqs = pickle.load(open(stn_fles[0], 'rb'))['freqs']
  depth = pickle.load(open(stn_fles[0], 'rb'))['depth']

  # Initialize arrays to hold the all the daily ηs and γs computed for the stn.
  daily_ηs = np.zeros(shape=(len(stn_fles), len(freqs)))
  daily_γs = np.zeros(shape=(len(stn_fles), len(freqs)))

  # Loop over daily ηs and γs for current stn and populate arrays to hold all.
  for j, fle in enumerate(stn_fles):
    current_η = pickle.load(open(fle, 'rb'))['η']
    current_γ = pickle.load(open(fle, 'rb'))['γ']
    daily_ηs[j] = current_η
    daily_γs[j] = current_γ

  # η and γ have been computed for all possible frequencies, however, we're only
  # able to reliably measure η where the pressure-vertical coherence is high.
  # Therefore, we now selectively filter contributions to the single-station 
  # signals by limiting our attention to where the average γ across the 
  # compliance freq-band is > 0.95.
  
  # Find the indices of the freqs closest to η_fmin and η_fmax.
  low_idx = misc.idx_of_closest(η_fmin, freqs)
  high_idx = misc.idx_of_closest(η_fmax, freqs)

  # Selective filtering.
  daily_ηs = daily_ηs[np.mean(daily_γs[:,low_idx:high_idx], axis=1) > 0.95]
  daily_γs = daily_γs[np.mean(daily_γs[:,low_idx:high_idx], axis=1) > 0.95]

  # Νote, we still have full frequency information for the signals that survived
  # the selective filtering, this is nice for plotting!

  # Calculate the single-station average η and γ after having selectively 
  # filtered contributions according to the requirement for high γ.
  μ_η = np.mean(daily_ηs, axis=0)
  μ_γ = np.mean(daily_γs, axis=0)

  # Compute the 2.5 and 97.5% percentiles for observations as signal statistics.
  σ_η, σ_γ = signal.statistics(μ_η, μ_γ, daily_ηs, daily_γs)

  # Store average signals and their statistics in a dictionary. Write to disk.
  signals = {'μ_η':μ_η,
             'μ_γ':μ_γ,
             'σ_η':σ_η,
             'σ_γ':σ_γ,
             'freqs':freqs,
             'depth':depth,
             'stn':stn,
             'η_f_bounds':(η_fmin, η_fmax)}

  pickle.dump(signals, open(output_dir + stn + '.pkl', 'wb'))

  # Plot the final η and γ for the current station if desired.
  if output_plots:
    N = len(stn_fles)
    plot.stn_avg_η_γ(N, signals, η_fmin, η_fmax, low_idx, high_idx, plot_dir)

  