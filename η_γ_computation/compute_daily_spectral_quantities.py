'''
SCRIPT compute_daily_spectral_quanities.py

This script grabs .SAC files stored in a directory. It then determines how many
unique ocean-bottom seismometers (OBSs) are represented by the data in that 
directory. It loops over each station that has data to be processed, then, 
computes several spectral quantities between the various components of the 
stations (pressure, vertical, and horizontals) for each day that the station 
has data. 

Daily spectral quantities, namely, auto- and cross-spectral densities b/w 
each of the components are written into a Python dictionary and written to an
output directory as .pkl files. Spectral densities are computed as follows:

 - A given day of OBS data (full 24-hours) is split into several (possibly 
   overlapping) windows.

 - Auto- and cross-spectral densities are computed b/w all components, for each
   window.

 - Individual windows are subjected to an F-test as a means of QC.

 - Only good windows are included in the final daily average of each spectral
   density.

A large portion of the code contained in this script was not written by me and 
comes directly from OBStools, developed and maintained by Pascal Audet, 
available at (https://github.com/nfsi-canada/OBStools). OBStools is itself a 
Python implementation of ATaCR, MATLAB code originally developed by Helen
Janiszewski, available at (https://github.com/helenjanisz/ATaCR). OBStools and
ATaCR are sophisticated suites of tools for processing and working with data
recorded by OBSs. 

The portions of code not written by me are clearly indicated below.

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# The usual suspects.
import pickle
import numpy as np
from obspy import read
from scipy.linalg import norm
from scipy.signal import spectrogram

# Several helper functions.
from utils import fetch, fourier, ftest, setup, smooth

##################################### SETUP ####################################

# Input and output directories. Output dir. created if doesn't already exist.
input_dir = '../data/raw_data/YL/'
output_dir = '../data/spectral/daily_spectral_quantities/'
setup.directory(output_dir)

# Parameters for computing spectral densities.
wlen_sec = 3600      # len of individual windows within a day [s]
olap_percent = 0.5   # window overlap [as a decimal fraction between 0 and 1]
minwin = 10          # minimum number of good windows required for a result.

##################################### MAIN #####################################

# Place all paths to files to be processed in a Python list.
fle_paths = fetch.data_paths(input_dir)

# Determine the stations that have data to be processed from the list of files.
stns = np.unique([path.split('/')[4].split('.')[6] for path in fle_paths])

# Loop over stations.
for stn in stns:

  # Filter files to only those that belong to the current station.
  stn_fles = [fle for fle in fle_paths if stn in fle]

  # Determine the days (time-keys) during which the station has data.
  stn_tks = np.unique([fle.split('/')[-1].split('YL')[0].split(':')[0] + ':00' for fle in stn_fles])
  
  # Loop and process data during days/time-keys that the station has data.
  for tk in stn_tks:
    
    # Handy print statement.
    print(stn, tk)

    # Filter the the stn_fles to only those files during the current day.
    current_day_fles = [fle for fle in stn_fles if tk in fle]

    # The order of the station's channels is critical! Ensure the following 
    # order: P, 1, 2, Z.
    current_day_fles.sort(key=lambda x: x.split('.')[10])

    # The above sorting doesn't always work. Need to address at some point. It's
    # also path dependent. Make more general in the future.

    # Read each station component as an ObsPy Trace Object, then group in list.
    trP = read(current_day_fles[0])[0]
    tr1 = read(current_day_fles[1])[0]
    tr2 = read(current_day_fles[2])[0]
    trZ = read(current_day_fles[3])[0]
    traces = [trP, tr1, tr2, trZ]

    # Length check. Sometimes traces have 1 point more or less than full day.
    # I can't remember why this is, it has something to do with DSP... Causes
    # trouble if not careful. 
    for tr in traces:

      # Check for 1 data point too many. If so, remove last point.
      if len(tr.data) - 86400 == 1:
        tr.data = tr.data[0:-1]

      # Check for 1 data point too few. If so, repeat final value and append.
      # Repeating the final value keeps spectra smooth when computing PSDs.
      if len(tr.data) - 86400 == -1: 
        last_val = tr.data[-1]
        tr.data = np.append(tr.data, last_val)

    # Grab station depth in meters (I like depth positive down)
    h = trZ.stats.sac.stel * -1

    ############ COMPUTATION OF AUTO- AND CROSS-SPECTRAL DENSITIES #############
    ################## CODE BELOW TAKEN DIRECTLY FROM OBStools #################

    # Credit to Pascal Audet and Helen Janizsewski et al.

    # Spectral QC.
    dt = trP.stats.sampling_rate                     # Station sampling [Hz]
    fs = trP.stats.delta                             # Station sampling [s]
    wlen_samples = int(wlen_sec / dt)                # Window length [samples]
    olap_samples = int(wlen_sec * olap_percent / dt) # Number overlap points
    
    # Construct a Hanning window with 2x the number of overlap samples.
    hanning = np.hanning(2 * olap_samples)
    window = np.ones(wlen_samples)
    window[0:olap_samples] = hanning[0:olap_samples]
    window[-olap_samples:wlen_samples] = hanning[olap_samples:wlen_samples]
    
    PSDs = []
    for tr in traces:
      f, t, psd = spectrogram(tr.data, fs, window=window, nperseg=wlen_samples, noverlap=olap_samples)
      PSDs.append(psd)
   
    # Select bandpass frequencies.
    ff = (f > 0.004) & (f < 2.0)

    # Smoothing
    for i, psd in enumerate(PSDs):
      
      # I added this check - SM. Need to handle zeros in psds, otherwise they
      # become problematic down the line. Set zeros to smallest nonzero value. 
      zero_check = np.sort(psd.flatten())
      if (zero_check == 0).any():
        smallest_non_zero = zero_check[zero_check != 0][0]
        psd[psd == 0] = smallest_non_zero 

      PSDs[i] = smooth.smooth(np.log(psd), 50, axis=0)

    # Remove mean of the log PSDs.
    for i, psd in enumerate(PSDs):
      PSDs[i] = psd[ff, :] - np.mean(psd[ff, :], axis=0)
  
    # Cycle through to kill high-std-norm windows.
    moveon = False
    good = np.repeat([True], len(t))
    indwin = np.argwhere(good == True)
    while moveon == False:
      ubernorm = np.empty((len(PSDs), np.sum(good)))
      for ind_u, psd in enumerate(PSDs):
        normvar = np.zeros(np.sum(good))
        for ii, tmp in enumerate(indwin):
          ind = np.copy(indwin)
          ind = np.delete(ind, ii)
          normvar[ii] = norm(np.std(psd[:, ind], axis=1), ord=2)
        ubernorm[ind_u, :] = np.median(normvar) - normvar
      penalty = np.sum(ubernorm, axis=0)
      kill = penalty > 2.0 * np.std(penalty)
      if np.sum(kill) == 0:
        moveon = True

      trypenalty = penalty[np.argwhere(kill == False)].T[0]
      if ftest.ftest(penalty, 1, trypenalty, 1) < 0.05:
        good[indwin[kill == True]] = False
        indwin = np.argwhere(good == True)
        moveon = False
      else:
        moveon = True

    if np.sum(good) < minwin:
      print("Too few good data segments to calculate average day spectra")
    else:
      print("{0} good windows. Proceeding...".format(np.sum(good)))
    
    # Compute spectra for each OBS component.
    ss = int(wlen_samples * (1 - olap_percent) / dt)
    ftP, f = fourier.calculate_windowed_fft(trP, wlen_samples, ss)
    ft1, f = fourier.calculate_windowed_fft(tr1, wlen_samples, ss)
    ft2, f = fourier.calculate_windowed_fft(tr2, wlen_samples, ss)
    ftZ, f = fourier.calculate_windowed_fft(trZ, wlen_samples, ss)

    # Compute auto-spectral quantities for good windows.
    cPP = np.abs(np.mean(ftP[good, :]*np.conj(ftP[good, :]), axis=0))[0:len(f)]
    c11 = np.abs(np.mean(ft1[good, :]*np.conj(ft1[good, :]), axis=0))[0:len(f)]
    c22 = np.abs(np.mean(ft2[good, :]*np.conj(ft2[good, :]), axis=0))[0:len(f)]
    cZZ = np.abs(np.mean(ftZ[good, :]*np.conj(ftZ[good, :]), axis=0))[0:len(f)]

    # Compute cross-spectral densities for good windows.
    c12 = np.mean(ft1[good, :]*np.conj(ft2[good, :]), axis=0)[0:len(f)]
    c1Z = np.mean(ft1[good, :]*np.conj(ftZ[good, :]), axis=0)[0:len(f)]
    c2Z = np.mean(ft2[good, :]*np.conj(ftZ[good, :]), axis=0)[0:len(f)]
    c1P = np.mean(ft1[good, :]*np.conj(ftP[good, :]), axis=0)[0:len(f)]
    c2P = np.mean(ft2[good, :]*np.conj(ftP[good, :]), axis=0)[0:len(f)]
    cZP = np.mean(ftZ[good, :]*np.conj(ftP[good, :]), axis=0)[0:len(f)]

    ##################### WRITE TO DICT. AND STORE TO DISK #####################
    
    # Dictionary of spectral quantities plus some helpful parameters.
    spectral_components = {'cPP': cPP,
                           'c11': c11,
                           'c22': c22,
                           'cZZ': cZZ,
                           'c12': c12,
                           'c1Z': c1Z,
                           'c2Z': c2Z,
                           'c1P': c1P,
                           'c2P': c2P,
                           'cZP': cZP,
                           'freqs': f,
                           'depth': h,
                           'npts': len(cPP),
                           'stn': stn,
                           'tk':tk}
    
    # Write spectral quantities for current stn,day to disk as a .pkl file.
    pickle.dump(spectral_components, open(output_dir+stn+'_'+tk+'.pkl','wb'))