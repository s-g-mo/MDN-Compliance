'''
SCRIPT invert.py

For each OBS/synthetic situation that you're interested in, this script will
call the relevant MDN that was trained to invert compliance and invert a 
synthetic or measured compliance signal.

Recall that the output of a MDN is a GMM. Therefore, given an input signal
(real or otherwise), the MDN provides a GMM, which can be sampled to determine
a distribution of Vs(z) profiles for the input signal. Statistics are computed
from the distribution of signals. All of this is plotted and saved to disk. 

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# Handy module for working with MDNs - available @ https://github.com/cpmpercussion/keras-mdn-layer
import mdn

# The usual.
import pickle
import numpy as np
import tensorflow as tf

# Helper functions.
from utils import fetch, misc, ML, plot, setup, structural

##################################### SETUP ####################################

# Input/Output directories.
signal_dir = './data/spectral/stn_avg_η_γ/'
network_dir = './MDN_models/'
plot_dir = './figs/inversion_results/'
setup.directory(plot_dir)

# Specify stations.
stns = ['A02W']

# Have to specify the number of Gaussian mixture model components used when
# training the MDNs (in principle, could be a list of different Ks).
K = 6

# The number times to sample from the GMM output by the MDNs.
N_samples = 1000

# Loop over stations and the MDNs trained for each station/situation.
for stn in stns:

  # Station-dependant directories.
  model_dir = './data/ML/' + stn + '/train_models/'
  scaling_dir = './data/ML/' + stn + '/scaled/'
  
  # Load a training example (first one) to grab a bunch of useful parameters
  model_fle = fetch.data_paths(model_dir)[0]
  sample_model = pickle.load(open(model_fle, 'rb'))
  
  # Extract training model parameters.
  zmax = sample_model['max_z_m']
  dimX = sample_model['dimX']
  dimY = sample_model['dimY']
  inv_freqs = sample_model['inv_freqs']

  # Order of Bernstein polynomials.
  order = dimY - 1

  # Load measured/synthetic compliance signal to be inverted.
  measured_data = pickle.load(open(signal_dir + stn + '.pkl', 'rb'))

  # Slice measured/synthetic η at the inversion frequencies.
  freqs = measured_data['freqs']
  inv_freq_idxs = [misc.idx_of_closest(target, freqs) for target in inv_freqs]
  η = measured_data['μ_η'][inv_freq_idxs]

  # Scale measured signal (it must be treated the same way the training signals
  # were treated).
  μ_scaling = pickle.load(open(scaling_dir + 'μ_train.pkl', 'rb'))
  σ_scaling = pickle.load(open(scaling_dir +'σ_train.pkl', 'rb'))

  # The input into the trained MDN is the measured compliance signal, scaled and
  # treated in the same manner as were the training signals.
  X = ML.scale_real_input(η, μ_scaling, σ_scaling)

  # Initialize arrays to hold Bernstein basis coefficients obtained from each
  # sample of the GMM and the corresponding Vs profiles.  
  coeffs = np.zeros(shape=(N_samples, dimY))
  profiles = np.zeros(shape=(N_samples, zmax))
  
  ################################ LOAD NETWORK ################################
  
  MDN = tf.keras.models.load_model(network_dir + stn + '.h5',  
                          custom_objects={'MDN': mdn.MDN,
                          'mdn_loss_func': mdn.get_mixture_loss_func(dimY, K)})
  
  ########################### PREDICT GMM PARAMETERS ###########################
  
  # The Guassian mixture model predicted by the MDN, based on input X.
  GMM = MDN.predict(X.reshape(1, dimX))
  
  ''' 
  Helpful to know how to get individual GMM components if desired...

  # Extract the GMM parameters (means μ, standard deviations σ, mixture
  # coefficients π.
  μ = np.apply_along_axis((lambda a: a[:K*dimY]), 1, GMM)
  σ = np.apply_along_axis((lambda a: a[K*dimY:2*K*dimY]), 1, GMM)
  π = np.apply_along_axis((lambda a: mdn.softmax(a[2*K*dimY:])), 1, GMM)

  '''
  
  ################################## ANALYSIS ##################################
  
  # Sample the GMM output by the MDN and record the Bernstein basis coefficients
  # of each sample.
  for i in range(N_samples):
    print('Sampling from GMM learned by the MDN: ' + str(i + 1))
    coeffs[i] = mdn.sample_from_output(GMM[0], dimY, K)
  
  # Compute the mean of the sampled coefficients.
  μ_coeffs = np.mean(coeffs, axis=0)

  # Construct the Vs profile that corresponds to the mean coeffs.
  z = np.linspace(0, zmax/1000, zmax)
  μ_profile = structural.bernstein_profile(z/(zmax/1000), order, μ_coeffs)
  
  # Νοw generate ALL profile estimates from ALL coefficient samples.
  profiles = [structural.bernstein_profile(z/(zmax/1000), order, c) for c in coeffs]
  profiles = np.array(profiles)

  # Compute 95% levels of the sampled profiles. L - 2.5% bounds, U - 97.5% bounds
  L, U = misc.percentile_levels(profiles) 

  ############################### ASSESSMENT PLOT ##############################

  plot.inversion_result(profiles, μ_profile, zmax, order, U, L, stn, plot_dir)
  