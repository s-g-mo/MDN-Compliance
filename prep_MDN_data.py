'''
SCRIPT prep_MDN_data.py

This script loads individual training and testing examples created for training
and evaluating a MDN, respectively. It then creates np.arrays to hold all the 
examples, performs feature scaling on those examples, then saves them to disk.

This way inputs to the network can be preloaded and prepared to be passed
directly to a MDN. 

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# The usual.
import pickle
import numpy as np

# Helper function.
from utils import fetch, ML, setup

##################################### MAIN #####################################

# Stations.
stn_db = pickle.load(open('./data/stn_db.pkl', 'rb'))
stns = stn_db.keys()

# Loop over stations.
for stn in stns:
  
  # Input directory.
  input_dir = './data/ML/'+stn+'/'
  
  # Output directory for scaling parameters - created if not exists.
  output_dir = './data/ML/'+stn+'/scaled/'
  setup.directory(output_dir)
  
  # Build lists of paths to training examples.
  train_m_fles = fetch.data_paths(input_dir + 'train_models/')
  train_η_fles = fetch.data_paths(input_dir + 'train_signals/')
  
  # Build lists of paths to testing examples.
  test_m_fles = fetch.data_paths(input_dir + 'test_models/')
  test_η_fles = fetch.data_paths(input_dir + 'test_signals/')
  
  # Count number of training and testing examples.
  N_train = len(train_η_fles)
  N_test = len(test_η_fles)
  
  # Load a sample training example to get dimensional information.
  sample_η = pickle.load(open(train_η_fles[0], 'rb'))
  sample_m = pickle.load(open(train_m_fles[0], 'rb'))
  dimX = len(sample_η)
  dimY = len(sample_m['B'])
  
  # Initialize np arrays to hold all train/test examples.
  X_train = np.zeros(shape=(N_train, dimX)) # Observed η(ω) at inv freqs.
  Y_train = np.zeros(shape=(N_train, dimY)) # Bernstein coeffs for Vs(z).
  X_test = np.zeros(shape=(N_test, dimX))   #             .              
  Y_test = np.zeros(shape=(N_test, dimY))   #             .
  
  # Loop through training examples and populate X and Y, then perform scaling.
  for i, (η, m) in enumerate(zip(train_η_fles, train_m_fles)):
    print('Loading/prepping training example: ' + str(i))
    signl = pickle.load(open(η, 'rb'))
    model = pickle.load(open(m, 'rb'))
    X_train[i] = signl
    Y_train[i] = model['B']
  X_train = ML.feature_scaling(X_train, 'train', output_dir)
  
  # Loop through test examples and populate X and Y, then perform scaling.
  for i, (η, m) in enumerate(zip(test_η_fles, test_m_fles)):
    print('Loading/prepping test example: ' + str(i))
    signl = pickle.load(open(η, 'rb'))
    model = pickle.load(open(m, 'rb'))
    X_test[i] = signl
    Y_test[i] = model['B']
  X_test = ML.feature_scaling(X_test, 'test', output_dir)
  
  # Scaling parameters get written to disk for later use.
  
  # Write X,Y to disk
  pickle.dump(X_train, open(output_dir+'X_train.pkl', 'wb'))
  pickle.dump(X_test, open(output_dir+'X_test.pkl', 'wb'))
  pickle.dump(Y_train, open(output_dir+'Y_train.pkl', 'wb'))
  pickle.dump(Y_test, open(output_dir+'Y_test.pkl', 'wb'))