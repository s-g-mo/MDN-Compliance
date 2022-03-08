'''
SCRIPT MDN_train.py

This script loads training data that have been previously prepared and scaled,
such that they can be passed directly into a MDN for training.

This script specifies an MDN architecture, and trains a single MDN for as many
stations/situations as you wish to invert η for. MDNs are trained and saved to
disk.

The MDNs are trained for up to 1000 epochs, however, early-stopping and
"reduction on plateau" are implemented. 30% of the training data is set aside
for cross-validation at the end of each training epoch. Loss as a function of
training epoch is tracked and saved to disk in a plot.

Early-Stopping: if the validation loss doesn't improve after X epochs (8) then
                training ceases.

Reduction on Plateau: if the validation loss doesn't improve after X epochs (5)
                      then the learning rate is reduced by a factor of Y (0.1)

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# Handy module for working with MDNs - available @ https://github.com/cpmpercussion/keras-mdn-layer
import mdn

# The usual.
import pickle
import numpy as np
import matplotlib.pyplot as plt

# My helper functions.
from utils import fetch, plot, setup

# Imports from TensorFlow to keep things clean below.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

##################################### SETUP ####################################

# Input/Output directories
stn_db_dir = './data/'
train_dir = './data/ML/'
output_dir = './MDN_models/'
plot_dir = './figs/MDN_models/'
setup.directory(output_dir)
setup.directory(plot_dir)

# Train a single MDN for each station to be studied/modeled.
stn_db = pickle.load(open(stn_db_dir + 'stn_db.pkl', 'rb'))
stns = stn_db.keys()

# Loop over stations.
for stn in stns:
  print(stn)

  # Load train data.
  X = pickle.load(open(train_dir + stn + '/scaled/X_train.pkl','rb'))
  Y = pickle.load(open(train_dir + stn + '/scaled/Y_train.pkl','rb'))
  dimX = X.shape[-1]
  dimY = Y.shape[-1]
  
  ############################# NETWORK ARCHITECTURE ###########################
  
  H = 42                # Number of hidden units
  K = 6                 # Number of mixture components
  max_epochs = 1000     # Max number training epochs
  
  MDN = Sequential([
        Dense(H, input_shape=(dimX,), activation='swish'),
        Dense(H, activation='swish'),
        Dense(H, activation='swish'),
        Dense(H, activation='swish'),
        Dense(H, activation='swish'),
        mdn.MDN(dimY, K)])

  MDN.summary()
  
  #################################### TRAIN ###################################
  
  MDN.compile(loss=mdn.get_mixture_loss_func(dimY, K),
            optimizer=Adam(learning_rate=0.001)) # default rate is 0.001

  # Try some callbacks :)
  reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=5, mode='min')
  
  early = EarlyStopping(monitor='val_loss', patience=8, mode='min') 
  
  history = MDN.fit(x=X, y=Y, batch_size=32, epochs=max_epochs,
                    validation_split=0.3, callbacks=[early, reduce_LR])

  ################################# SAVE & PLOT ################################

  MDN.save(output_dir+stn+'.h5')
  plot.training_curves(history, plot_dir, stn)