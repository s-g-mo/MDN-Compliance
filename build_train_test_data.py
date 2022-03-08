'''
SCRIPT build_train_test_data.py

This script builds training and testing examples used to train a MDN and then
evaluate its performance.

A single training/testing example consists of a synthetic Earth structure, that
is (Vp(z), Vs(z), and ρ(z)) and the synthetic η signal foward computed for that
structure. However, this approach simplifies things by either fixing Vp(z) and
ρ(z) to be constant, or defining them in terms of some relationship with Vs(z).
Vs(z) is in turn parameterized in terms of coefficients of a Bernstein 
polynomial basis of a given order.

The model constructor function that gets called to build and forward compute
examples below enforces a monotonicity constraint on generated Vs profiles.
If you increase the order of the Bernstein polynomial basis, the time taken to
generate Vs profiles that satisfy the monotonicity constraint will significantly
increase. This is because polynomials of increasing order are more "wiggly" and
likely to feature negative Vs gradients. In my experience it's best not to use
Bernstein polynomials of order > 4.

Alternatively, you can try and remove the monotonicity constraint, but it will
be much more work to successfully train a MDN for compliance inversion. 

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# The usual suspects.
import pickle
import numpy as np

# Import helper functions.
from utils import fetch, ML

##################################### SETUP ####################################

# For consistency while prototyping/testing. Comment out if you like.
np.random.seed(0)

# Input/output directories. Output directories created if don't exist.
stn_db = pickle.load(open('./data/stn_db.pkl', 'rb'))
output_dir = './data/ML/'

# Required parameters.
Nm_train = 100000                # Number of training models
Nm_test = 30000                  # Number of testing models
zmax = 2000                      # Max Vs structural depth [m]
order = 3                        # Bernstein polynomial order
Nf = 6                           # Number of inversion frequencies
low = 0.1                        # Minimum Vs value.
high = 3.0                       # Maximum Vs value.
plot = True                      # Show plots of models? Useful to test. Will
                                 # only show 3 models.

##################################### MAIN #####################################

# Loop over stations and corresponding data contained in stn_db.
for stn, data in stn_db.items():

  # Construct randomly generated training models for current station/depth.
  ML.model_constructor(data, zmax, Nm_train, Nf, low, high, order, plot, output_dir+stn+'/train_')
  
  # Construct randomly generated testing models for current station/depth.
  ML.model_constructor(data, zmax, Nm_test, Nf, low, high, order, plot, output_dir+stn+'/test_')


  
  