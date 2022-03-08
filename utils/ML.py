'''
FUNCTION SET ML.py

A set of functions to for various aspects of this project related to machine 
learning, such as testing/training model construction, model preparation, etc.

Stephen Mosher, Mar. 2022
'''

#################################### IMPORTS ###################################

# The usual.
import time
import pickle
import numpy as np

# Forward modelling code
from forward_funcs import ncomp_fortran

# Helper functions.
from utils import misc, ML, plot, setup, structural

################################### FUNCTIONS ##################################

def model_constructor(data, zmax, Nm, Nf, low, high, order, test_plot, outdir):
  '''
  This function constructs "examples" for machine learning applications.

  Examples consist of an Earth structure, which, for us is Vs(z), parameterized
  using Bernstein polynomial coefficients, and its forward computed η signal.

  The function that performs the forward computation was translated, by myself,
  from MATLAB code origianlly written by Wayne Crawford. His original code can
  be found at http://www.ipgp.fr/~crawford/Homepage/Software.html
  '''

  # Setup output directories for both the structural models and the signals.
  setup.directory(outdir+'models/')
  setup.directory(outdir+'signals/')

  # Extract parameters for current depth-context.
  stn = data['stn']
  γ = data['γ']
  σ = data['σ']
  h = data['depth']
  η_freqs = data['freqs']
  η_fmin = data['η_f_bounds'][0]
  η_fmax = data['η_f_bounds'][1]

  # Determine which frequencies η will be forward computed for by linearly
  # spacing Nf frequencies between the η bandwidth limits established for this
  # depth. 
  inv_freqs = np.linspace(η_fmin, η_fmax, Nf)

  # γ and σ have been computed over the full range of frequencies for which
  # η was computed. But the forward code below only computes η at Nf freqs.
  # So we need to know what frequencies η was computed over, and thus, their
  # indices, so we can pick them out and match them up with the inversion freqs.
  idxs_of_query_freqs = [misc.idx_of_closest(f, η_freqs) for f in inv_freqs]

  # Limit γ and ε to the same frequencies for which η was forward computed.
  γ = γ[idxs_of_query_freqs]
  σ = σ[idxs_of_query_freqs]

  # Array of model depths in [km], discretized at 1m intervals.
  z = np.linspace(0, zmax/1000, zmax)

  # Initialize a timer.
  t1 = time.time()

  # Initialize a model counter.
  j = 0

  # Loop until Nm models have been successfully created.
  while j < Nm:

    print('generating a random model...')
    # Generate random Bernstein coefficients on the interval [low, high]
    coeff = np.random.uniform(low=low, high=high, size=order+1)

    # Construct random Vs profile from the Bernstein coefficients.
    Vs = structural.bernstein_profile(z/(zmax/1000), order, coeff)

    # Enforce monotonicity constraint.
    print('enforcing monotonicity...')
    if (np.diff(Vs) < 0).any():
      print('monotonicity condition violated...')
      continue

    # Compute Vp and ρ from Vs (kept simple here, more options in 
    # ./utils/structural.py)
    Vp = np.ones(len(Vs)) * 6.0
    ρ = np.ones(len(Vs)) * 2.0       # Following Zha and Webb, 2016.
    
    # Plot a model?
    if test_plot == True:
      if j < 3:
        plot.model(zmax, Vp, Vs, ρ)
  
    # Forward compute normalized compliance of model using my translation of
    # Wayne Crawford's code (location of source indicated in title block).
    
    # Layer thicknesses in meters (we're effectively assuming 1m thicknesses)
    thicknesses = np.ones(zmax)
    thicknesses = thicknesses.reshape(len(thicknesses), 1)

    ρ = ρ.reshape(thicknesses.shape)
    Vp = Vp.reshape(thicknesses.shape)
    Vs = Vs.reshape(thicknesses.shape)

    structure = np.hstack([thicknesses, ρ, Vp, Vs])

    # Call to fortran code to compute η. It's faster than MATLAB but needs to 
    # be compiled on your machine.
    η = ncomp_fortran.ncomp_fortran(depth=h, freqs=inv_freqs, model=structure)
    
    # Noise.
    ε = np.random.uniform(low=σ[:,0], high=σ[:,1])
    
    # Weight forward computed signal by γ and apply noise.
    η = γ * η * ε
  
    # Some sanity checks.
    if (η < 0).any():
      continue
    if (η == 0).any():
      continue
    if np.isnan(η).any():
      continue

    # If model survives monotonicity constraint and sanity, print out statement.
    model_type = outdir.split('/')[-1].split('_')[0]+'ing'
    print('generated '+ model_type +' model: ' + str(j) + ', ' + stn)
  
    # Write model components into a dictionary.
    model = {'Vs': Vs,
             'B': coeff,
             'max_z_km': zmax/1000,
             'max_z_m': zmax,
             'inv_freqs': inv_freqs,
             'h': h,
             'dimX': len(inv_freqs),
             'dimY': len(coeff)}
    
    # Use model counter as an id.
    number = str(j)

    # Write both the model and signal to disk.
    model_fpath = outdir + 'models/mod_' + number + '.pkl'
    pickle.dump(model, open(model_fpath, 'wb'))
  
    # Save signal.
    pickle.dump(η, open(outdir + 'signals/sig_' + number + '.pkl', 'wb'))
  
    # Increase j
    j += 1
  
  # Let's see how long it takes to make the models.  
  t2 = time.time()
  print('Total Time: ' + str(t2 - t1), 'seconds for', j, 'models')

def feature_scaling(X, dset, outdir):

  '''
  Feature-scaling for MDN data. The scaling parameters get written to disk
  as they are required when presenting new examples to the trained network.
  '''

  # I found this critically important for getting the MDN to work!
  X = np.log10(X)
  
  # Record and dump scaling parameters - need later.
  μ = np.mean(X, axis=0)
  σ = np.std(X, axis=0)
  pickle.dump(μ, open(outdir + '/μ_'+ dset +'.pkl', 'wb'))
  pickle.dump(σ, open(outdir + '/σ_'+ dset +'.pkl', 'wb'))
  
  # Scale.
  X = (X - μ)/σ

  print('writing scaling parameters to disk...')

  return(X)

def scale_real_input(η, μ, σ):
  '''
  Feature-scaling for a real compliance signal to be passed to a trained MDN.
  '''
  return (np.log10(η) - μ)/σ