'''
FUNCTION SET plot.py

A set of plotting functions.

Stephen Mosher, Mar. 2022
'''
#################################### IMPORTS ###################################

# The usual suspects.
import numpy as np
import matplotlib.pyplot as plt

################################### FUNCTIONS ################################## 


# Plot a training or testing model (Vs, Vp, ρ).
def model(z_m, Vp, Vs, ρ):

  # Convert max depth in meters to km.
  z_km = z_m/1000

  # Create an array of depth values in km, discretized to 1 m
  z = np.linspace(0, z_km, z_m)
  
  # Plot setup.
  fig, ax = plt.subplots(figsize=(6,8))
  
  # Plot Vp, Vs, and ρ as a function of depth.
  ax.plot(Vp, z, label='$V_P$')
  ax.plot(Vs, z, label='$V_S$')
  ax.plot(ρ, z, label='$ρ$')

  # Make plot pretty (labels, legend, limits, etc.)
  ax.set_xlim(0, np.amax([Vs, Vp]) + 0.5)
  ax.set_ylabel('Depth [km]')
  ax.set_xlabel('$V_S$ [km/s] / ' + '$ρ$ [g/cm$^3$]')
  ax.xaxis.tick_top()  
  ax.xaxis.set_label_position('top')
  ax.invert_yaxis()
  plt.legend()
  plt.show()

# Plot MDN training curves, post training.
def training_curves(history, outdir, stn):

  # Plot loss and validation loss.
  plt.plot(history.history['loss'], linewidth=1.5)
  plt.plot(history.history['val_loss'], linewidth=1.5)

  # Make plot pretty.
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Training', 'Validation'], loc='upper right')

  # Save.
  plt.savefig(outdir+stn+'_training.png')
  plt.clf()
  plt.close()

# Plot an inversion result (the mean Vs profile, sample profiles, and confidence)
def inversion_result(profiles, μ_Vs, zmax, order, U, L, stn, output):

  # Plot setup.
  fig, ax = plt.subplots(figsize=(6,8))

  # Depth axis.
  z = np.linspace(0, zmax/1000, zmax)
  
  # Plot each sampled profile in orange, transparent. Place bottom layer.
  for Vs in profiles:
    ax.plot(Vs, z, color='orange', alpha=0.05, zorder=0)
  
  # Plot 95% levels of sampled profiles in black.
  ax.plot(U, z, 'k')
  ax.plot(L, z, 'k')

  # Plot mean Vs profile from samples.
  ax.plot(μ_Vs, z, 'b', ls='--', label='Mean $V_S$ Recovered')

  # Plot labels and Title
  ax.set_ylabel('Depth [km]')
  ax.set_xlabel('$V_S$ [km/s]')

  # Axis limits.
  ax.set_xlim(0, np.amax(profiles + 1.0))
  
  # Depth down.
  ax.xaxis.tick_top()  
  ax.xaxis.set_label_position('top')
  ax.invert_yaxis()
  
  # Legend. Save.
  plt.legend()
  plt.savefig(output + stn + '.png')
  plt.close()