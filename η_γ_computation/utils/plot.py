'''
FUNCTION SET plot.py

A set of functions to create plots of η and γ for OBSs.

Stephen Mosher, Mar. 2022
'''
#################################### IMPORTS ###################################

# The usual suspects.
import numpy as np
import matplotlib.pyplot as plt

################################### FUNCTIONS ################################## 

def η_γ_curves(freq, depth, stn, tk, η, γ, output_dir):
  
  '''
  Plots normalized compliance η and coherence γ curves for an OBS.
  '''

  # Plot setup.
  fig, axes = plt.subplots(2, 1, figsize=(7,12))
  ax1 = axes[0]
  ax2 = axes[1]
  
  # Log-log plot for η.
  ax1.loglog(freq, η, linewidth=1.25)
  ax1.set_title(stn + ' Depth: ' +  str(int(depth)) + ' m, ' + 'Date: ' + ','.join(tk.split('.')[0:2]))
  ax1.set_ylabel('$η(f)$')
  
  # Semi-log plot for γ.
  ax2.semilogx(freq, γ, linewidth=1.25)
  ax2.set_xlabel('Frequency [Hz]')
  ax2.set_ylabel('$γ_{PZ}$')
  plt.savefig(output_dir + stn + '_' + tk + '.png')
  plt.close()

def stn_avg_η_γ(N, params, fl, fc, low_idx, high_idx, output_dir):
  
  '''
  Plots the average single-station η and γ curve computed from a set of daily
  averaged η and γ signals recorded by an OBS. Plots signal statistics too.

  '''

  # Extract parameters
  μ_η = params['μ_η']
  μ_γ = params['μ_γ']
  σ_η = params['σ_η']
  σ_γ = params['σ_γ']
  freq = params['freqs']
  depth = str(int(params['depth']))
  stn = params['stn']

  # Plot setup.
  fig, axes = plt.subplots(2, 1, figsize=(6,8))
  ax1 = axes[0]
  ax2 = axes[1]

  # Plot η in log-log space.
  ax1.loglog(freq, μ_η, linewidth=1.25)

  # Shade the curve a nice semi-transparent light blue bewtween its 95% limits.
  ax1.fill_between(freq, σ_η[:,0] * μ_η, σ_η[:,1] * μ_η, alpha=0.2)

  # Plot compliance frequency limits.
  ax1.axvline(fl, ls='--', color='#1C2360')
  ax1.axvline(fc, ls='--', color='#C5271A')

  # Use the compliance frequency limits to shade the plot a nice semi-trans.
  # cyan to indicate the compliance frequency band for the stn.
  ax1.fill_between(freq[low_idx:high_idx],
                   np.zeros(len(μ_η[low_idx:high_idx])),
                   μ_η[low_idx:high_idx],
                   alpha=0.2,
                   color='cyan',
                   zorder=-2)
 
  # Plot title and Y-axis.
  ax1.set_title(stn + ' Depth=' +  depth + ' m, ' + ' $N_{Obs.}=$'+str(N))
  ax1.set_ylabel('$η(f)$')

  # Plot γ in semi-log space.
  ax2.semilogx(freq, μ_γ, linewidth=1.25)
  
  # Shade the curve a nice semi-transparent light blue bewtween its 95% limits.
  ax2.fill_between(freq, σ_γ[:,0] * μ_γ, σ_γ[:,1] * μ_γ, alpha=0.2)
  
  # X-axis and Y-axis
  ax2.set_xlabel('Frequency [Hz]')
  ax2.set_ylabel('$γ_{PZ}$')

  # Plot compliance frequency limits.
  ax2.axvline(fl, ls='--', color='#1C2360')
  ax2.axvline(fc, ls='--', color='#C5271A')

  # Use the compliance frequency limits to shade the plot a nice semi-trans.
  # cyan to indicate the compliance frequency band for the stn.
  ax2.fill_between(freq[low_idx:high_idx],
                   np.zeros(len(μ_γ[low_idx:high_idx])),
                   μ_γ[low_idx:high_idx],
                   alpha=0.2,
                   color='cyan',
                   zorder=-2)

  plt.tight_layout()
  plt.savefig(output_dir + stn + '.png')
  plt.close()
  plt.clf()