# MDN Compliance Inversion
## Stephen Mosher (2022)

### INTRODUCTION

This repository consists of several scripts that I wrote during my time as PhD student at the University of Ottawa, in Canada. These scripts ultimately allow one to train one, or several, mixture density networks (MDNs) which can then be used to invert normalized compliance signals η(ω) recorded by ocean-bottom seismometers (OBSs). I have done my best to thoroughly comment and document all the code contained in this repository, such that anyone should be able to use/adapt this code to train their own MDN to invert η(ω) signals recorded by any OBS they wish. Below I provide a brief overview and description of all the components. If you use this code, in part or in full, please cite either one of my publications listed below ([1,2]), which are studies I conducted using this technique. Further details on this method or how I used it can be found in those publications.

### REQUIREMENTS

This code requires the following packages:

- NumPy
- SciPy
- ObsPy
- Matplotlib
- TensorFlow
- mdn (https://github.com/cpmpercussion/keras-mdn-layer)

In addition, there is a Fortran95 script involved (raydep_ft.f95), which needs to be compiled on your machine. In particular, the code related to that script was written with F2PY. F2PY is a Fortran to Python interface generator, from NumPy, that provides a connection between the two languages. In other words, it allows one to write functions in Fortran77/90/95, which, once compiled with F2PY, become callable as Python functions. The easiest way to accomplish this is to run the following in your terminal:

```
python -m numpy.f2py -c raydep_ft.f95 -m raydep_ft
```

**The code will not run without this step**.

### SOME NOTES AND ACKNOWLEDGEMENTS

Before describing the code and its various modules I need to make a few acknowledgements.

Significant portions of the code which compute compliance η(ω) and coherence γ(ω) curves from OBS data were not written by me and come directly from OBStools, developed and maintained by Pascal Audet, available at (https://github.com/nfsi-canada/OBStools). OBStools is itself a Python implementation of ATaCR, MATLAB code originally developed by Helen Janiszewski, available at (https://github.com/helenjanisz/ATaCR). OBStools and ATaCR are sophisticated suites of tools for processing and working with data recorded by OBSs. References are included below and in the code.

The portion of the code which forward computes η(ω) from synthetic Earth structures originally came from Wayne Crawford. Wayne's software is written in MATLAB and I translated it into Python to facilitate a fully Python implementation of this software. While translating Wayne's code, I made a few modifications of my own, most notably to the gravd function (details in [2]). I also translated the main function that performs the actual computation of η(ω) from synthetic Earth structures, raydep, from MATLAB to Fortran95. If compiled with F2PY from NumPy as noted above, then the function becomes callable as a Python function, and it provides a substantial speed up when forward computing η(ω). This is especially necessary in this application, where one needs to forward model η(ω) for 100,000+ training examples. References are included below and in the code.

Finally, regarding MDNs and the actual machine learning portion of this software, I used TensorFlow. However, at the time I was working on this project, TensorFlow did not have support for MDN layers in a neural network. Therefore, I used the mdn package written by Charles Martin (https://github.com/cpmpercussion/keras-mdn-layer). This package allows one to implement simple MDN layers in TensorFlow. References are included below and in the code.

### SOFTWARE STRUCTURE

This software is a set of Python scripts which can be grouped into the following categories, some of which are optional.

#### Step 1 - Computation of η(ω) and γ(ω) for real OBSs - OPTIONAL

This step is optional. It is only required if you desire to invert normalized compliance signals computed from real OBS data. Even then, if you already have a preferred means for doing this, such as ATaCR or OBStools, then it's not necessary. The only caveat would be that your data structure for your η(ω) and γ(ω) signals needs to match with what I've written. By contrast, if for some reason you only want to invert synthetic compliance signals, then you don't need to worry about any of this.

The scripts and associated functions that facilitate this task are found in the directory named η_γ_computation. Within that directory, there are 3 scripts and they should be run in the following order:

*compute_daily_spectral_quantities.py*
*compute_daily_η_γ.py*
*compute_stn_avg_η_γ.py*

The names of these scripts are self-descriptive, and their function is thoroughly documented in their comments. For copious details on what is going on here refer to [1] and [2] below.

One of the most important aspects of this step is that not only are η(ω) and γ(ω) signals computed for a group of OBSs, but, so too are their corresponding statistics. These statistics are crucial in the next step, as they form a solid basis for properly modelling synthetic signals. Otherwise, by default, synthetic compliance signal assume perfect pressure-vertical coherence (which is not realistic), and zero-noise.

#### Step 2 - Machine Learning Tasks

This step is the heart and soul of this project. The scripts and associated functions that facilitate these tasks are found in the main directory, MDN_compliance. There are 5 scripts and they should be run in the following order. Most of the names are self-explanatory (I think), but I'll describe a few important aspects below. Again, these are all thoroughly documented in the comments in the actual scripts themselves, and copious details on what is going on here can be found in [1] and [2] below.

*build_stn_db.py*

This script builds a station database to use in conjunction with other aspects of this project. In particular, since the frequency band over which η(ω) is measurable is both depth-dependant, and depends on γ(ω), several quantities need to be specified for every station/target water depth you wish to work with. If you don't have real signal statistics for η(ω) and γ(ω), or if you wish to work with purely synthetic data, then you can put your assumptions on statistics in here, along with names for dummy stations.

*build_train_test_data.py*

This script builds training and testing data for every station/target water depth you wish to work with. In principle, a single MDN could be trained to invert η(ω) for any station, deployed at any water depth, but this is a much move involved problem (discussed in [1] and [2] below). Therefore, the approach taken in this software is to build training data for every MDN you wish to consider.
This script is where the forward code for computing η(ω) from randomly generated Earth models gets called. All the code associated with the forward computation is contained in the forward_funcs directory.

*prep_MDN_data.py*

This script performs feature scaling on all training/testing data, and prepares data to be passed directly to an MDN for training.

*MDN_train.py*
*invert.py*

#### Step 0 - Data Acquisition - OPTIONAL

I've set up this repository so that you can clone it (or download the code), run the previously described scripts with all the default parameters, in the order I described, and reproduce the main inversion result for OBS station A02W from my publication listed below [1]. If you want to do this, then I would advise also cloning (or downloading) my request_data repository (https://github.com/s-g-mo/request_data). You should be able to run that with the default parameters and obtain all the data necessary to reproduce my results. You can then work through the above code to reproduce the result.

### REFERENCES/CITATIONS

#### Publications

[1] Mosher, S. G., Audet, P., & Gosselin, J. M. (2021). Shear-wave velocity structure of sediments on Cascadia's continental margin from probabilistic inversion of seafloor compliance data. Geochemistry, Geophysics, Geosystems, 22, e2021GC009720. https://doi.org/10.1029/2021GC009720

[2] S. G. Mosher, Z. Eilon, H. Janiszewski, P. Audet, Probabilistic inversion of seafloor compliance for oceanic crustal shear velocity structure using mixture density neural networks, Geophysical Journal International, Volume 227, Issue 3, December 2021, Pages 1879–1892, https://doi.org/10.1093/gji/ggab315

#### Software

##### OBStools

  - Pascal Audet
  - https://github.com/nfsi-canada/OBStools
  - https://doi.org/10.5281/zenodo.4281480

##### ATaCR

  - Helen Janiszewski
  - https://github.com/helenjanisz/ATaCR
- Helen A Janiszewski, James B Gaherty, Geoffrey A Abers, Haiying Gao, Zachary C Eilon, Amphibious surface-wave phase-velocity measurements of the Cascadia subduction zone, Geophysical Journal International, Volume 217, Issue 3, June 2019, Pages 1929–1948, https://doi.org/10.1093/gji/ggz051

##### Compliance Calculation Software

  - Wayne Crawford
  - http://www.ipgp.fr/~crawford/Homepage/Software.html

##### Mixture Density Network Layer

  - Charles Martin
  - https://github.com/cpmpercussion/keras-mdn-layer