'''
FUNCTION setup.py

A simple function to build a directory if it doesn't already exist.

Stephen Mosher (2020)
'''

#################################### IMPORTS ###################################

import os

################################### FUNCTION ###################################

def directory(dir_name):
  if not os.path.isdir(dir_name):
    print('\n Directory ' + dir_name + ' doesn`t exist - creating it.')
    os.makedirs(dir_name)