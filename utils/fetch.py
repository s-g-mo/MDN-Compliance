'''
FUNCTION fetch.py

Grabs all files in a directory that match a given key.

Returns absolute paths to those files as a Python list.

Stephen Mosher, Feb. 2022
'''

#################################### IMPORTS ###################################

import glob

################################### FUNCTION ###################################

def data_paths(data_dir, matchkey='*'):
  files = glob.glob(data_dir + matchkey)
  return sorted(files)