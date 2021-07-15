import sys
sys.path.append('../')

import os
import json

import pdb

DEFAULT_CONFIG_PATH = os.path.join('..', 'config', 'whistle_system_config.json')

def read_config():
    '''
        reads config and return dictionary
    '''
    cfg = None
    if not os.path.isfile(DEFAULT_CONFIG_PATH):
        return cfg
    with open(DEFAULT_CONFIG_PATH, 'rb') as fh:
        cfg = json.load(fh)
    return cfg
