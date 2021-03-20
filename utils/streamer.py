import sys
sys.path.append('../')

import json
import pyaudio

from utils.config_utils import read_config

class Streamer(object):
    '''
        class to stream and save audio data from mic
    '''
    def __init__(self):
        config = read_config()
        self.sample_rate =
