import sys
sys.path.append('../')

import unittest

from utils.config_utils import *

class TestConfigUtils(unittest.TestCase):
    '''
        test config utility
    '''
    def test_config_read(self):
        '''
            ensure json is in right format
        '''
        cfg = read_config()
        self.assertIsNotNone(cfg, "read_config returned None")

if __name__ == '__main__':
    unittest.main()
