import sys
sys.path.append("../../")

import os
import tensorflow as tf
from utils.logger import Logger
from utils.metrics import Metrics

import pdb

class Experiment(object):
    '''
    '''
    def __init__(self, config):
        # set up experiment parameters
        self.name = self.__class__.__name__
        self.logger_config = config.get('logger_config', None)
        self.metrics_set = config.get('metrics_set', None)

        # set up dset, model and training paramters
        self.dset = config['dset']
        self.input_config = config['input_config']
        input_shape = self.input_config['input_shape']
        batch_size = self.input_config['batch_size']
        self.model = config['model_generator'].get_model(input_shape=input_shape, batch_size=batch_size)
        self.experiment_name = '{}'.format(config['model_generator'].name)
        self.run_id = config.get('run_id', 0)
        self.optimizer = config.get('optimizer', None)
        self.loss_object = config.get('loss_object', None)
        self.lr_params = config.get('lr_params', None)

        # set up logging and metrics utilites
        if self.logger_config is None:
            self.logger_config = {'histogram_freq':0,
                                  'write_graph':True,
                                  'write_images':False,
                                  'update_freq':'epoch'
                                  }

        # experiment directory (../logs/<experiment_name>_<run_id>)
        # create a .log text file in ../logs/ directory
        # instantiate logger object
        self.logger_config['log_dir'] = os.path.join(os.path.abspath(r'..'), 'logs', '{}_{}'.format(self.experiment_name,str(self.run_id)))
        self.logger = Logger(self.logger_config, log_file=os.path.join(os.path.abspath('..'), 'logs', '{}_{}.log'.format(self.experiment_name,str(self.run_id))))
        # instantiate metrics object
        self.metrics = Metrics(self.metrics_set)

    def train(self):
        raise NotImplementedError
