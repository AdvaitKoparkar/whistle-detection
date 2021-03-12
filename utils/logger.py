import os
from datetime import datetime
import tensorflow as tf

class Logger(tf.keras.callbacks.TensorBoard):
    '''
    '''
    def __init__(self, config, **kwargs):
        self.name = self.__class__.__name__
        # save name of the log file
        self.log_file = kwargs.get('log_file', None)
        # initiate the TensorBoard class
        super(Logger, self).__init__(**config)

    def log(self, msg, verbose=True):
        '''
        '''
        # if verbose, print to stdout
        if verbose:
            print(msg)
        # check if self.log_file is specified
        if self.log_file is None:
            return
        # append/create log_file with msg if exists
        with open(self.log_file, 'a+') as fh:
            fh.write("{}: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
