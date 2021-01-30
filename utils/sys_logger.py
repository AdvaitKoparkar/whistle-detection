'''
'''

from datetime import datetime

class SysLogger(object):
    def __init__(self, log_path, **kwargs):
        self.log_path = log_path

    def log(self, msg, verbose=True, newline=True):
        prefix = '[{}] '.format(datetime.now().strftime('%Y%m%d %H:%M:%S'))
        msg = prefix + msg
        if verbose:
            print(msg)
        if newline:
            msg = msg + '\n'
        with open(self.log_path, 'a+') as fh:
            fh.write(msg)

if __name__ == '__main__':
    log_path = r'D:\Advait\Handouts_and_assignments\Raspberry Pi\voice_activated_camera\logs\20201212_log.txt'
    logger = SysLogger(log_path)
    logger.log('test1')
    logger.log('test2')
