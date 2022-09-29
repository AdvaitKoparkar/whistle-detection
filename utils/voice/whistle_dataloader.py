import sys
sys.path.append('../')

import os
import librosa
import numpy as np
from tqdm.notebook import tqdm
import tensorflow as tf

from utils.voice.mfcc import *
from utils.config_utils import read_config
from utils.voice.wav_utils import pad_random, get_spectrogram

class WhistleDataloader(tf.keras.utils.Sequence):
    def __init__(self, bg_files, wh_files, kwargs):
        self.name = self.__class__.__name__
        default_config = read_config()
        self.bg_files = bg_files
        self.wh_files = wh_files
        self.shuffle = kwargs.get('shuffle', True)
        self.batch_size = kwargs.get('batch_size', default_config['data_config']['batch_size'])
        self.epoch_size = kwargs.get('epoch_size', default_config['data_config']['epoch_size'])

    def _get_whistle_data(self, fname):
        rev = np.random.uniform(low=0, high=1)
        w, s = self._load_waveform(fname)
        if rev > 0.5:
            w = w[::-1]
        return w

    def _load_waveform(self, file_path):
        waveform, sr = None, None
        max_len = self.wav_cfg['sr'] * self.wav_cfg['duration']
        if os.path.isfile(file_path):
            waveform, sr = librosa.load(file_path, **self.wav_cfg)
            if waveform.shape[0] < max_len:
                waveform = pad_random(waveform, max_len)
        return waveform, sr

    def prepare_epoch(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

class WhistleDataloaderSpectrogram(WhistleDataloader):
    def __init__(self, bg_files, wh_files, **kwargs):
        super(WhistleDataloaderSpectrogram, self).__init__(bg_files, wh_files, kwargs)
        default_config = read_config()
        self.spectrogram_dim = kwargs.get('spectrogram_size', (default_config['spectrogram_config']['spectrogram_size'],default_config['spectrogram_config']['spectrogram_size']))
        self.wav_cfg = kwargs.get('wav_cfg', {'sr': default_config['sound_config']['sr'], 'duration': default_config['sound_config']['duration'], 'mono': True})
        self.spec_cfg = kwargs.get('spec_cfg', default_config['spectrogram_config'])

    def prepare_epoch(self):
        if self.shuffle:
            fnames = np.random.choice(self.bg_files, size=self.epoch_size)
        else:
            fnames = self.bg_files[0:self.epoch_size]

        self.epoch_fnames = fnames
        max_len = self.wav_cfg['sr'] * self.wav_cfg['duration']
        self.epoch_waveforms = np.zeros((len(fnames), max_len))
        self.epoch_X = np.zeros((len(fnames), self.spectrogram_dim[0], self.spectrogram_dim[1], 1), dtype=np.float32)
        self.epoch_y = np.random.randint(low=0, high=2, size=(len(fnames),), dtype=np.uint8)
        self.epoch_wts = np.ones((len(fnames),), dtype=np.float32)
        for idx, fname in enumerate(tqdm(fnames)):
            w, s = self._load_waveform(fname)
            if self.epoch_y[idx] == 1:
                fname_wh = np.random.choice(self.wh_files)
                w_whistle = self._get_whistle_data(fname_wh)
                beta = np.random.uniform(low=0.25, high=0.90)
                w = beta * w_whistle + (1-beta) * w
            self.epoch_waveforms[idx] = w
            spectrogram = get_spectrogram(w, self.spec_cfg)
            self.epoch_X[idx, :, :, 0] = spectrogram

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
        self.epoch = data_generator.flow(x=self.epoch_X, y=self.epoch_y,
                                          sample_weight=self.epoch_wts,
                                          batch_size=self.batch_size,
                                          shuffle=False)

    def __len__(self):
        return len(self.epoch)

    def __getitem__(self, idx):
        return self.epoch[idx]

class WhistleDataloaderMFCC(WhistleDataloader):
    def __init__(self, bg_files, wh_files, **kwargs):
        super(WhistleDataloaderMFCC, self).__init__(bg_files, wh_files, kwargs)
        default_config = read_config()
        self.wav_cfg = kwargs.get('wav_cfg', {'sr': default_config['sound_config']['sr'], 'duration': default_config['sound_config']['duration'], 'mono': True})
        self.mfcc_cfg = kwargs.get('mfcc_cfg', default_config['mfcc_config'])
        self.filterbanks = generate_filterbanks(mfcc_cfg=self.mfcc_cfg)

    def prepare_epoch(self):
        if self.shuffle:
            fnames = np.random.choice(self.bg_files, size=self.epoch_size)
        else:
            fnames = self.bg_files[0:self.epoch_size]

        self.epoch_fnames = fnames
        max_len = self.wav_cfg['sr'] * self.wav_cfg['duration']
        mfcc_cols = max_len //(self.mfcc_cfg['spectrogram_config']['hop_length']) + 1
        self.epoch_waveforms = np.zeros((len(fnames), max_len))
        self.epoch_X = np.zeros((len(fnames), self.mfcc_cfg['num_ceps'], mfcc_cols, 1), dtype=np.float32)
        self.epoch_y = np.random.randint(low=0, high=2, size=(len(fnames),), dtype=np.uint8)
        self.epoch_wts = np.ones((len(fnames),), dtype=np.float32)
        for idx, fname in enumerate(tqdm(fnames)):
            w, s = self._load_waveform(fname)
            fname_bg = np.random.choice(self.bg_files)
            alpha = np.random.uniform(low=0, high=1)
            w_bg, s_bg = self._load_waveform(fname_bg)
            w = alpha * w + (1-alpha) * w_bg
            if self.epoch_y[idx] == 1:
                fname_wh = np.random.choice(self.wh_files)
                w_whistle = self._get_whistle_data(fname_wh)
                beta = np.random.uniform(low=0.25, high=0.90)
                w = beta * w_whistle + (1-beta) * w
            self.epoch_waveforms[idx] = w
            m = mfcc(w, filterbanks=self.filterbanks)
            self.epoch_X[idx, :, :, 0] = m

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
        self.epoch = data_generator.flow(x=self.epoch_X, y=self.epoch_y,
                                              sample_weight=self.epoch_wts,
                                              batch_size=self.batch_size,
                                              shuffle=False)

    def __len__(self):
        return len(self.epoch)

    def __getitem__(self, idx):
        return self.epoch[idx]
