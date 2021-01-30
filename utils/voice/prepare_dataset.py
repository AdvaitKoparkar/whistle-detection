import sys
sys.path.append('../../')

import os
import librosa
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf

from utils.voice.wav_utils import *

import pdb

class WhistleDataloader(object):
    def __init__(self, fg_files, bg_files, whistle_files, **kwargs):
        self.name = self.__class__.__name__
        self.fg_files = fg_files
        self.bg_files = bg_files
        self.whistle_files = whistle_files
        self.pattern = kwargs.get('pattern', '*.wav')
        self.wav_cfg = kwargs.get('wav_cfg', {'sr': 16000, 'mono': True, 'duration': 4})
        self.epoch_size = kwargs.get('epoch_size', 5000)
        self.shuffle = kwargs.get('shuffle', True)
        self.batch_size = kwargs.get('batch_size', 64)

    def _load_waveform_dset(self, fnames):
        if not self.epoch_size is None:
            fnames = np.random.choice(fnames, size=self.epoch_size, replace=False)
        max_len = self.wav_cfg['sr'] * self.wav_cfg['duration']
        X = np.zeros((len(fnames), max_len), dtype=np.float32)
        y = np.random.randint(low=0, high=2, size=(len(fnames),), dtype=np.uint8)
        for idx, fname in enumerate(tqdm(fnames)):
            w, s = self._load_waveform(fname)
            fname_bg = np.random.choice(self.bg_files)
            alpha = np.random.uniform(low=0, high=1)
            w_bg, s_bg = self._load_waveform(fname_bg)
            w = alpha * w + (1-alpha) * w_bg
            if y[idx] == 1:
                fname_wh = np.random.choice(self.whistle_files)
                w_whistle = self._get_whistle_data(fname_wh)
                beta = np.random.uniform(low=0.25, high=0.90)
                w = beta * w_whistle + (1-beta) * w
            X[idx, :] = w
        label_ds = tf.data.Dataset.from_tensor_slices(y)
        waveform_ds = tf.data.Dataset.from_tensor_slices(X)
        return waveform_ds, label_ds

    def prepare_epoch_dset(self):
        self.waveform_dset, self.label_dset = self._load_waveform_dset(self.fg_files)
        self.spectrogram_dset = self.waveform_dset.map(get_spectrogram)
        self.spectrogram_dset = self.spectrogram_dset.map(resize_spectrogram)
        dset = tf.data.Dataset.zip((self.spectrogram_dset, self.label_dset))
        if self.shuffle:
            dset = dset.shuffle(buffer_size=200)
        dset = dset.batch(self.batch_size)
        return dset

    def _get_whistle_data(self, fname):
        rev = np.random.uniform(low=0, high=1)
        # pitch_shift = np.random.choice([-5,-4,-3,-2,-1,0,1,2,3,4,5])
        w, s = self._load_waveform(fname)
        if rev > 0.5:
            w = w[::-1]
        return w

    def _load_waveform(self, file_path):
        waveform, sr = None, None
        cfg = self.wav_cfg
        max_len = cfg['sr'] * cfg['duration']
        if os.path.isfile(file_path):
            waveform, sr = librosa.load(file_path, **cfg)
            if waveform.shape[0] < max_len:
                waveform = pad_random(waveform, max_len)
        return waveform, sr
