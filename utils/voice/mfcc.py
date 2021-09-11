import sys
sys.path.append('../')

import scipy
import numpy as np

from utils.config_utils import *
from utils.voice.wav_utils import *

def _f2mel(f):
    return 1125.0 * np.log(1.0 + f / 700.0)

def _mel2f(m):
    return 700.0 * (np.exp(m / 1125.0) - 1.0)

def generate_filterbanks(**kwargs):
    mfcc_cfg = kwargs.get('mfcc_cfg')
    if mfcc_cfg is None:
        mfcc_cfg = read_config()['mfcc_config']
    spec_cfg = mfcc_cfg['spectrogram_config']

    k = (spec_cfg['n_fft']+1)/spec_cfg['sr']
    bins  = np.floor(k * _mel2f(np.linspace(_f2mel(mfcc_cfg['low_freq']),
                                            _f2mel(mfcc_cfg['high_freq']),
                                            (mfcc_cfg['num_banks']+2))))
    filterbanks = np.zeros((mfcc_cfg['num_banks'], spec_cfg['n_fft']//2+1))

    for b in range(mfcc_cfg['num_banks']):
        c = bins[b]
        nb = ((bins[b+1:])[bins[b+1:] > c])[0]
        nnb = ((bins[b+2:])[bins[b+2:] > nb])[0]
        for f0 in range(int(c), int(nb)):
            filterbanks[b,f0] = (f0-c) / (nb-c)
        for f1 in range(int(nb), int(nnb)):
            filterbanks[b,f1] += (nnb-f1) / (nnb-nb)
    return filterbanks

def mfcc(waveform, **kwargs):
    mfcc_cfg = kwargs.get('mfcc_cfg')
    filterbanks = kwargs.get('filterbanks')
    if mfcc_cfg is None:
        mfcc_cfg = read_config()['mfcc_config']

    # step#1: spectrogram
    spectrogram = get_spectrogram(waveform, spec_cfg=mfcc_cfg['spectrogram_config'], resize=False, spec_only=True)

    # step#2: devide signal into filterbanks
    if filterbanks is None:
        filterbanks = generate_filterbanks(mfcc_cfg=mfcc_cfg)
    filterbank_en = filterbanks.dot(spectrogram)
    filterbank_en[filterbank_en == 0] += 1e-8
    log_filterbank_en = np.log(filterbank_en)

    # step#3: DCT+liftering
    mfcc = scipy.fftpack.dct(log_filterbank_en, type=2, axis=0, norm="ortho")[:mfcc_cfg['num_ceps']]
    if mfcc_cfg['apply_lifters']:
        lifter = 1 + mfcc_cfg['num_lifters'] / 2.0 * np.sin(np.pi * np.arange(mfcc_cfg['num_ceps']) / mfcc_cfg['num_lifters'])
        mfcc *= lifter[:, None]

    return mfcc
