import os
import librosa
import numpy as np
from glob import glob
from skimage.transform import resize
from librosa.feature import melspectrogram

import pdb

def pad_random(waveform, padded_size):
    wav_len = waveform.shape[0]
    waveform_padded = np.zeros((padded_size,), dtype=waveform.dtype)
    wav_start = np.random.randint(padded_size-wav_len)
    waveform_padded[wav_start:wav_start+wav_len] = waveform
    return waveform_padded

def resize_spectrogram(spectrogram):
    new_size = [64,64]
    spectrogram = resize(spectrogram, new_size,
                         mode='reflect', anti_aliasing=True)
    return spectrogram

def get_spectrogram(waveform, spec_cfg):
    spectrogram = np.abs(librosa.stft(waveform, **spec_cfg))
    # spectrogram = librosa.feature.chroma_stft(waveform)
    spectrogram = resize_spectrogram(spectrogram)
    return spectrogram

def find_files(base_folder, pattern='*.wav'):
    files = []
    for root, _, _ in os.walk(base_folder):
        matched_files = glob(os.path.join(root, pattern))
        files += matched_files
    return files
