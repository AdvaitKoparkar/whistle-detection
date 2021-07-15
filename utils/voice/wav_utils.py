import os
import librosa
import numpy as np
from glob import glob
from scipy.signal import stft
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
    # spectrogram = np.abs(librosa.stft(waveform, **spec_cfg))
    # spectrogram = librosa.feature.chroma_stft(waveform)
    # scipy
    _,_,spectrogram = stft(waveform, fs=spec_cfg['sr'],
                          nperseg=spec_cfg['n_fft'],
                          window=spec_cfg['window'])
    spectrogram = np.abs(spectrogram)
    spectrogram = resize_spectrogram(spectrogram)
    return spectrogram

def find_files(base_folder, pattern='*.wav'):
    files = []
    for root, _, _ in os.walk(base_folder):
        matched_files = glob(os.path.join(root, pattern))
        files += matched_files
    return files

def make_n_audio_snippets(src_file, n_snippets, **cfg):
    src_waveform, sr = librosa.load(src_file, sr=cfg['sr'])
    snippets = []
    snippet_duration_samples = int(cfg['duration'] * cfg['sr'])
    max_st_idx = src_waveform.shape[0] - snippet_duration_samples
    for sidx in range(n_snippets):
        st_idx = np.random.randint(low=0, high=max_st_idx, size=1)[0]
        _snippet = src_waveform[st_idx:st_idx+snippet_duration_samples]
        snippets.append(_snippet)
    return snippets

def clear_folder(folder, pattern='*.wav'):
    files = find_files(folder, pattern)
    for file in files:
        os.remove(file)
