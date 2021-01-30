import sys
sys.path.append('../../')

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.voice.wav_utils import find_files

def _split_folder(base_folder, splits, seed):
    files = find_files(base_folder)
    train, test = train_test_split(files, random_state=seed, test_size=splits['test'])
    return train, test

def split_data(base_folders, seed=None):
    train_folder = r'..\data\csv\train'
    test_folder = r'..\data\csv\test'

    fg_folder = base_folders['fg_folder']
    fg_split = {'test': 0.3}
    fg_train, fg_test = _split_folder(fg_folder, fg_split, seed)
    pd.DataFrame({'fname': fg_train}).to_csv(os.path.join(train_folder, 'fg.csv'), index=False)
    pd.DataFrame({'fname': fg_test}).to_csv(os.path.join(test_folder, 'fg.csv'), index=False)

    bg_folder = base_folders['bg_folder']
    bg_split = {'test': 0.3}
    bg_train, bg_test = _split_folder(bg_folder, bg_split, seed)
    pd.DataFrame({'fname': bg_train}).to_csv(os.path.join(train_folder, 'bg.csv'), index=False)
    pd.DataFrame({'fname': bg_test}).to_csv(os.path.join(test_folder, 'bg.csv'), index=False)

    wh_folder = base_folders['whistle_folder']
    wh_split = {'test': 0.3}
    wh_train, wh_test = _split_folder(wh_folder, wh_split, seed)
    pd.DataFrame({'fname': wh_train}).to_csv(os.path.join(train_folder, 'wh.csv'), index=False)
    pd.DataFrame({'fname': wh_test}).to_csv(os.path.join(test_folder, 'wh.csv'), index=False)
