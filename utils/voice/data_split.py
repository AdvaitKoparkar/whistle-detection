import sys
sys.path.append('../../')

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.voice.wav_utils import find_files, clear_folder

def _split_folder(base_folder, splits, seed):
    files = find_files(base_folder)
    train, test = train_test_split(files, random_state=seed, test_size=splits['test'])
    return train, test

def split_data(base_folders, seed=None):
    train_folder = r'..\data\csv\train'
    test_folder = r'..\data\csv\test'

    clear_folder(train_folder, '*.csv')
    clear_folder(test_folder, '*.csv')

    fg_train_csv_fpath = os.path.join(train_folder, 'fg.csv')
    fg_test_csv_fpath = os.path.join(test_folder, 'fg.csv')
    for fg_folder in base_folders['fg_folders']:
        fg_split = {'test': 0.3}
        fg_train, fg_test = _split_folder(fg_folder, fg_split, seed)
        if not os.path.isfile(fg_train_csv_fpath):
            pd.DataFrame({'fname': fg_train}).to_csv(fg_train_csv_fpath, index=False)
        else:
            df0 = pd.read_csv(fg_train_csv_fpath)
            df = pd.concat([df0, pd.DataFrame({'fname': fg_train})])
            df.to_csv(fg_train_csv_fpath, index=False)

        if not os.path.isfile(fg_test_csv_fpath):
            pd.DataFrame({'fname': fg_test}).to_csv(fg_test_csv_fpath, index=False)
        else:
            df0 = pd.read_csv(fg_test_csv_fpath)
            df = pd.concat([df0, pd.DataFrame({'fname': fg_test})])
            df.to_csv(fg_test_csv_fpath, index=False)

    bg_train_csv_fpath = os.path.join(train_folder, 'bg.csv')
    bg_test_csv_fpath = os.path.join(test_folder, 'bg.csv')
    for bg_folder in base_folders['bg_folders']:
        bg_split = {'test': 0.3}
        bg_train, bg_test = _split_folder(bg_folder, bg_split, seed)
        if not os.path.isfile(bg_train_csv_fpath):
            pd.DataFrame({'fname': bg_train}).to_csv(bg_train_csv_fpath, index=False)
        else:
            df0 = pd.read_csv(bg_train_csv_fpath)
            df = pd.concat([df0, pd.DataFrame({'fname': bg_train})])
            df.to_csv(bg_train_csv_fpath, index=False)

        if not os.path.isfile(bg_test_csv_fpath):
            pd.DataFrame({'fname': bg_test}).to_csv(bg_test_csv_fpath, index=False)
        else:
            df0 = pd.read_csv(bg_test_csv_fpath)
            df = pd.concat([df0, pd.DataFrame({'fname': bg_test})])
            df.to_csv(bg_test_csv_fpath, index=False)

    wh_train_csv_fpath = os.path.join(train_folder, 'wh.csv')
    wh_test_csv_fpath = os.path.join(test_folder, 'wh.csv')
    for wh_folder in base_folders['whistle_folders']:
        wh_split = {'test': 0.3}
        wh_train, wh_test = _split_folder(wh_folder, wh_split, seed)
        if not os.path.isfile(wh_train_csv_fpath):
            pd.DataFrame({'fname': wh_train}).to_csv(wh_train_csv_fpath, index=False)
        else:
            df0 = pd.read_csv(wh_train_csv_fpath)
            df = pd.concat([df0, pd.DataFrame({'fname': wh_train})])
            df.to_csv(wh_train_csv_fpath, index=False)

        if not os.path.isfile(wh_test_csv_fpath):
            pd.DataFrame({'fname': wh_test}).to_csv(wh_test_csv_fpath, index=False)
        else:
            df0 = pd.read_csv(wh_test_csv_fpath)
            df = pd.concat([df0, pd.DataFrame({'fname': wh_test})])
            df.to_csv(wh_test_csv_fpath, index=False)
