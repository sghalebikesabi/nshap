import io
import numpy as np
import pandas as pd
import pathlib
import requests
from typing import Tuple
import zipfile


def int2coal(i, nfeats):
    # str_coal = '0'*(nfeats-len(bin(i))+2) + bin(i)[2:]
    str_coal = format(i, f'#0{nfeats+2}b')[2:]
    arr = np.array([int(j) for j in str_coal])
    return arr

def sample_coals(arr, refs=None):
    ncoals, nfeats = arr.shape
    int_coals = np.random.choice(2**nfeats, size=(ncoals))
    coals = np.stack([int2coal(c, nfeats) for c in int_coals])
    if refs is None:
        refs = arr[np.random.choice(ncoals, size=(ncoals))]
    imps = arr * coals + refs * (1-coals)
    return imps, refs

def is_jpnb():
    try:
        is_nb = len(get_ipython().__class__.__name__) > 0
    except:
        is_nb = False
    return is_nb

def download_uci_bikeshare(data_dir: pathlib.Path, keep_date=False) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads the UCI Bikeshare dataset and caches it as pickled
    pandas dataframes"""

    # Downlaod the dataset (with caching to disk)
    train_pkl = data_dir / 'train_df.pkl'
    test_pkl = data_dir / 'test_df.pkl'

    if data_dir.exists() and len(tuple(data_dir.iterdir())) > 1:
        train_df = pd.read_pickle(train_pkl)
        test_df = pd.read_pickle(test_pkl)
    else:
        # import and save the dataset to a gitignored directory
        data_dir.mkdir(exist_ok=True)
        with (data_dir / '.gitignore').open('w') as gitignore:
            gitignore.write('*.pkl\n')

        # download the dataset from UCI
        zip_url = 'https://archive.ics.uci.edu/ml/' \
                  'machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
        z = zipfile.ZipFile(io.BytesIO(requests.get(zip_url).content))
        dtypes = {
            'holiday': 'bool',
            'workingday': 'bool',
            'weathersit': 'category',
            'season': 'category'
        }
        with z.open('hour.csv') as csv:
            full_df = pd.read_csv(csv, dtype=dtypes)

        # split train/test by year
        is_2011 = full_df['yr'] == 0
        train_df = full_df[is_2011].reset_index(drop=True)
        test_df = full_df[~is_2011].reset_index(drop=True)

        # serialize datasets to disk
        train_df.to_pickle(train_pkl)
        test_df.to_pickle(test_pkl)

    features = [
        # the tree-based model will not be able to
        # capture the linear trend in the instant variable
        # across our train-test split
    #     'instant', 
        'season', 
        'mnth', 
        'hr', 
        'holiday', 
        'weekday', 
        'workingday', 
        'weathersit', 
        'temp', 
        # dropped due to high correlation to 'temp'
    #     'atemp', 
        'hum', 
        'windspeed'
    ]
    if keep_date:
        features += ['dteday']

    target = 'cnt'

    x_train = train_df[features]
    x_test = test_df[features]
    y_train = train_df[target]
    y_test = test_df[target]

    return x_train, x_test, y_train, y_test

def countSetBits(n):
    ''' This is a helper function to compute the number of bits in the binary representation of n. '''
    
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count

def exp_kernel(x, sigma):
    return np.exp(-x/sigma**2)

def dist_weights(x, instances, sigma, kernel, scaled=True):
    distances = ((instances - x)**2).sum(1)
    weights = kernel(distances, sigma)
    if scaled:
        weights /= np.sum(weights)
    return weights

def smooth(vals, instances, sigma, correct=False, model=None, kernel=exp_kernel, return_weights=False):
    smoothed_vals = np.zeros_like(vals)
    evals = model(instances)
    for idx, x in enumerate(instances):
        weights = dist_weights(x, instances, sigma, kernel, scaled=True)
        smoothed_vals[idx] = (weights.reshape((-1, 1))*vals).sum(0)
        if correct:
            smoothed_vals[idx] += evals[idx] - (weights*evals).sum(0)
    if return_weights:
        return smoothed_vals, weights
    return smoothed_vals