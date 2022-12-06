import pickle

def pklW(name, data):
    with open(name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def pklR(name):
    with open(name, 'rb') as handle:
        data = pickle.load(handle)
    return data

import csv

def csvW(name, data):
    with open(name, 'w', newline='') as handle:
        wr = csv.writer(handle, quoting=csv.QUOTE_ALL)
        wr.writerow(data)
    return True

def csvR(name, num=False):
    with open(name, 'r', newline='') as handle:
        rr = csv.reader(handle)
        data = list(rr)
    return data

import glob
import numpy as np

def gdf(path):
    """ Loads spike times of each spike detector.

    Parameters
    -----------
    path
        Path where the files with the spike times are stored.

    Returns
    -------
    data
        Dictionary containing spike times in the interval from 'begin'
        to 'end'.

    """
    data_raw = np.loadtxt(path)
    if data_raw.size > 0:
        if data_raw.ndim > 1:
            ts, es = data_raw[:,1], data_raw[:,0].astype(np.int)
        else:
            ts, es = [data_raw[1]], [np.int(data_raw[0])]
    else:
        ts, es = [], []

    return list(ts), list(es)

def gdfR(recpath):
    """Load gdf recordings from recpath

    Args:
        recpath (str): directory of recordings

    Returns:
        array: ts, es
    """
    paths = glob.glob(recpath + '*.gdf')
    ts, es = [], []
    for path in paths:
        t_ts, t_es = gdf(path)
        ts += t_ts
        es += t_es
    ts = np.array(ts)
    es = np.array(es)

    return ts, es
