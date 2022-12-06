# system
import time
import os

# tools
import argparse
import json
from tqdm import tqdm

# computation libs
import numpy as np

# plot
import matplotlib.pyplot as plt

# help
from lib.util.mio import gdfR
from lib.util.mviz import animF
from lib.util.mcalc import spikeCoin

def main():
    # create argument parser
    parser = argparse.ArgumentParser(description='Analysis of motif PV-SOM-VIP.')
    # profiling
    parser.add_argument('--dpath', type=str, help='data path', default='./data')
    parser.add_argument('--fpath', type=str, help='figure path', default='./plot')
    parser.add_argument('--rate', action='store_true', default=False)
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--fano', action='store_true', default=False)
    parser.add_argument('--dim', action='store_true', default=False)

    """
    Parsing
    """
    print('Parsing arguments ... ... ')
    # parse argument
    args = parser.parse_args()
    # I/O
    datapath = args.dpath + '/'
    figpath = args.fpath + '/dist/'
    os.makedirs(figpath, exist_ok=True)

    print('Loading data ... ... ')
    # recording list
    eset, pset, sset = [-4, 4.1, 1.0], [-60, 60.1, 15], [-4, 4.1, 1]
    erange = list(np.arange(*eset))
    prange = list(np.arange(*pset))
    srange = list(np.arange(*sset))
    vrange = [0]
    fr_list = []
    for ev in erange:
        for pv in prange:
            for sv in srange:
                for vv in vrange:
                    fr_val = [ev, pv, sv, vv]
                    fr_list.append(fr_val)
    with open('./data/fr_list.json', 'w') as saveFile:
        json.dump(fr_list, saveFile)

    # load record setting
    fr_val = fr_list[0]
    with open(datapath + 'E%.1fP%.1fS%.1fV%.1f/set.json'%(fr_val[0], fr_val[1], fr_val[2], fr_val[3]), 'r') as net_fp:
        motif_set = json.loads(net_fp.read())
    nrn_tps, nrn_cnt, simts = motif_set[0], motif_set[1], motif_set[2]
    warmup, simtime = simts
    print('neuron types, population size:')
    print(list(zip(nrn_tps, nrn_cnt)))

    print('Analysing data ... ... ')
    if args.rate:
        print('extract firing rate')
        id_bins = np.concatenate([[0], np.cumsum(nrn_cnt)]) + 1
        ts_bins = np.array([0, warmup, warmup + simtime])
        fr = np.zeros((len(fr_list), len(nrn_tps)))
        for i in tqdm(range(len(fr_list))):
            fr_val = fr_list[i]
            rec = gdfR(datapath + 'E%.1fP%.1fS%.1fV%.1f/'%(fr_val[0], fr_val[1], fr_val[2], fr_val[3]))
            fr_2d = np.histogram2d(rec[0], rec[1], bins=[ts_bins, id_bins])[0]
            fr[i] = fr_2d[-1]*1e3/simtime/nrn_cnt
        with open('./data/fr.json', 'w') as saveFile:
            json.dump(fr.tolist(), saveFile)
            
    if args.dist:
        print('extract distribution')
        idx_pos = np.concatenate([[0], np.cumsum(nrn_cnt)])
        id_bins = np.arange(np.sum(nrn_cnt)+1) + 1
        ts_bins = np.array([0, warmup, warmup + simtime])
        rates = np.zeros((len(fr_list), np.sum(nrn_cnt)))
        for i in tqdm(range(len(fr_list))):
            fr_val = fr_list[i]
            rec = gdfR(datapath + 'E%.1fP%.1fS%.1fV%.1f/'%(fr_val[0], fr_val[1], fr_val[2], fr_val[3]))
            fr_2d = np.histogram2d(rec[0], rec[1], bins=[ts_bins, id_bins])[0]
            rates[i] = fr_2d[-1]*1e3/simtime
        with open('./data/fr_dist.json', 'w') as saveFile:
            json.dump(rates.tolist(), saveFile)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
