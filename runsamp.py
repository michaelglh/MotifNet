""" A cortical neuron motif simulator
"""

# system
import time
from datetime import datetime
import os
from tqdm import tqdm

# tools
import argparse
import json
import pandas as pd

# computation libs
import numpy as np

# plot
from lib.util.mviz import visSpk, visCurve, multiPlot
import seaborn as sns
import matplotlib.pyplot as plt

# simulation libs
from lib import params
import nest

def main():
    """Interneuron motif PC-PV-SOM-VIP
    """
    # create argument parser
    parser = argparse.ArgumentParser(description='Interneuron motif PC-PV-SOM-VIP.')
    # profiling
    parser.add_argument('--dpath', type=str, help='data path', default='./data/sampling')
    parser.add_argument('--fpath', type=str, help='data path', default='./plot/sampling')
    parser.add_argument('--epoch', type=int, help='number of epochs', default=100)
    parser.add_argument('--vtype', type=str, help='variance type', default='variance')
    # simulation setting
    parser.add_argument('--T', type=int, help='simulation time', default=500)
    parser.add_argument('--nrow', type=int, help='row size', default=80)
    parser.add_argument('--ncol', type=int, help='col size', default=60)
    # input setting
    parser.add_argument('--modE', type=float, help='modulation E', default=0)
    parser.add_argument('--modP', type=float, help='modulation P', default=0)
    parser.add_argument('--modS', type=float, help='modulation S', default=0)
    parser.add_argument('--modV', type=float, help='modulation V', default=0)
    parser.add_argument('--alpha', type=float, help='ratio EP', default=1.0)
    parser.add_argument('--beta', type=float, help='ratio ES', default=1.0)
    parser.add_argument('--gamma', type=float, help='ratio PS', default=1.0)
    parser.add_argument('--a', type=float, help='ratio EP', default=0.3)
    parser.add_argument('--b', type=float, help='ratio ES', default=-0.3)
    parser.add_argument('--inpE', type=float, help='input E', default=16.0)
    parser.add_argument('--inpP', type=float, help='input P', default=72.0)
    parser.add_argument('--inpS', type=float, help='input S', default=10.7)
    parser.add_argument('--inpV', type=float, help='input V', default=13.5)
    # connection setting
    parser.add_argument('--inhibit', type=float, help='mutual inhibition', default=1.0)
    parser.add_argument('--recurrence', type=float, help='recurret inhibition', default=0.0)
    parser.add_argument('--conJ', type=float, help='connection scale', default=5e1)
    # visualize
    parser.add_argument('--viz', action='store_true', default=False)

    # parsing
    print('Parsing arguments ... ... ')
    # parse argument
    args = parser.parse_args()
    simtime = args.T
    epoch = args.epoch
    # paths
    suffixpath = '/%s/J%.1f-m%.1f-r%.1f-a%.1f-b%.1f-c%.1f/'%(args.vtype, args.conJ, args.inhibit, args.recurrence, args.alpha, args.beta, args.gamma)  
    modpath = 'E%.1fP%.1fS%.1fV%.1f'%(args.modE, args.modP, args.modS, args.modV)              # path for simulation with different modulatory inputs
    recpath = args.dpath + suffixpath + modpath + '/'                                   # path for saving recordings (gdf files)
    figpath = args.fpath + suffixpath + '/'                                                   # path for saving figures
    spkpath = figpath + 'spk/'
    ratpath = figpath + 'rat/'
    wtspath = figpath + 'wt/'
    for path in [recpath, spkpath, ratpath, wtspath]:
        os.makedirs(path, exist_ok=True)

    #* Motif network
    # neurons
    nrow, ncol = args.nrow, args.ncol
    nneuron = nrow*ncol
    nrnTypes = ['E', 'P', 'S', 'V']
    # nrnParams = params.paramSin
    nrnParams = params.paramHom
    # E-I ratio 75:25; Layer II/III P-S-V ratio 40:40:30 (Bernardo Rudy, Jens, 2011)
    popRatio = np.array([0.75, 0.25*0.4, 0.25*0.3, 0.25*0.3])                                           # ratio of populations
    popSize = np.array(popRatio*nneuron, dtype=int)                                                     # population sizes computed from ratio
    # connections
    conProb = params.conProb                                                                            # connection probability between different populations
    
    # Hertag
    dendToSoma = 2.0
    conStr = [  [0.42/dendToSoma, -0.7, -1.96/dendToSoma, 0.],
                [1., -1.5, -1.3, 0.],
                [1., 0., -args.recurrence, -args.inhibit],
                [1., 0., -args.inhibit, -args.recurrence]]
    conStr = np.divide(conStr, np.outer(np.ones(len(nrnTypes)), popSize)*np.array(conProb))
    conStr[:,0] *= 2.0
    print(conStr*args.conJ)

    # Pfeffer

    # clipping
    clips = [   [0.5, 2.0, 2.0, 2.0],
                [0.5, 2.0, 2.0, 2.0],
                [0.5, 0., 2.0, 2.0],
                [0.5, 0., 2.0, 2.0]]

    # scale for PSP
    synvol = np.load('./lib/synvol.npz')
    factors = synvol['factors']
    factors = np.concatenate([factors, np.array(factors[:, 1]).reshape(len(nrnTypes), 1), np.array(factors[:, 1]).reshape(len(nrnTypes), 1)], axis=1)
    synStr = np.multiply(conStr, factors)*args.conJ
    synMax = np.multiply(clips, factors)

    extIn = np.array([args.inpE, args.inpP, args.inpS, args.inpV])              # external stimuli level
    modulationFreqs = np.array([args.modE, args.modP, args.modS, args.modV])    # modulation input frequency
    print('Modulation sizes', modulationFreqs)

    alpha, beta, gamma, scale = args.alpha, args.beta, args.gamma, 5e-2

    if args.vtype=='variance':
        # across-trial variance ratio
        corlevel = gamma
        sigma_e = 3.0/(1 + abs(1/alpha) + abs(1/beta))
        sigma_p = sigma_e / abs(alpha)
        sigma_s = sigma_e / abs(beta)
        cov_var = np.array([[sigma_e, corlevel, corlevel], [corlevel, sigma_p, corlevel], [corlevel, corlevel, sigma_s]])*scale
        samples = np.random.multivariate_normal(np.zeros(3), cov_var, epoch)
        if (alpha < 0) & (beta > 0):
            samples[:, 1] *= -1
        elif (alpha > 0) & (beta < 0):
            samples[:, 2] *= -1
        elif (alpha < 0) & (beta < 0):
            samples[:, 0] *= -1
    elif args.vtype=='covariance':
        # across-trial covariance
        a, b = args.a, args.b
        sigma_e = 3.0/(1 + abs(1/a) + abs(1/b))
        sigma_p = sigma_e / abs(a)
        sigma_s = sigma_e / abs(b)
        cov_var = np.array([[sigma_e, beta, alpha], [beta, sigma_p, gamma], [alpha, gamma, sigma_s]])*scale
        samples = np.random.multivariate_normal(np.zeros(3), cov_var, epoch)
        if (a < 0) & (b > 0):
            samples[:, 1] *= -1
        elif (a > 0) & (b < 0):
            samples[:, 2] *= -1
        elif (a < 0) & (b < 0):
            samples[:, 0] *= -1

    assert np.all(np.linalg.eigvals(cov_var) > 0)

    # sampling
    modscale = np.array([4, 60, 4])
    samples = np.concatenate([samples, np.zeros((1,3))])
    samples *= modscale
    modulationFreqs = np.array([args.modE + samples[:,0], args.modP + samples[:,1], args.modS + samples[:,2], [args.modV]*(epoch+1)]).T

    warmup = 250.
    preptime = 250.
    totalT = warmup + preptime + simtime

    with open(recpath + 'set.json', 'w') as saveFile:                                                   # save settings
        json.dump([nrnTypes, popSize.tolist(), [warmup ,preptime, simtime], modulationFreqs.tolist()], saveFile)

    # initialization of NEST
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.SetKernelStatus({'overwrite_files':True, 'local_num_threads':1})
    countVritualProcess = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    msd = int(datetime.now().strftime("%m%d%H%M%S"))
    nest.SetKernelStatus({'grng_seed': msd})
    nest.SetKernelStatus({'rng_seeds': range(msd+1, msd+countVritualProcess+1)})

    # neuron
    print('create neurons')
    neuronPopulations = {}
    for ntype, ncnt, nparam in zip(nrnTypes, popSize, nrnParams):
        neuronPopulations[ntype] = nest.Create('aeif_cond_alpha', ncnt)
        nest.SetStatus(neuronPopulations[ntype], nparam)
    # select random targets to measure membrane and current fluctuations
    popPos = np.concatenate([[0], np.cumsum(popSize)])
    multiTargets = [int(np.random.choice(np.arange(a+1, b+1), 1, replace=False)[0]) for a, b in zip(popPos[:-1], popPos[1:])]
    nest.SetStatus(multiTargets, {'V_peak':1e3, 'V_th':1e2})

    # connections
    print('generating connections')
    con_frames = []
    conMat = np.zeros((nneuron, nneuron))
    for typePre, populationPre in neuronPopulations.items():
        idxPre = nrnTypes.index(typePre)
        for typePost, populationPost in neuronPopulations.items():
            idxPost = nrnTypes.index(typePost)
            cprob, cstr = conProb[idxPost][idxPre], synStr[idxPost][idxPre]
            # generate connections
            conPrePost = np.where(np.random.rand(popSize[idxPre], popSize[idxPost]) < cprob, cstr, 0.)
            conPrePost = np.multiply(conPrePost, np.random.lognormal(mean=0., sigma=1., size=conPrePost.shape))
            if typePre is typePost:
                np.fill_diagonal(conPrePost, 0.)

            # clipping
            conPrePost[np.abs(conPrePost) > synMax[idxPost, idxPre]] = np.sign(np.min(conPrePost))*synMax[idxPost, idxPre]

            # connect
            nest.Connect(populationPre, populationPost, syn_spec={'weight': conPrePost.T, 'delay': 2.0})

            # statistics
            if np.sum(conPrePost != 0) > 0:
                print(typePre, typePost, cprob, cstr, np.mean(conPrePost[conPrePost != 0]), conPrePost.shape)
                # save connection
                con_valid = np.abs(conPrePost[conPrePost!=0].ravel())
                con_frames.append(pd.DataFrame(list(zip(np.log10(con_valid), ['%s_%s'%(typePost, typePre)]*np.sum(conPrePost != 0))), columns=[r'$log_{10} |J|$', 'post_pre']))
                conMat[popPos[idxPre]:popPos[idxPre+1], popPos[idxPost]:popPos[idxPost+1]] = conPrePost
            else:
                print(typePre, typePost, cprob, cstr, conPrePost.shape)

    con_frame = pd.concat(con_frames)
    gax = sns.displot(data=con_frame, x=r'$log_{10} |J|$', hue='post_pre', kind="kde")
    gax.set(xlim=(-2, 1))
    plt.savefig(wtspath + modpath + '_J.png', dpi=300)
    plt.close()

    # record device
    print('link devices')
    spikeDetector = nest.Create('spike_detector', 1, {'to_file':True, 'label':recpath + 'spk'})
    for npop in neuronPopulations.values():
        nest.Connect(npop, spikeDetector)
    multiDetector = nest.Create('multimeter', len(neuronPopulations), {'withtime': True, 'interval': 0.1, 'record_from': ['V_m', 'g_ex', 'g_in']})
    nest.Connect(multiDetector, tuple(multiTargets), 'one_to_one')

    # input device
    inputModulation = []
    for ext, ntype in zip(extIn, nrnTypes):
        inputModulation.append(nest.Create('poisson_generator', 20, {'rate':ext, 'start':0.}))
        nest.Connect(inputModulation[-1], neuronPopulations[ntype], syn_spec={'weight':5e1})

    for e in tqdm(range(epoch)):
        print('init')
        for npop in neuronPopulations.values():
            rvs = np.random.normal(-70., 5., len(npop))
            nest.SetStatus(npop, params='V_m', val=rvs.tolist())

        # simulate
        print('warmup')
        for ext, modulation, moddevice in zip(extIn, modulationFreqs[-1], inputModulation):
            nest.SetStatus(moddevice, {'rate':ext+float(modulation)})
        nest.Simulate(warmup)

        print('stimulate')
        for ext, modulation, moddevice in zip(extIn, modulationFreqs[e], inputModulation):
            nest.SetStatus(moddevice, {'rate':np.maximum(ext+float(modulation), 0.)})
        nest.Simulate(simtime+preptime)

        # visualize
        print('visualize')
        # spikes
        spikeEvents = nest.GetStatus(spikeDetector, 'events')[0]
        spikeTimes, spikeIds = spikeEvents['times'], spikeEvents['senders']
        idxVis = (spikeTimes > totalT*e) & (spikeTimes < totalT*(e+1)) 
        visSpk(spikeTimes[idxVis], spikeIds[idxVis], path=spkpath + modpath + '_%d'%e)

        # firing rates
        idBins = np.concatenate([[0], np.cumsum(popSize)]) + 1
        tmBins = np.arange(totalT*e, totalT*(e+1)+1, 100)
        rateTimePop = np.histogram2d(spikeTimes, spikeIds, bins=[tmBins, idBins])[0]*10/popSize
        visCurve([tmBins[1:]]*len(nrnTypes), rateTimePop.T,
                'time (ms)', 'rate (Hz)', 'firing rate', ratpath + 'fr_' + modpath + '_%d'%e, nrnTypes)

        # # multimeter
        # multiEvents = [nest.GetStatus([detector])[0]['events'] for detector in multiDetector]
        # multiPlot(multiEvents, [totalT*e, totalT*(e+1)], nrnTypes, ratpath + 'vg_' + modpath + '_%d'%e)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
