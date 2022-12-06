""" A cortical neuron motif simulator
"""

# system
import time
from datetime import datetime
from tqdm import tqdm

# tools
import matplotlib.pyplot as plt

# computation libs
import numpy as np


# simulation libs
from lib import params
import nest

def main():
    nrnTypes = ['E', 'P', 'S', 'V']
    numtype = len(nrnTypes)
    nrnParams = params.paramHom
    popSize = np.ones(numtype, dtype=int)

    for nparam in nrnParams:
        nparam['E_L'] = -55.0

    # initialization of NEST
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.SetKernelStatus({'overwrite_files':True, 'local_num_threads':1})
    # random seed
    countVritualProcess = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    msd = int(datetime.now().strftime("%m%d%H%M%S"))
    nest.SetKernelStatus({'grng_seed': msd})
    nest.SetKernelStatus({'rng_seeds': range(msd+1, msd+countVritualProcess+1)})

    # neuron
    print('create neurons')
    neuronPopulations = {}
    popAll = ()
    for ntype, ncnt, nparam in zip(nrnTypes, popSize, nrnParams):
        neuronPopulations[ntype] = nest.Create('aeif_cond_alpha', ncnt)
        nest.SetStatus(neuronPopulations[ntype], nparam)
        popAll += neuronPopulations[ntype]

    # connections
    print('generating connections')
    wscales = np.linspace(0., 1.0, 101)
    ntrial = len(wscales)
    simtime = 50.
    sg = nest.Create('spike_generator',numtype)
    nest.Connect(sg, popAll, 'one_to_one')

    # record device
    print('link devices')
    multiDetector = nest.Create('multimeter', numtype, {'withtime': True, 'interval': 0.1, 'record_from': ['V_m']})
    nest.Connect(multiDetector, popAll, 'one_to_one')

    # simulations
    print('init states')
    offset = 10.0
    for tr in tqdm(range(ntrial)):
        for npop, nparam in zip(neuronPopulations.values(), nrnParams):
            nest.SetStatus(npop, params='V_m', val=nparam['E_L'])

        conn = nest.GetConnections(sg, popAll)
        nest.SetStatus(conn, {'weight': wscales[tr]})

        nest.SetStatus(sg,{'spike_times':np.array([tr*simtime + offset])})
        nest.Simulate(simtime)

    offset = 10.0 + ntrial*simtime
    for tr in tqdm(range(ntrial)):
        for npop, nparam in zip(neuronPopulations.values(), nrnParams):
            nest.SetStatus(npop, params='V_m', val=nparam['E_L'])

        conn = nest.GetConnections(sg, popAll)
        nest.SetStatus(conn, {'weight': -wscales[tr]})

        nest.SetStatus(sg,{'spike_times':np.array([tr*simtime + offset])})
        nest.Simulate(simtime)

    multiEvents = [nest.GetStatus([detector])[0]['events'] for detector in multiDetector]
    tsArr = [events['times'] for events in multiEvents]
    vmArr = [events['V_m'] for events in multiEvents]
    synvol = np.zeros((numtype, 2, ntrial))
    offset = 10.0
    for ntp, nparam in enumerate(nrnParams):
        ts, vm = tsArr[ntp], vmArr[ntp]
        for tr in range(ntrial):
            synvol[ntp, 0, tr]  = np.max(vm[(ts>tr*simtime + offset) & ((ts<(tr+1)*simtime + offset))]) - nparam['E_L']

    offset = 10.0 + ntrial*simtime
    for ntp, nparam in enumerate(nrnParams):
        ts, vm = tsArr[ntp], vmArr[ntp]
        for tr in range(ntrial):
            synvol[ntp, 1, tr]  = np.min(vm[(ts>tr*simtime + offset) & ((ts<(tr+1)*simtime + offset))]) - nparam['E_L'] 

    plt.subplots(1,numtype, figsize=(8.0, 2.5))
    for fidx in range(numtype):
        plt.subplot(1,numtype, fidx+1)
        plt.plot(wscales, synvol[fidx,0], c='b', label='EPSP')
        plt.plot(wscales, synvol[fidx,1], c='r', label='IPSP')
        if fidx == 0:
            plt.xlabel(r'$g_{syn}$', fontsize='large', labelpad=-10)
            plt.ylabel('PSP (mV)', fontsize='large')
            plt.yticks([-1.0, -0.5, 0., 0.2])
            plt.xticks([0., 1.0])
        else:
            plt.xticks([])
            plt.yticks([])
        if fidx == 1:
            plt.legend()
        plt.ylim([-1, 0.2])
        plt.gca().set_title(nrnTypes[fidx])
    plt.tight_layout()
    plt.savefig('./plot/synvol.pdf', dpi=1000)

    plt.subplots(numtype,1, figsize=(8.0, 10.0))
    for fidx in range(numtype):
        plt.subplot(numtype,1,fidx+1)
        plt.plot(tsArr[fidx][tsArr[fidx]<simtime*ntrial], vmArr[fidx][tsArr[fidx]<simtime*ntrial], c='b', label='EPSP')
        plt.plot(tsArr[fidx][tsArr[fidx]>simtime*ntrial]-simtime*ntrial, vmArr[fidx][tsArr[fidx]>simtime*ntrial], c='r', label='IPSP')
        plt.xticks([0, simtime*ntrial/4, simtime*ntrial*3/4, simtime*ntrial], ['0', '0.25', '0.75', '1.0'])
        plt.xlabel(r'$g_{syn}$', labelpad=-10)
        plt.ylabel('Vm (mV)')
        plt.ylim([-56.0, -54.8])
        plt.yticks([-56.0, -55.5, -55., -54.8])
        plt.legend()
    plt.tight_layout()
    plt.savefig('./plot/synvm.pdf', dpi=1000)

    factors = np.zeros((numtype, 2))
    for idx in range(numtype):
        for ii in range(2):
            factors[idx, ii] = 1/np.abs((synvol[idx,ii,0]-synvol[idx,ii,-1])/(wscales[0]-wscales[-1]))
    
    np.savez('./lib/synvol.npz', ws=wscales, synvol=synvol, factors=factors)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))