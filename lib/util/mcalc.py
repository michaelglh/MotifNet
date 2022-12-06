import numpy as np

def spike2bin(signal, template, kernel, dt):
    """Binning spikes from spiking times.

    Parameters
    ----------
    signal : ndarray
        Spiking times.
    template : ndarray
        Binning template.
    kernel : ndarray
        Spike convolution kernel.

    Returns
    -------
    ndarray
        Binned signal from spiking times.

    """
    if signal is []:
        return template
    else:
        for s in signal:
            template[int(s/dt)-1] = 1
        return np.convolve(template, kernel, 'same')

def spikeCoin(spikeMat, type=True):
    """Cout coincidence of several spike trains.

    Parameters
    ----------
    spikeMat : numpy matrix
        A matrix of binned spiking actitivities.
    type : bool
        Coincidence normalization type.

    Returns
    -------
    type
        Description of returned object.

    """
    # Get number of trials
    N, _ = spikeMat.shape

    # Calculate coincidence
    coV_ori = np.dot(spikeMat, np.transpose(spikeMat))
    coV = coV_ori.copy()

    # Normalize
    if not type:
        # Absolute coincidence count
        pass
    else:
        # Normalize to geometric mean energy
        for i in range(N):
            for j in range(N):
                energy = coV_ori[i, i]*coV_ori[j, j]
                if energy:
                    coV[i, j] /= np.sqrt(energy)
                else:
                    coV[i, j] = 0

    # Correlation
    cor = (np.sum(coV) - np.trace(coV))/N/(N-1)

    return coV, cor