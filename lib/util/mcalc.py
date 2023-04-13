import numpy as np
from scipy.stats import norm, gamma

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

def distance(sxy, txy, wh=1.0):
    dxy = np.abs(sxy[:, np.newaxis, :] - txy[np.newaxis, :, :])
    dxy[dxy > wh/2] = wh - dxy[dxy > wh/2]

    return np.sum(dxy, axis=-1)

def gen_con(rs, psrc, ptar, param_deg, param_wt, landscape=None):
    # calculate distance
    M, N, ncon = psrc.shape[0], ptar.shape[0], param_deg['ncon']

    # degree
    deg = np.zeros((M,N))
    if param_deg['type'] == 'gaussian':
        sigma, ncon = param_deg['sigma'], ncon
        phi = rs.uniform(low=-np.pi, high=np.pi, size=(M, ncon))
        rad = rs.normal(loc=0., scale=sigma, size=(M, ncon))*np.random.choice([-1,1], size=phi.shape)
    elif param_deg['type'] == 'gamma':
        k, theta, ncon = param_deg['k'], param_deg['theta'], ncon
        phi = rs.uniform(low=-np.pi, high=np.pi, size=(M, ncon))
        rad = rs.gamma(shape=k, scale=theta, size=(M, ncon))*np.random.choice([-1,1], size=phi.shape)
    else:
        assert False

    # position of target neurons
    xx, yy = rad*np.cos(phi) + np.reshape(psrc[:,0], (M,1)), rad*np.sin(phi) + np.reshape(psrc[:,1], (M,1))

    # landscape
    if landscape is not None:
        xx += np.reshape(landscape[:,0], (M,1))
        yy += np.reshape(landscape[:,1], (M,1))

    # selected points
    pts = np.concatenate([np.reshape(xx, (M, ncon, 1)),np.reshape(yy, (M, ncon,1))], axis=-1)

    # transform positions to index
    dist = np.zeros(deg.shape)
    for i in np.arange(ncon):
        dist_tar = distance(pts[:, i], ptar)
        idx = np.argmin(dist_tar, axis=1)
        deg[np.arange(M),idx] += 1
        dist[np.arange(M),idx] = dist_tar[np.arange(M), idx]
    # print(pts.shape, idx.shape, np.sum(deg), np.sum(deg, axis=1).mean(), param_deg)

    # weight
    if param_wt['type'] == 'gaussian':
        sigma = param_wt['sigma']
        wt = norm.pdf(dist, scale=sigma)/norm.pdf(0., scale=sigma)
    elif param_wt['type'] == 'gamma':
        k, theta, ncon = param_wt['k'], param_deg['theta'], param_deg['ncon']
        wt = gamma.pdf(dist, k, scale=theta)/gamma.pdf((k-1)*theta, k, scale=theta)
    elif param_wt['type'] == 'uniform':
        wt = rs.uniform(0., 1.0, dist.shape)
    else:
        assert False

    con = np.multiply(deg, wt)  # weight x degree
    if M==N:
        np.fill_diagonal(con, 0.)   # no self connection

    # delay
    if param_wt['type'] == 'gaussian':
        sigma = param_wt['sigma']
        delay = norm.pdf(dist, scale=sigma)/norm.pdf(0., scale=sigma)
    elif param_wt['type'] == 'gamma':
        k, theta, ncon = param_wt['k'], param_deg['theta'], param_deg['ncon']
        delay = gamma.pdf(dist, k, scale=theta)/gamma.pdf((k-1)*theta, k, scale=theta)
    elif param_wt['type'] == 'uniform':
        delay = rs.uniform(0., 1.0, dist.shape)
    else:
        assert False
    delay = (1-delay)*3 + 2

    return con, delay
