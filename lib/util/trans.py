import numpy as np

__all__ = [
    'gcf2ann',
    'mat2ann',
]

def gcf2ann(N, ts, evs):
    """Change gdf form (seperate lists of neuron id and corresponding spike events) recording to ann form (nested lists of each neuron's spike events).

    Parameters
    ----------
    N : int
        Number of neurons.
    ts : ndarray
        Spike times.
    evs : ndarray
        Spike index.

    Returns
    -------
    ndarray
        Ann form recording.

    """
    spks = [[] for _ in range(N)]
    for t, idx in zip(ts, evs):
        spks[idx-1].append(t)
    return spks

def mat2ann(conn, deg=True, offset=0):
    """Change matrix form connection (conneciton matrix) to ann form connection (nested lists).

    Parameters
    ----------
    conn : ndarray
        Connection matrix.
    offset : int
        Offset of index

    Returns
    -------
    ndarray
        Ann form connection lists.

    """
    nsrc, _ = conn.shape
    spks = [[] for _ in range(nsrc)]
    wts = []
    for i in range(nsrc):
        idx = np.nonzero(conn[i])[0]
        wts += [conn[i][idx]]
        if deg:
            idx = list(map(lambda el, cnt:[el]*cnt, idx, conn[i][idx]))
            idx = np.array([item for sublist in idx for item in sublist])
        spks[i] = idx
    return spks, wts
