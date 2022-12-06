"""Matrix decompositions
"""

# computation libs
import numpy as np
from scipy.linalg import schur, eig

# plot
import matplotlib.pyplot as plt

def plotVectors(axis, vecs, cols, alpha=1):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    for i in range(len(vecs)):
        x = np.concatenate([[0, 0], vecs[i]])
        axis.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                   alpha=alpha)

def schurDecomposition(connectionStength, labels, number=False, figname=None):
    """Schur decomposition of connection matrix

    Args:
        connectionStength (array): connection matrix
        labels (array): axis labels
        number (bool): label pixels with value or not
        figname (string, optional): save file name. Defaults to None.

    Returns:
        array: U*T*U^t
    """

    # schur decomposition
    schurDecT, schurDecU = schur(connectionStength, output='real')
    eigenDecW, eigenDecV = eig(connectionStength)

    schurDecT = np.round(schurDecT*100)/100
    schurDecU = np.round(schurDecU*100)/100
    connectionStength = np.round(connectionStength*100)/100
    patterns = ['P%d'%i for i in range(len(labels))]

    plt.subplots(2, 2)
    # connection
    axis = plt.subplot(221)
    axis.imshow(connectionStength)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if number:
        for i in range(len(labels)):
            for j in range(len(labels)):
                axis.text(j, i, connectionStength[i][j], ha="center", va="center", color="w")
    plt.title('C')

    # T
    axis = plt.subplot(222)
    axis.imshow(schurDecT)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(patterns)
    axis.set_yticklabels(patterns)
    if number:
        for i in range(len(labels)):
            for j in range(len(labels)):
                axis.text(j, i, schurDecT[i][j], ha="center", va="center", color="w")
    plt.title('T')

    # U
    axis = plt.subplot(223)
    axis.imshow(schurDecU)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(patterns)
    axis.set_yticklabels(labels)
    if number:
        for i in range(len(labels)):
            for j in range(len(labels)):
                axis.text(j, i, schurDecU[i][j], ha="center", va="center", color="w")
    plt.title('U')

    # eigenvalues
    axis = plt.subplot(224)
    axis.axis('equal')
    axis.scatter(eigenDecW.real, eigenDecW.imag)
    # plotVectors(axis, schurDecU.T, ['red', 'red'])
    # plotVectors(axis, np.absolute(eigenDecV.T), ['blue', 'blue'])
    plt.title('Eigen')

    plt.subplots_adjust(hspace=0.5)
    if figname is not None:
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()

    return schurDecT, schurDecU
