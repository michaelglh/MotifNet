# manipulation
import numpy as np

# plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

__all__ = [
    'visSpk',
    'visCurve',
    'animF',
    'multiPlot',
]

# plot spikes
def visSpk(ts, es, path=None):
    """Visualization of spiking activities in gdf form

    Args:
        ts (array): time points
        es (array): spike events
        path (string, optional): save path of image. Defaults to None.
    """
    plt.figure()
    plt.scatter(ts, es, s=1, marker='.')
    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron')

    if path is not None:
        plt.savefig(path + '.png')
        plt.close()

def visCurve(xs, ys, x_label, y_label, title, path, labels=None, x_lim=None, y_lim=None):
    """Visualization of curves with detailed setting

    Args:
        xs (arrays): x of curves
        ys (arrays): y of curves
        x_label (string): x unit
        y_label (string): y unit
        title (string): figure title
        path (string): save path
        labels (list, optional): lable of curves. Defaults to None.
        x_lim (list, optional): x limits. Defaults to None.
        y_lim (list, optional): y limits. Defaults to None.
    """
    fig, ax = plt.subplots(1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.title.set_text(title)
    if labels is not None:
        for x, y, label in zip(xs, ys, labels):
            ax.plot(x, y, label=label)
        ax.legend()
    else:
        for x, y in zip(xs, ys):
            ax.plot(x, y)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    if path is None:
        plt.show()
    else:
        fig.savefig(path + '.png')
        plt.close()

def animF(data, path=None):
    """Create animation from data frames, each frame is a 2d matrix

    Args:
        data (array): array of 2d matrices
        path (string, optional): save path of video. Defaults to None.
    """
    # generate anim
    fig, ax = plt.subplots(1)
    ims = []
    for state in data:
        ims.append([ax.imshow((state))])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,)

    # save anim
    if path is not None:
        ani.save(path + ".mp4", writer='ffmpeg')
    plt.close()

def multiPlot(multiEvents, xlim, labels, path=None):
    """Visualization of multi measurements in NEST simulator

    Args:
        multiEvents (list): different measurements in gdf form
        xlim (list): x limits
        labels (list): labels of measurements
        path (string, optional): save path of figure. Defaults to None.

    Returns:
        list: events arrays
    """
    # extract data
    smoothKern = np.ones(50)/50
    tsArr = [events['times'] for events in multiEvents]
    vmArr = [events['V_m'] for events in multiEvents]
    gexArr = [np.convolve(events['g_ex'], smoothKern, 'same') for events in multiEvents]
    ginArr = [np.convolve(events['g_in'], smoothKern, 'same') for events in multiEvents]
    tsArr, vmArr, gexArr, ginArr = np.array(tsArr), np.array(vmArr), np.array(gexArr), np.array(ginArr)
    gratArr = np.divide(gexArr, ginArr, out=np.zeros_like(gexArr), where=ginArr!=0)

    # visualization
    tsLim = (tsArr > xlim[0]) & (tsArr < xlim[1])
    plt.subplots(2, 2, figsize=(20, 10))
    plt.subplot(221)
    [plt.plot(ts, vms, linewidth=1) for ts, vms in zip(tsArr, vmArr)]
    plt.xlabel('Time [ms]')
    plt.xlim(xlim)
    plt.ylabel('V_m [mV]')
    plt.ylim([np.min(vmArr[tsLim]), np.max(vmArr[tsLim])])
    plt.legend(labels=labels)

    plt.subplot(222)
    [plt.plot(ts, gexs, linewidth=1) for ts, gexs in zip(tsArr, gexArr)]
    plt.xlabel('Time [ms]')
    plt.xlim(xlim)
    plt.ylabel('g_ex [nS]')
    plt.ylim([np.min(gexArr[tsLim]), np.max(gexArr[tsLim])])
    plt.legend(labels=labels)

    plt.subplot(223)
    [plt.plot(ts, gins, linewidth=1) for ts, gins in zip(tsArr, ginArr)]
    plt.xlabel('Time [ms]')
    plt.xlim(xlim)
    plt.ylabel('g_in [nS]')
    plt.ylim([np.min(ginArr[tsLim]), np.max(ginArr[tsLim])])
    plt.legend(labels=labels)

    plt.subplot(224)
    [plt.plot(ts, grats, linewidth=1) for ts, grats in zip(tsArr, gratArr)]
    plt.xlabel('Time [ms]')
    plt.xlim(xlim)
    plt.ylabel('gex/gin [nS]')
    plt.ylim([0, 20])
    plt.legend(labels=labels)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '.png')
        plt.close()
        return [tsArr, vmArr, gexArr, ginArr]
