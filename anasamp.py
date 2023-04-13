# system
import time
import os

# tools
import argparse
import json

# computation libs
import numpy as np
from scipy.stats import skew

# plot
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# help
from lib.util.mio import gdfR

def main():
    # create argument parser
    parser = argparse.ArgumentParser(description='Analysis of motif PV-SOM-VIP.')
    # profiling
    parser.add_argument('--dpath', type=str, help='data path', default='./data/sampling')
    parser.add_argument('--exppath', type=str, help='plot path', default='./experiments/J25.0-m0.5-r0.0')
    parser.add_argument('--epoch', type=int, help='number of epoch', default=100)
    parser.add_argument('--vtype', type=str, help='variance type', default='variance')
    # input setting
    parser.add_argument('--modE', type=float, help='modulation E', default=0)
    parser.add_argument('--modP', type=float, help='modulation P', default=0)
    parser.add_argument('--modS', type=float, help='modulation S', default=0)
    parser.add_argument('--modV', type=float, help='modulation V', default=0)
    parser.add_argument('--a', type=float, help='ratio EP', default=0.3)
    parser.add_argument('--b', type=float, help='ratio ES', default=-0.3)
    # connection setting
    parser.add_argument('--inhibit', type=float, help='mutual inhibition', default=0.5)
    parser.add_argument('--recurrence', type=float, help='recurret inhibition', default=0.0)
    parser.add_argument('--conJ', type=float, help='connection scale', default=5e1)
    # visualize
    parser.add_argument('--viz', action='store_true', default=False)

    # parsing
    print('Parsing arguments ... ... ')
    # parse argument
    args = parser.parse_args()
    epoch = args.epoch
    modscale = np.array([4, 60, 4, 1])

    # variables
    if args.vtype == 'covariance':
        alphas, betas = [0.1, 0.4, 0.1, 0.4], [0.1, 0.1, 0.4, 0.4]
        gammas = [0.5]
    elif args.vtype == 'variance':
        alphas, betas = [0.3, 3.0, 0.3, -0.3], [0.3, 3.0, -0.3, 0.3]
        gammas = [0.5]
    
    # paths
    modpath = 'E%.1fP%.1fS%.1fV%.1f'%(args.modE, args.modP, args.modS, args.modV)
    figpath = args.exppath + '/fig/' + args.vtype
    savepath = args.exppath + '/frs'
    for path in [figpath, savepath]:
        os.makedirs(path, exist_ok=True)

    binsize = 20.
    for gamma in gammas:
        filename = savepath + '/%s_%s_gamma_%.1f.npy'%(args.vtype, modpath, gamma)
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                sps = np.load(f)
                frs = np.load(f)
                rts = np.load(f)
            print(sps.shape, frs.shape, rts.shape)
            numts = int(rts.shape[1]/epoch)
            T = numts * binsize
            warmup = 250.
            nrn_tps = ['E', 'P', 'S', 'V']
            os.makedirs(args.exppath + '/fig/%s/'%args.vtype, exist_ok=True)
        else:
            frs = []
            rts = []
            sps = []
            for alpha, beta in zip(alphas, betas):
                # path
                suffixpath = '/%s/J%.1f-m%.1f-r%.1f-a%.1f-b%.1f-c%.1f/'%(args.vtype, args.conJ, args.inhibit, args.recurrence, alpha, beta, gamma)    
                recpath = args.dpath + suffixpath + modpath + '/'                               # path of recordings (gdf files)

                print('Loading data %.1f %.1f %.1f ... ... '%(alpha, beta, gamma))
                with open(recpath + 'set.json', 'r') as net_fp:
                    motif_set = json.loads(net_fp.read())
                nrn_tps, nrn_cnt, simtimes, samps = motif_set[0], motif_set[1], motif_set[2], np.array(motif_set[3])
                warmup, preptime, simtime = simtimes
                print('neuron types, population size:')
                print(list(zip(nrn_tps, nrn_cnt)))
                print('external input: ', samps[-1])
                samps = (samps[:epoch]-samps[-1])/modscale
                sps.append(samps)
                rec = gdfR(recpath)

                print('Analysing data ... ... ')
                T = np.sum(simtimes)
                numts = int(T/binsize)
                id_bins = np.concatenate([[0], np.cumsum(nrn_cnt)]) + 1
                ts_bins = np.concatenate([[T*idx, T*idx + warmup + preptime] for idx in range(epoch)] + [[T*epoch]])
                frs.append(np.histogram2d(rec[0], rec[1], bins=[ts_bins, id_bins])[0]*1e3/simtime/nrn_cnt)
                fr = frs[-1][1::2, 0]

                ts_bins = np.arange(0., T*epoch+1, binsize)
                rts.append(np.histogram2d(rec[0], rec[1], bins=[ts_bins, id_bins])[0]*1e3/binsize/nrn_cnt)
        
            sps, frs, rts = np.array(sps), np.array(frs), np.array(rts)
            with open(filename, 'wb') as f:
                np.save(f, sps)
                np.save(f, frs)
                np.save(f, rts)
        
        # visualization
        frs = frs[:, 1::2, :]
        covin = np.zeros((len(frs), 3, 3))
        covout = np.zeros((len(frs), 3, 3))
        for idx in range(len(frs)):
            alpha, beta = alphas[idx], betas[idx]
            if args.vtype == 'covariance':
                a, b = args.a, args.b
                sigma_e = 3.0/(1 + abs(1/a) + abs(1/b))
                sigma_p = sigma_e / abs(a)
                sigma_s = sigma_e / abs(b)
                covin[idx] = np.array([[sigma_e, beta, alpha], [beta, sigma_p, gamma], [alpha, gamma, sigma_s]])
                if (a < 0) & (b > 0):
                    covin[idx] = np.array([[sigma_e, -beta, alpha], [-beta, sigma_p, -gamma], [alpha, -gamma, sigma_s]])
                elif (a > 0) & (b < 0):
                    covin[idx] = np.array([[sigma_e, beta, -alpha], [beta, sigma_p, -gamma], [-alpha, -gamma, sigma_s]])
                elif (a < 0) & (b < 0):
                    covin[idx] = np.array([[sigma_e, -beta, -alpha], [-beta, sigma_p, gamma], [-alpha, gamma, sigma_s]])
            elif args.vtype == 'variance':
                sigma_e = 3.0/(1 + abs(1/alpha) + abs(1/beta))
                sigma_p = sigma_e / abs(alpha)
                sigma_s = sigma_e / abs(beta)
                covin[idx] = np.array([[sigma_e, gamma, gamma], [gamma, sigma_p, gamma], [gamma, gamma, sigma_s]])
                if (alpha < 0) & (beta > 0):
                    covin[idx] = np.array([[sigma_e, -gamma, gamma], [-gamma, sigma_p, -gamma], [gamma, -gamma, sigma_s]])
                elif (alpha > 0) & (beta < 0):
                    covin[idx] = np.array([[sigma_e, gamma, -gamma], [gamma, sigma_p, -gamma], [-gamma, -gamma, sigma_s]])
                elif (alpha < 0) & (beta < 0):
                    covin[idx] = np.array([[sigma_e, -gamma, -gamma], [-gamma, sigma_p, gamma], [-gamma, gamma, sigma_s]])
                
            covout[idx] = np.cov(frs[idx,:,:3].T)
        covmin, covmax = np.min([covin.min(), covout.min()]), np.max([covin.max(), covout.max()])
        fig = plt.figure(figsize=(3*len(frs), 5))
        for idx in range(len(frs)):
            ax = fig.add_subplot(2, len(frs), idx+1)
            ax.imshow(covin[idx], vmin=covmin, vmax=covmax)
            if idx == 0:
                ax.text(s=r'$Cov^{in}$', rotation=90, x=-1.3, y=1.2, fontsize='medium')
                ax.xaxis.tick_top()
                ax.set_xticks(range(3))
                ax.set_xticklabels(['E', 'P', 'S'])
                ax.set_yticks(range(3))
                ax.set_yticklabels(['E', 'P', 'S'])
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                
            for (j,i),label in np.ndenumerate(covin[idx]):
                if label < 0:
                    ax.text(i,j,'%.1f'%label,ha='center',va='center',color='w')
                else:
                    ax.text(i,j,'%.1f'%label,ha='center',va='center',color='k')
            ax = fig.add_subplot(2, len(frs), idx+1+len(frs))
            ax.imshow(covout[idx], vmin=covmin, vmax=covmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.text(s=r'$Cov^{out}$', rotation=90, x=-1.3, y=1.2, fontsize='medium')
            for (j,i),label in np.ndenumerate(covout[idx]):
                if label < 0:
                    ax.text(i,j,'%.2f'%label,ha='center',va='center',color='w')
                else:
                    ax.text(i,j,'%.2f'%label,ha='center',va='center',color='k')
            rect = patches.Rectangle((-0.48, -0.48), 0.98, 0.98, linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.savefig(figpath + '/%s_gamma_%.1f.pdf'%(modpath, gamma))
        plt.close()

        # trials
        for idx, ntype in enumerate(nrn_tps):
            fr = frs[:, :, idx]
            rt = rts[:, :, idx]
            offset = 0
            
            fig = plt.figure(figsize=(3*len(fr), 5))
            # sampling distribution
            for idx in range(len(fr)):
                alpha, beta, samps = alphas[idx+offset], betas[idx+offset], sps[idx+offset]
                ax = fig.add_subplot(2, len(frs), idx+1, projection='3d')
                p = ax.scatter(samps[:,0], samps[:,1], samps[:,2], c=fr[idx])
                ax.set_xlim([-0.5, 0.5])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([-0.5, 0.5])
                if idx == 0:
                    ax.set_xticks([-0.5, 0.5])
                    ax.set_yticks([-0.5, 0.5])
                    ax.set_zticks([-0.5, 0.5])
                    ax.set_xlabel(r'$\lambda_E^{in}$', labelpad=-5)
                    ax.set_ylabel(r'$\lambda_P^{in}$', labelpad=-5)
                    ax.set_zlabel(r'$\lambda_S^{in}$', labelpad=-5)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
            cax = plt.axes([0.37, 0.5, 0.5, 0.02])
            cbar = plt.colorbar(p, cax=cax, ax=fig.axes, orientation = 'horizontal')
            cbar.ax.set_ylabel(r'$\lambda_%s^{out}$(Hz)'%ntype)
            
            # trials
            ymax = np.array(rt).max()
            for idx in range(len(fr)):
                rs = np.reshape(rt[idx+offset], (epoch, numts))
                ax = fig.add_subplot(2, len(fr), idx+1+len(fr))
                for r in rs:
                    ax.plot(np.arange(0., T, binsize), r, 'k-', linewidth=1, alpha=0.25)
                ax.plot(np.arange(0., T, binsize), np.mean(rs, axis=0), c='r', linewidth=3)
                ax.axvline(warmup, ymin=0., ymax=ymax, color='b', linestyle='--')
                ax.set_ylim([0., ymax])
                if idx == 0:
                    ax.set_ylabel(r'$\lambda_E^{out}$(Hz)')
                    ax.set_xlabel('t (ms)')
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
            plt.subplots_adjust(hspace=0.5, wspace=0.05)
            plt.savefig(figpath + '/%s_%s_gamma_%.1f.pdf'%(ntype, modpath, gamma))
            plt.close()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
