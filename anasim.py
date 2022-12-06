# system
import time
import os

# tools
import argparse
import json
import pandas as pd
from itertools import product

# computation libs
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
import scipy.linalg
from scipy.stats import skew

# plot
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import matplotlib.pyplot as plt
import figurefirst as fifi

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def main():
    # create argument parser
    parser = argparse.ArgumentParser(description='Analysis of motif PV-SOM-VIP.')
    # profiling
    parser.add_argument('--exppath', type=str, help='experiment path', default='./experiments/J25.0-m0.5-r0.0')
    parser.add_argument('--transfer', help='generate Fig1.D-F', action='store_true', default=False)
    parser.add_argument('--var', help='generate Fig3.A', action='store_true', default=False)
    parser.add_argument('--cov', help='generate Fig3.B', action='store_true', default=False)
    parser.add_argument('--sig', help='generate Fig3.C', action='store_true', default=False)
    """
    
    Parsing
    """
    print('Parsing arguments ... ... ')
    # parse argument
    args = parser.parse_args()
    # I/O
    exppath = args.exppath
    figpath = exppath.rsplit('/',1)[0] + '/fig/'
    os.makedirs(figpath, exist_ok=True)

    print('Loading data ... ... ')
    # load record setting
    with open(exppath + '/set.json', 'r') as net_fp:
        motif_set = json.loads(net_fp.read())
    nrn_tps, nrn_cnt = motif_set[0], motif_set[1]
    nrn_pos = np.concatenate([[0], np.cumsum(nrn_cnt)])
    print('neuron types, population size:')
    print(list(zip(nrn_tps, nrn_cnt)))

    # load recording data
    with open(exppath + '/fr_list.json') as f:
        fr_range = np.array(json.load(f))
    with open(exppath + '/fr.json') as f:
        fr_on = np.array(json.load(f))

    plt.rcParams['text.usetex'] = True

    # figures to draw
    transfer_map = args.transfer

    var_map = args.var
    sig_map = args.sig
    cov_map = args.cov

    # reshaping data
    eset, pset, sset = [-4, 4.1, 1.0], [-60, 60.1, 15], [-4, 4.1, 1.0]
    erange = list(np.arange(*eset))
    prange = list(np.arange(*pset))
    srange = list(np.arange(*sset))
    erange_old = erange; prange_old = prange; srange_old = srange;
    ein_old, pin_old, sin_old = np.meshgrid(erange, prange, srange, indexing='ij')
    rate_on = np.zeros((len(erange), len(prange), len(srange), len(nrn_tps)))
    for i, fr_val in enumerate(fr_range):
        ee, pp, ss, vv = fr_val
        if (ee < eset[1]) & (pp < pset[1]) & (ss < sset[1]):
            rate_on[erange.index(ee), prange.index(pp), srange.index(ss)] = fr_on[i]

    # target
    rate_tar = rate_on
    cmap_name = 'viridis'

    # figure setting
    figsquare = (3.0, 3.0)
    rasterflag = True

    numpt = 21

    # for nidx in [0, 1, 2, 3]:
    for nidx in [0]:
        # boundary
        rmin = np.min(rate_tar[:,:,:,nidx])
        rmax = np.max(rate_tar[:,:,:,nidx])

        # interpolation
        erange_norm = prange_norm = srange_norm =  list(np.linspace(*[-1.0, 1.0, len(erange)]))
        rinterp = rgi((erange_norm, prange_norm, srange_norm), rate_tar[:,:,:,nidx])

        # 3d tranfer-function
        eset = pset = sset = [-1.0, 1.0, numpt]
        erange = prange = srange = list(np.linspace(*eset))
        ein, pin, sin = np.meshgrid(erange, prange, srange, indexing='ij')
        rin = rinterp(np.array([sin, pin, ein]).T)
        rmin, rmax = rin.min(), rin.max()

        if transfer_map:
            layout = fifi.FigureLayout(figpath + 'transferfunction.svg', autogenlayers=True, make_mplfigures=True, hide_layers=[]);

            # manifold
            points = np.reshape(rate_tar, newshape=(-1, 4))
            ax = layout.axes[('fig_epsv', 'ax_manifold')]
            ax['axis'].remove()
            ax['axis'] = plt.axes(list(ax['axis'].get_position().bounds), projection='3d')
            p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], cmap=cmap_name, rasterized=rasterflag)
            ax.tick_params(axis='x', which='major', pad=-4)
            ax.tick_params(axis='y', which='major', pad=-4)
            ax.tick_params(axis='z', which='major', pad=-4)
            ax.set_xlabel(r'$\lambda_E^{out}$(Hz)', labelpad=-6)
            ax.set_ylabel(r'$\lambda_P^{out}$(Hz)', labelpad=-7)
            ax.set_zlabel(r'$\lambda_S^{out}$(Hz)', labelpad=-8)
            cbar = plt.colorbar(p, fraction=0.1, shrink=0.55, pad=0.15)
            cbar.set_label(r'$\lambda_V^{out}(Hz)$', labelpad=0)

            # transfer function
            ax = layout.axes[('fig_epsv', 'ax_trans')]
            ax['axis'].remove()
            ax['axis'] = plt.axes(list(ax['axis'].get_position().bounds), projection='3d')
            p = ax.scatter(ein, pin, sin, c=rin, cmap=cmap_name, rasterized=rasterflag)
            ax.tick_params(axis='x', which='major', pad=-4)
            ax.tick_params(axis='y', which='major', pad=-4)
            ax.tick_params(axis='z', which='major', pad=-4)
            ax.set_xlabel(r'$\lambda_E^{in}$', labelpad=-6)
            ax.set_ylabel(r'$\lambda_P^{in}$', labelpad=-7)
            ax.set_zlabel(r'$\lambda_S^{in}$', labelpad=-8)

            # iso-curves
            rrange = np.arange(rmin, rmax, 1.0)
            raxes = []
            for rlow, rhigh in zip(rrange[:-1], rrange[1:]):
                ridx = np.where(np.logical_and(rin >= rlow, rin < rhigh))
                raxes.append([ein[ridx], pin[ridx], sin[ridx], rin[ridx]])
            rs = pd.DataFrame(raxes, columns=['e', 'p', 's', 'r'])
            samps = [int(len(rrange)/3), int(len(rrange)/2), int(len(rrange)/3*2)]
            rsheets = pd.DataFrame(rs.iloc[samps])

            ax = layout.axes[('fig_epsv', 'ax_iso')]
            ax['axis'].remove()
            ax['axis'] = plt.axes(list(ax['axis'].get_position().bounds), projection='3d')
            for i, rsheet in rsheets.iterrows():
                x, y, z, r = rsheet.p, rsheet.s, rsheet.e, rsheet.r
                
                X,Y = np.meshgrid(np.linspace(x.min(), x.max(), numpt), np.linspace(y.min(), y.max(), numpt))
                XX, YY = X.flatten(), Y.flatten()

                # best-fit quadratic curve
                xy = np.array([x,y]).T
                A = np.c_[np.ones(len(x)), xy, np.prod(xy, axis=1), xy**2]
                C,_,_,_ = scipy.linalg.lstsq(A, z)
                Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
        
                p = ax.scatter(rsheet.e, rsheet.p, rsheet.s, c=r, s=10, cmap=cmap_name, vmin=rmin, vmax=rmax, rasterized=rasterflag)
            ax.set_axis_off()
            ax.view_init(elev=22., azim=-117.)

            cbar = plt.colorbar(p, fraction=0.1, shrink=0.6, pad=-0.05)
            cbar.set_label(r'$\lambda_E^{out}$(Hz)', labelpad=1)

            # 2d slices
            ax = layout.axes[('fig_epsv', 'ax_slice_ps')]
            ax.imshow(rate_tar[-2,:,:,nidx].T, extent=[pin_old.min(), pin_old.max(), sin_old.min(), sin_old.max()], vmin=rmin, vmax=rmax, aspect=np.abs((pin_old.min()-pin_old.max())/(sin_old.min()-sin_old.max())), origin='lower', rasterized=rasterflag)
            ax.set_xlabel(r'$\lambda_P^{in}$(Hz)')
            ax.set_ylabel(r'$\lambda_S^{in}$(Hz)', labelpad=-5)
            ax.set_title(r'$\lambda_E^{{in}} = {0}$Hz'.format(erange_old[-2]), pad=5)

            ax = layout.axes[('fig_epsv', 'ax_slice_es')]
            ax.imshow(rate_tar[:,2,:,nidx].T, extent=[ein_old.min(), ein_old.max(), sin_old.min(), sin_old.max()],  vmin=rmin, vmax=rmax, aspect=np.abs((ein_old.min()-ein_old.max())/(sin_old.min()-sin_old.max())), origin='lower', rasterized=rasterflag)
            ax.set_xlabel(r'$\lambda_E^{in}$(Hz)')
            ax.set_ylabel(r'$\lambda_S^{in}$(Hz)', labelpad=-5)
            ax.set_title(r'$\lambda_P^{{in}} = {0}$Hz'.format(prange_old[2]), pad=5)

            ax = layout.axes[('fig_epsv', 'ax_slice_ep')]
            ax.imshow(rate_tar[:,:,2,nidx].T, extent=[ein_old.min(), ein_old.max(), pin_old.min(), pin_old.max()],  vmin=rmin, vmax=rmax, aspect=np.abs((ein_old.min()-ein_old.max())/(pin_old.min()-pin_old.max())), origin='lower', rasterized=rasterflag)
            ax.set_xlabel(r'$\lambda_E^{in}$(Hz)')
            ax.set_ylabel(r'$\lambda_P^{in}$(Hz)', labelpad=-5)
            ax.set_title(r'$\lambda_S^{{in}} = {0}$Hz'.format(srange_old[-2]), pad=5)

            layout.append_figure_to_layer(layout.figures['fig_epsv'], 'fig_epsv', cleartarget=True)
            layout.write_svg(figpath + 'transferfunction.svg')

        scale = 5e-2
        corlevel = 0.5
        labels = ['NC', 'PS', 'EP', 'ES']
        cs = ['r', 'b', 'g', 'c', 'm', 'y']
        covs = np.array([ [ [1., 0., 0.], 
                            [0., 1., 0.], 
                            [0., 0., 1.]], 
                          [ [1., 0., 0.], 
                            [0., 1., corlevel], 
                            [0., corlevel, 1.]], 
                          [ [1., corlevel, 0.], 
                            [corlevel, 1., 0.], 
                            [0., 0., 1.]], 
                          [ [1., 0., corlevel], 
                            [0., 1., 0.], 
                            [corlevel, 0., 1.]]])*scale

        means = np.array([[-0.02, 0., 0.], [-0.15, -0.5, 0.45], [0.55, 0.22, 0.35]]) # low rate
        # means = np.array([[0.8, -0.2, -0.6], [0.8, -0.8, 0.25], [0.8, -0.4, 0.0]]) # high rate
        orients = np.array([[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]])
        numsamp = int(1e4)
            
        if var_map:
            lb, rb = 0.9, 0.9
            # variance map
            erange_tmp = prange_tmp = srange_tmp = list(np.linspace(-lb, rb, numpt))
            ein, pin, sin = np.meshgrid(erange_tmp, prange_tmp, srange_tmp, indexing='ij')
            fs, vs, ss = [], [], []
            # ! uncorrelated
            for cov, label in zip([covs[0]], [labels[0]]):
                vin = np.zeros((numpt, numpt, numpt))
                fin = np.zeros((numpt, numpt, numpt))
                skews = np.zeros((numpt, numpt, numpt))
                for e, p, s in zip(ein.ravel(), pin.ravel(), sin.ravel()):
                    mean = np.array([e, p, s])
                    samples = np.random.multivariate_normal(mean, cov, numsamp).T
                    for dim, mrange in zip(range(3), [erange, prange, srange]):
                        samples[dim, samples[dim,:]<=mrange[0]] = mrange[0]
                        samples[dim, samples[dim,:]>=mrange[-1]] = mrange[-1]
                    ratesam = rinterp(np.array([samples[0], samples[1], samples[2]]).T)
                    vin[erange_tmp.index(e), prange_tmp.index(p), srange_tmp.index(s)] = np.var(ratesam)
                    fin[erange_tmp.index(e), prange_tmp.index(p), srange_tmp.index(s)] = np.mean(ratesam)
                    skews[erange_tmp.index(e), prange_tmp.index(p), srange_tmp.index(s)] = skew(ratesam)
 
                fs.append(fin)
                vs.append(vin)
                ss.append(skews)

            fig = plt.figure(figsize=plt.figaspect(0.5))

            alphas ,colors = [0.1, 1.0, 0.5, 1.0, 1.0], ['grey', 'b', 'g', 'r', 'k']
            # rate variance distribution
            ax = fig.add_subplot(1, 2, 2)
            xx, yy = fs[0].ravel(), vs[0].ravel()
            idx_bump = xx < 5.0
            idx_high = (xx > 5.0) & (xx < 18.0) & (yy > 1.5*xx-2)
            idx_low = (xx > 5.0) & (xx < 18.0) & (yy < 1.2*xx-2.5)
            idx_mid = (xx > 5.0) & (xx < 18.0) & (yy < 1.5*xx-2) & (yy > 1.2*xx-2.5)
            idx_sat = xx > 18.0
            idxes = [idx_bump, idx_high, idx_mid, idx_low, idx_sat]
            for idx, c, a in zip(idxes, colors, alphas):
                ax.scatter(xx[idx], yy[idx], s=0.1, c=c, alpha=a, rasterized=rasterflag) 
            # ax.scatter(xx, yy, s=0.5, c='k', rasterized=rasterflag)
            ax.set_xlabel(r'$\lambda_E^{out}$(Hz)')
            ax.set_ylabel(r'$\sigma_{EE}^{out}$')
        
            # variance transfer
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            for idx, c, a in zip(idxes, colors, alphas):
                ax.scatter(ein.ravel()[idx], pin.ravel()[idx], sin.ravel()[idx], s=1.0, c=c, alpha=a, rasterized=rasterflag) 
            ax.tick_params(axis='x', which='major', pad=-3)
            ax.tick_params(axis='y', which='major', pad=-3)
            ax.tick_params(axis='z', which='major', pad=-3)
            ax.set_xlabel(r'$\lambda_E^{in}$', labelpad=-5)
            ax.set_ylabel(r'$\lambda_P^{in}$', labelpad=-6)
            ax.set_zlabel(r'$\lambda_S^{in}$', labelpad=-7)
            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.05])
            ax.set_zlim([-1.05, 1.05])

            plt.savefig(figpath + 'var.pdf', dpi=1000)
            plt.close()

        if sig_map:
            acc = 102
            sigmarange = np.concatenate([-np.power(5, np.linspace(1, -1, int(acc/2))), np.power(5, np.linspace(-1, 1., int(acc/2)))])
            sigmas = list(product(sigmarange, repeat=2))
            corlevel = 0.5

            # sampling covariances
            data = {}
            for i, cov_mean in enumerate(means):
                rates = []
                samps = []
                
                for ab in sigmas:
                    alpha , beta = ab[0], ab[1]
                    sigma_e = 3.0/(1 + abs(1/alpha) + abs(1/beta))
                    sigma_p = sigma_e / abs(alpha)
                    sigma_s = sigma_e / abs(beta)
                    cov_var = np.array([[sigma_e, corlevel, corlevel], [corlevel, sigma_p, corlevel], [corlevel, corlevel, sigma_s]])*scale

                    if is_pos_def(cov_var):
                        samples = np.random.multivariate_normal(np.zeros(3), cov_var, numsamp).T
                        if (alpha < 0) & (beta > 0):
                            samples[1] *= -1
                        elif (alpha > 0) & (beta < 0):
                            samples[2] *= -1
                        elif (alpha < 0) & (beta < 0):
                            samples[0] *= -1
                        samples += np.reshape(cov_mean, (3,1))
                        for dim, mrange in zip(range(3), [erange, prange, srange]):
                            samples[dim, samples[dim,:]<=mrange[0]] = mrange[0]
                            samples[dim, samples[dim,:]>=mrange[-1]] = mrange[-1]
                        ratesam = rinterp(np.array([samples[0], samples[1], samples[2]]).T)
                    else:
                        samples = []
                        ratesam = []
                    samps.append(samples)
                    rates.append(ratesam)

                data['m%d'%i] = {'mean': cov_mean, 'samples': samps, 'rates': rates}

            # saving data
            os.makedirs(figpath, exist_ok=True)
            sigmacnt = len(sigmarange)
            rvars = np.zeros((len(means), sigmacnt, sigmacnt))
            rmeans = np.zeros((len(means), sigmacnt, sigmacnt))
            rskews = np.zeros((len(means), sigmacnt, sigmacnt))
            for idx_m, m in enumerate(means):
                # save path
                sampset = 'm%d'%idx_m
                sampdat = data[sampset]
                for idx_eps in range(len(sigmas)):
                    # indexes
                    idx_ep = int(idx_eps/sigmacnt)
                    idx_es = int(idx_eps-idx_ep*sigmacnt)

                    # samplings
                    rates = sampdat['rates'][idx_eps]
                    if len(rates) > 0:
                        rmeans[idx_m, idx_ep, idx_es] = np.mean(rates)
                        rvars[idx_m, idx_ep, idx_es] = np.var(rates)
                        rskews[idx_m, idx_ep, idx_es] = skew(rates)
                    else:
                        rvars[idx_m, idx_ep, idx_es] = np.nan
                        rmeans[idx_m, idx_ep, idx_es] = np.nan

            fig = plt.figure(figsize=(2,3))
            widths = [1,0.1]
            gs = GridSpec(len(means), 2, figure=fig, width_ratios=widths)
            axes = []

            for idx_m in range(len(means)):
                vardata = rvars[idx_m, :, :]
                meandata = rmeans[idx_m, :, :]
                axes.append(fig.add_subplot(gs[idx_m, 0]))
                ax = axes[-1]
                im = ax.imshow(np.ma.array(vardata, mask=np.isnan(vardata)), extent=[sigmarange[0], sigmarange[-1], sigmarange[0], sigmarange[-1]], origin='lower', interpolation='nearest', cmap=cmap_name, vmin=np.min(vardata[~np.isnan(vardata)]), vmax=np.max(vardata[~np.isnan(vardata)]))
                ax.set_xscale('symlog')
                ax.set_yscale('symlog')
                if idx_m == len(means)-1:
                    ax.tick_params(axis='both', which='major')
                    ax.set_xlabel(r'$\sigma_{E/S}^{in}$', labelpad=-10)
                    ax.set_xticks([sigmarange[0], sigmarange[-1]])
                    ax.set_xticklabels([sigmarange[0], sigmarange[-1]])
                else:
                    ax.set_xticks([])
                if idx_m == 1:
                    ax.set_yticks([sigmarange[0], sigmarange[-1]])
                    ax.set_yticklabels([sigmarange[0], sigmarange[-1]])
                    ax.set_ylabel(r'$\sigma_{E/P}^{in}$', labelpad=-13)
                else:
                    ax.set_yticks([0])

                if idx_m == 0:
                    for xx, yy, label, ha, va in zip([0.3, 3.0, -0.3, 0.3], [0.3, 3.0, 0.3, -0.3], ['I', 'II', 'III', 'IV'], ['left', 'right', 'right', 'left'], ['bottom', 'top', 'bottom', 'top']):
                        ax.scatter(xx, yy, s=3, c='w')
                        ax.text(xx,yy,label, ha=ha, va=va, fontsize='x-small', color='white')
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="8%", pad=0.05)
                fig.colorbar(im, cax=cax)
                if idx_m == 1:
                    cax.tick_params(axis='both', which='major')
                    cax.set_ylabel(r'$\sigma_{EE}^{out}$', rotation=90, labelpad=1) 

            plt.subplots_adjust(left=0.1, right=0.9, wspace=-0.4, hspace=0.05)
            plt.savefig(figpath + 'sig.pdf', dpi=1000)
            plt.close()
        
        if cov_map:
            varange = np.linspace(0., 1.0, 21)
            varrs = list(product(varange, repeat=3))

            # sampling covariances
            data = {}
            for i, cov_mean in enumerate(means):
                alpha , beta = orients[i,0], orients[i,1]
                sigma_e = 3.0/(1 + abs(1/alpha) + abs(1/beta))
                sigma_p = sigma_e / abs(alpha)
                sigma_s = sigma_e / abs(beta)
                covs = []
                rates = []
                samps = []
                for variances in varrs:
                    cov_var = np.array([[sigma_e, variances[0], variances[1]], [variances[0], sigma_p, variances[2]], [variances[1], variances[2], sigma_s]])*scale
                    if is_pos_def(cov_var):
                        samples = np.random.multivariate_normal(np.zeros(3), cov_var, numsamp).T
                        samples[0] *= erange[-1]
                        samples[1] *= prange[-1]
                        samples[2] *= srange[-1]
                        if (alpha < 0) & (beta > 0):
                            samples[1] *= -1
                        elif (alpha > 0) & (beta < 0):
                            samples[2] *= -1
                        elif (alpha < 0) & (beta < 0):
                            samples[0] *= -1
                        samples += np.reshape(cov_mean, (3,1))
                        for dim, mrange in zip(range(3), [erange, prange, srange]):
                            samples[dim, samples[dim,:]<=mrange[0]] = mrange[0]
                            samples[dim, samples[dim,:]>=mrange[-1]] = mrange[-1]
                        ratesam = rinterp(np.array([samples[0], samples[1], samples[2]]).T)
                    else:
                        samples = []
                        ratesam = []

                    covs.append(cov_var)
                    samps.append(samples)
                    rates.append(ratesam)

                data['m%d'%i] = {'mean': cov_mean, 'covs': np.array(covs), 'samples': samps, 'rates': rates}

            # saving data
            os.makedirs(figpath, exist_ok=True)
            rvars = np.zeros((len(means), len(varange), len(varange), len(varange)))
            rmeans = np.zeros((len(means), len(varange), len(varange), len(varange)))
            rskews = np.zeros((len(means), len(varange), len(varange), len(varange)))
            for idx_m, m in enumerate(means):
                # save path
                sampset = 'm%d'%idx_m
                sampdat = data[sampset]
                for idx_eps in range(len(varrs)):
                    # indexes
                    idx_ep = int(idx_eps/len(varange)/len(varange))
                    idx_es = int(idx_eps/len(varange)-idx_ep*len(varange))
                    idx_ps = idx_eps - idx_ep*len(varange)*len(varange) - idx_es*len(varange)

                    # samplings
                    rates = sampdat['rates'][idx_eps]
                    cov_var = sampdat['covs'][idx_eps]
                    if len(rates) > 0:
                        rmeans[idx_m, idx_ep, idx_es, idx_ps] = np.mean(rates)
                        rvars[idx_m, idx_ep, idx_es, idx_ps] = np.var(rates)
                        rskews[idx_m, idx_ep, idx_es, idx_ps] = skew(rates)
                    else:
                        rvars[idx_m, idx_ep, idx_es, idx_ps] = np.nan
                        rmeans[idx_m, idx_ep, idx_es, idx_ps] = np.nan

            # low-mid-high
            ps_samp = [6, 10, 14]
            fig = plt.figure(figsize=(4,3.5))

            # Design your figure properties
            widths = [1,1,1.15]
            gs = GridSpec(len(means), len(ps_samp), figure=fig, width_ratios=widths)
            axes = []

            for idx_m, c in enumerate(cs[:len(means)]):
                vardata = rvars[idx_m, :, :, :]
                # vardata = rskews[idx_m, :, :, :]
                meandata = rmeans[idx_m, :, :, :]
                for i, idx_ps in enumerate(ps_samp):
                    axes.append(fig.add_subplot(gs[idx_m, i]))
                    ps = varange[idx_ps]
                    ax = axes[-1]
                    im = ax.imshow(np.ma.array(vardata[:, :, idx_ps], mask=np.isnan(vardata[:, :, idx_ps])), extent=[0, 1, 0, 1], origin='lower', interpolation='nearest', cmap=cmap_name, vmin=np.min(vardata[~np.isnan(vardata)]), vmax=np.max(vardata[~np.isnan(vardata)]))
                    if (idx_m==len(means)-1) & (i==1):
                        ax.set_xlabel(r'$\sigma_{ES}^{in}$', labelpad=-10)
                        ax.set_xticks([0, 1])
                    else:
                        if idx_m==len(means)-1:
                            ax.set_xticks([0.5])
                        else:
                            ax.set_xticks([])
                    
                    if (idx_m == 0) & (i==1):
                        for xx, yy, label in zip([0.1, 0.4, 0.1, 0.4], [0.1, 0.1, 0.4, 0.4], ['I', 'II', 'III', 'IV']):
                            ax.scatter(xx, yy, s=3, c='w')
                            ax.text(xx,yy,label, ha='left', va='bottom', fontsize='x-small', color='gray')

                    if (idx_m==1) & (i==0):
                        ax.set_ylabel(r'$\sigma_{EP}^{in}$', labelpad=-7)
                        ax.set_yticks([0, 1])
                    else:
                        if i == 0:
                            ax.set_yticks([0.5])
                        else:
                            ax.set_yticks([])
                    
                    if i == 0:
                        ax.text(s=r'$\lambda_E^{out}$=%.1fHz'%np.mean(meandata[~np.isnan(meandata)]), rotation=90, x=-0.6, y=0, fontsize='medium', color=c)

                    if idx_m == 0:
                        ax.set_title(r'$\sigma_{PS}^{in}$=%.1f'%ps, fontsize='medium')

                # colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="8%", pad=0.05)
                fig.colorbar(im, cax=cax)
                if idx_m == 1:
                    cax.tick_params(axis='both', which='major')
                    cax.set_ylabel(r'$\sigma_{EE}^{out}$', rotation=90, labelpad=1) 
                    # cax.set_ylabel(r'$skew_{EE}^{out}$', rotation=90, labelpad=1)

            plt.subplots_adjust(left=0.2, right=0.85, wspace=0.1, hspace=-0.2)
            plt.savefig(figpath + 'cov.pdf', dpi=1000)
            # plt.savefig('./fig/critical/cov.pdf', dpi=1000)
            plt.close()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
