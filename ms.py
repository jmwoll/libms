# Copyright (C) 2018  Jan Wollschläger <jm.wollschlaeger@gmail.com>
# This file is part of libms.
#
# libms is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from numba import jit

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams['svg.fonttype'] = 'none'

def cut_xy(xs, ys, x_from, x_to):
    x_from = min(x_from, x_to)
    x_to = max(x_from, x_to)
    from_idx = None
    to_idx = None
    for i,x in enumerate(xs):
        if x > x_from:
            from_idx = i
            break
    for i,x in enumerate(xs):
        if x > x_to:
            to_idx = i
            break
    assert(from_idx!=None)
    assert(to_idx!=None)
    return (xs[from_idx:to_idx], ys[from_idx:to_idx])

def cut_int(xs, thresh=0.0, cut_to=0.0):
    return list(map(lambda x: x if x > thresh else cut_to, xs))

def norm(xs):
    lstmax, lstmin = max(xs), min(xs)
    return list(map(lambda x: (x - lstmin) / float(lstmax - lstmin) ,xs))

def scale(ys, scale=1):
    return list(map(lambda y: y*scale, ys))

def load_xy(fle):
    inp_str = None
    with open(fle,'r') as fin:
        inp_str = fin.read()
    if inp_str is None:
        raise Exception("Error reading file {}".format(fle))
    xs,ys = [],[]
    for lne in inp_str.strip().replace('\t',' ').replace('  ', ' ').replace('  ', ' ').split('\n'):
        lne = lne.split(' ')
        try:
            xs.append(float(lne[0]))
            ys.append(float(lne[1]))
        except:
            print('error reading line:\n'+lne)
    return xs,ys

def load_ms(fle, sep=','):
    xs = []
    ys = []

    if sep == ',':
        with open(fle, 'r') as f_in:
            data = f_in.read().strip()
            for lne in data.split('\n'):
                lne = lne.split(sep)
                lne = ''.join([lne[0],'.',lne[1],sep,lne[2],lne[3]])
                lne = lne.split(sep)
                xs.append(float(lne[0]))
                ys.append(float(lne[1]))
    else:
        with open(fle, 'r') as f_in:
            data = f_in.read().strip()
            for lne in data.split('\n'):
                try:
                    lne = lne.split(sep)
                    xs.append(float(lne[0]))
                    ys.append(float(lne[1]))
                except:
                    print('Warning -> Skipping line:\n', lne)
    return xs, ys

def annotate_ms(xs, ys, int_thresh=.5, margin_x=0.05, margin_y=0.05, decimal_places=1):
    # annotate the given mass spectrum, such that the peaks
    # are annoated by their respective m/z
    max_y = max(ys)
    format_str = '{0:.'+str(decimal_places)+'f}' if decimal_places is not None else '{}'
    def annotate_peak(x,y):
        #plt.gca().annotate('{}'.format(x),(x,y),textcoords='data',horizontal_alignment='center')
        plt.gca().text(x,y+margin_y*max_y,format_str.format(x),horizontalalignment="center")

    thresh = int_thresh * max_y
    visited = {}
    for i,x in enumerate(xs):
        y = ys[i]
        try:
            if y >= max(ys[i-10:i]+ys[i+1:i+10]):
                if not(x in visited) and y > thresh:
                    visited[x] = True
                    #print('annotate {} {}'.format(x,y))
                    annotate_peak(x,y)
        except IndexError:
            pass

_plot_mass_spectrum_cache, _plot_mass_spectrum_cache_size = {}, 4
def plot_mass_spectrum(sample, start=None, end=None, int_thresh=.5, margin_x=0.05, margin_y=0.05,
                       save_as=None, scale_relative=True, decimal_places=1, remove_frequencies=False,
                       title=None, add_background_frequencies=False,
                       report=False, process_ys=None, process_xs=None,
                       process_xy=None, direct_data_feed=False, report_n=None,
                       get_data=None, integrate=None, show=None, rasterize=None,
                       fig_size=None, cache=None, annotate=None):
    global _plot_mass_spectrum_cache
    if len(_plot_mass_spectrum_cache) > _plot_mass_spectrum_cache_size:
        _plot_mass_spectrum_cache = {}

    xs,ys = None,None
    if direct_data_feed:
        assert(sample is not None)
        xs,ys = sample
    else:
        if cache is None:
            cache = True
        if cache and sample in _plot_mass_spectrum_cache:
            xs, ys = _plot_mass_spectrum_cache[sample]
        else:
            try:
                xs, ys = load_ms(sample, sep=',')
            except:
                xs, ys = load_ms(sample, sep='\t')
            if cache:
                _plot_mass_spectrum_cache[sample] = (xs,ys)

    assert(xs != None and ys != None)
    org_xs, org_ys = [itm for itm in xs], [itm for itm in ys]
    if process_xy is not None:
        assert(process_xs is None and process_ys is None)
        xs, ys = process_xy(xs, ys)
    elif process_xs is not None:
        xs = list(process_xs(xs))
    elif process_ys is not None:
        ys = list(process_ys(ys))

    if remove_frequencies:
        xs, ys = remove_background_frequencies(xs, ys,
                                add_background_freqs=add_background_frequencies)

    if report:
        show_report(xs, ys, report_n=report_n)

    if fig_size is not None:
        #plt.figure(figsize=(8*1.5,6*1.5))
        plt.figure(figsize=fig_size)

    if (start is not None) and (end is not None):
        xs, ys = cut_xy(xs, ys, start, end)

    if scale_relative:
        ys = norm(ys)
        ys = scale(ys,scale=100)

        plt.ylim([0,105])

    if start is None:
        start = min(xs)
    if end is None:
        end = max(xs)
    ax = plt.gca()
    ax.get_yaxis().set_tick_params(right=False, which='both', direction='out')
    ax.get_xaxis().set_tick_params(top=False, which='both', direction='out')
    plt.xlabel('m / z', style='italic')
    if scale_relative:
        plt.ylabel('rel. intensity / %')
    else:
        plt.ylabel('arb. intensity / counts')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if title is not None:
        plt.title(title)
    #xs,ys = cut_xy(xs,ys,start,end)
    if rasterize is not None:
        if rasterize:
            rasterize = True
    else:
        rasterize = False
    if rasterize:
        plt.plot(xs,ys,color='black',rasterized=rasterize)
    else:
        plt.plot(xs,ys,color='black')
    if annotate is None:
        annotate = True
    if show is None:
        show = True
    if show and annotate:
        annotate_ms(xs,ys,int_thresh=int_thresh,margin_x=margin_x,margin_y=margin_y,decimal_places=decimal_places)

    if save_as is not None:
        plt.savefig(save_as)

    if show == False:
        plt.clf()
    else:
        plt.show()

    data = {}
    if integrate is not None:
        assert(len(integrate) == 2)
        int_from = integrate[0]
        int_to = integrate[1]
        assert(int_from is not None)
        assert(int_to is not None)
        intxs, intys = cut_xy(org_xs, org_ys, int_from, int_to)
        data['int'] = np.trapz(intys, x=intxs)
        get_data = True

    if get_data is not None:
        if get_data:
            data['xs'] = xs
            data['ys'] = ys
            return data
#
