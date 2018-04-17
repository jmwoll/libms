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

import math
import numpy as np


def range_dict(dct):
    rslt = { }
    for k in dct:
        if len(k) > 1:
            for i in range(k[0],k[1]+1,1):
                rslt[i] = dct[k]
        else:
            rslt[k] = dct[k]
    return rslt


def fwhm(xs,ys):
    def nearest(lst,val):
        best_d,best_i = abs(lst[0]-val),0
        for i,x in enumerate(lst):
            if abs(x-val) < best_d:
                best_d,best_i=abs(x-val),i
        return best_i

    max_y = max(ys)
    max_x = xs[ys.index(max_y)]
    return abs(xs[nearest(ys,max_y/2)]-max_x)*2




def sigmoid(t, a, b, c, d, e):
    return d + a / (b + math.e**(c*(t-e)))

def simple_sigmoid(t, a, b, c, d):
    return a + b * 1 / (1 + math.e ** (c * (t - d)))

def scaled_sigmoid(t, a, b, c, d, e):
    return e * simple_sigmoid(t, a, b, c, d)

def rigid_sigmoid(t, a, b):
    return 1 / (1 + math.e ** (a * (t - b)))

def rigid_sigmoid_with_bias(t, a, b, c):
    return c + rigid_sigmoid(t, a, b)

def gauss(x,a,b,c):
    return a* math.e ** (-((x-b)**2)/(2*c**2))


def mean(lst):
    return sum(lst) / float(len(lst))

def std(lst):
    cpy_lst = [itm for itm in lst]
    return np.std(cpy_lst)
