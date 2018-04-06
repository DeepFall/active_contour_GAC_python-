# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:51:50 2018

@author: Administrator
"""
#import numpy as np
#import scipy.stats as st
#
#def gaussian_kern(kernlen, nsig):
#    #kernlen=np.around(nsig*2)*2+2
#    #Returns a 2D Gaussian kernel array.
#    interval = (2*nsig+1.)/(kernlen)
#    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
#    kern1d = np.diff(st.norm.cdf(x))
#    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#    kernel = kernel_raw/kernel_raw.sum()
#    return kernel

import numpy as np
import scipy.ndimage.filters as fi

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)