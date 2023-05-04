import numpy as np
import pandas as pd
from skimage.morphology import binary_dilation as bd
from skimage.filters import gaussian
from scipy.spatial.distance import cdist


def coord_array(dfd, channel):
    sx = dfd[channel].X
    sy = dfd[channel].Y
    x = sx.to_numpy(sx)
    y = sy.to_numpy(sy)
    res = np.stack([x, y], axis=1)
    #print(res.shape)
    return res

def min_distances(a1, a0):
    ''' return the indices in a0 that are closest to indices in a1'''
    dm = cdist(a1, a0)
    dm_near = dm.argmin(axis=1)
    return dm_near, dm.min(axis=1)

def set_channel_pixels(cdf, channel, g, display=1):
    i = channel -1 
    d = cdf[cdf.display == display]
    gx = d['xi']
    gy = d['yi']
    g[gy, gx, i] = 1
    
    selem = np.ones((3,3))
    g[:,:,i] = bd(g[:,:,i], selem)
    g[:,:,i] = gaussian(g[:,:,i].astype(np.float32), 4)

#array_dict = {ch : coord_array(dfdict, ch) for ch in dfdict.keys()}

