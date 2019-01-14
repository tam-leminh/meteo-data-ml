# -*- coding: utf-8 -*-
"""
Custom distances, notably Great Circle distance with Haversine formula
Includes custom pairwise distances functions cdist and pdist

@author: Tâm Le Minh
"""
from scipy._lib.six import callable, string_types
import numpy as np
import math


##Euclidian angular distance, should be used only for testing
def sq_distance(lat1, lon1, lat2, lon2):
    if isinstance(lat1, np.ndarray):
        d = np.sqrt(np.power(lon2-lon1,2) + np.power(lat2-lat1,2))
        if len(d.shape) == 1:
            d = d.reshape([d.shape[0],1])
    else: 
        d = math.sqrt((lon2-lon1)**2 + (lat2-lat1)**2)
    return d
    
    
##Great circle distance with Haversine formula
def hv_distance(lat1, lon1, lat2, lon2):
    radius = 6371 # km
    if isinstance(lat1, np.ndarray) or isinstance(lat2, np.ndarray):
        dlat = np.radians(lat2-lat1)
        dlon = np.radians(lon2-lon1)
        a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
            * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        if len(c.shape) == 1:
            c = c.reshape([c.shape[0],1])
    else:
        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d
    
    
##Returns the distances between elements of a vector
def cPdist(X, metric='haversine'):

    X = np.asarray(X, order='c')

    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m, n = s
    dm = np.empty((m * (m - 1)) // 2, dtype=np.double)

    if isinstance(metric, string_types):
        mstr = metric.lower()

        if mstr == 'haversine':
            
            dm = hv_distance(X[:,0,None], X[:,1,None], X[:,0], X[:,1])
            return dm
                    
        elif mstr == 'euclidian':
        
            dm = sq_distance(X[:,0,None], X[:,1,None], X[:,0], X[:,1])
            return dm
        
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm
    
    
##Returns the pariwise distances between elements of 2 vectors
def cCdist(XA, XB, metric='haversine'):

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]
    dm = np.empty((mA, mB), dtype=np.double)

    if isinstance(metric, string_types):
    
        mstr = metric.lower()
        
        if mstr == 'haversine':
            dm = hv_distance(XA[:,0,None], XA[:,1,None], XB[:,0], XB[:,1])   
            return dm

        elif mstr == 'euclidian':
            dm = sq_distance(XA[:,0,None], XA[:,1,None], XB[:,0], XB[:,1])   
            return dm
        
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm