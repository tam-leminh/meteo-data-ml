from scipy._lib.six import callable, string_types
import numpy as np
import math

#Define a distance
def sq_distance(lat1, lon1, lat2, lon2):
    d = (lon2-lon1)**2 + (lat2-lat1)**2
    return d
    
def hv_distance(lat1, lon1, lat2, lon2):
    radius = 6371 # km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d
    
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
            
            k = 0
            for i in range(0, m):
                for j in range(i+1, m):
                    dm[k] = hv_distance(X[i][0],X[i][1],X[j][0],X[j][1])
                    k += 1
                    
            return dm
                    
        elif mstr == 'euclidian':
        
            k = 0
            for i in range(0, m):
                for j in range(i+1, m):
                    dm[k] = sq_distance(X[i][0],X[i][1],X[j][0],X[j][1])
                    k += 1
                
            return dm
        
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm
    
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
        
            for i in range(0, mA):
                for j in range(0, mB):
                    dm[i][j] = hv_distance(XA[i][0],XA[i][1],XB[j][0],XB[j][1])

        elif mstr == 'euclidian':
        
            for i in range(0, mA):
                for j in range(0, mB):
                    dm[i][j] = sq_distance(XA[i][0],XA[i][1],XB[j][0],XB[j][1])
        
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm