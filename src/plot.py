# -*- coding: utf-8 -*-
"""
Custom Plotter to draw a map and display meteo data 
interpolation

@author: Tâm Le Minh
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.patches import Path, PathPatch


##Formatting functions
def format_grid_for_prediction(ylat, ylon):
    grid = np.column_stack((np.hstack((ylat)),np.hstack((ylon))))
    return grid

def format_prediction_to_grid(prediction, nrow, ncol):
    matrix = np.reshape(prediction, (nrow,ncol))  
    return matrix


##Plotter class
class MeteoPlotter:
    
    ##Map boundaries
    lon_min = -15.56
    lat_min = 24.65
    lon_max = 49.88
    lat_max = 79.17
    
    def __init__(self, lon_min = -15.56, lat_min = 24.65, lon_max = 49.88, lat_max = 79.17, 
                 resolution = 'l', epsg = 4668, nx = 100, ny = 100):
        
        self.lon_min = lon_min
        self.lat_min = lat_min
        self.lon_max = lon_max
        self.lat_max = lat_max
        self._create_map(resolution=resolution, epsg=epsg)
        self._create_grid(nx, ny)
        self.predict = np.empty([nx, ny])
        
    ##Initialize the map object
    def _create_map(self, resolution = 'l', epsg = 4668):
        self.m = Basemap(llcrnrlon = self.lon_min, llcrnrlat = self.lat_min, urcrnrlon = self.lon_max, urcrnrlat = self.lat_max, 
                      resolution = resolution, epsg = epsg)

    ##Initialize meshgrid to predict to plot the surfaces
    def _create_grid(self, nx, ny):
        glons, glats = self.m.makegrid(nx, ny)
        self.gx, self.gy = self.m(glons, glats)
        self.grid = format_grid_for_prediction(glats, glons)

    ##Predict the values of the grid
    def _predict_grid(self, model):
        preds = model.predict(self.grid)
        self.predict = format_prediction_to_grid(preds, self.predict.shape[0], self.predict.shape[1])
    
    ##Display temperature annotations on the figure
    def _draw_annotations(self, X, Y):
        lon = X[:,1]
        lat = X[:,0]
        temps = Y[:,0]
        x, y = self.m(lon, lat)
        
        for i in range(0, len(x)):
            if lon[i] > self.lon_min and lon[i] < self.lon_max and lat[i] > self.lat_min and lat[i] < self.lat_max:
                plt.text(x[i], y[i], "{0:.1f}".format(temps[i]),fontsize=10,fontweight='bold', ha='center',va='center',color='k')
        
    ##Display surfaces of isotemperature    
    def _plot_contours(self):
        clevs = [-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20,22]
        cs = self.m.contourf(self.gx, self.gy, self.predict, clevs, cmap='Spectral_r')
        
        cbar = self.m.colorbar(cs,location='bottom',pad="5%")
        cbar.set_label('degrees Celsius')
        
    ##Masks the ocean surfaces
    def _mask_ocean(self):
        ##Getting the limits of the map:
        x0,x1 = self.ax.get_xlim()
        y0,y1 = self.ax.get_ylim()
        map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    
        ##Getting all polygons used to draw the coastlines of the map
        polys = [p.boundary for p in self.m.landpolygons]
    
        ##Combining with map edges
        polys = [map_edges]+polys[:]
    
        ##Creating a PathPatch
        codes = [
            [Path.MOVETO] + [Path.LINETO for p in p[1:]]
            for p in polys
        ]
        polys_lin = [v for p in polys for v in p]
        codes_lin = [c for cs in codes for c in cs]
        path = Path(polys_lin, codes_lin)
        patch = PathPatch(path,facecolor='cyan',lw=0)
        return patch
 
    ##Plot the map on the figure
    def plot_map(self, model, X, Y):
        self._predict_grid(model)
        
        self.fig, self.ax = plt.subplots(figsize=(24,24))
        
        self.m.drawmapboundary(fill_color='white')
        self.m.drawcoastlines()
        
        self._draw_annotations(X, Y)
        
        self._plot_contours()
        patch = self._mask_ocean()
        self.ax.add_patch(patch)
        self.ax.set_title(model.__class__.__name__, fontsize=36)
        plt.show()