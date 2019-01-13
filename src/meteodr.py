import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.patches import Path, PathPatch
import time, datetime
import CustomKernels, CustomModels
from sklearn.model_selection import train_test_split
import ast


##Formatting functions
def format_for_learning(xlat, xlon, temp):
    X = np.column_stack((xlat, xlon))
    Y = np.asarray(temp).reshape(len(temp),1)
    return X, Y

def format_grid_for_prediction(ylat, ylon):
    grid = np.column_stack((np.hstack((ylat)),np.hstack((ylon))))
    return grid
    
def format_prediction_to_grid(prediction, nrow, ncol):
    matrix = np.reshape(prediction, (nrow,ncol))  
    return matrix

class Pipeline:
    
    report_path = ""
    data_path = "data/current-version/"
    datafile = ""
    report = False
    verbose = False
    
    def __init__(self, data_path="data/current-version/", data_file=None, verbose=False, report=False):
        if report:
            self._create_report_folder()
        self.data_path = data_path
        if not data_file is None:
            self.load_data(data_file)
        self.verbose = verbose
        self.report = report

    def _create_report_folder(self):
        t = datetime.datetime.now()
        self.report_path = 'reports/{0:%Y_%m_%d-%H_%M_%S}'.format(t)
        os.mkdir(self.report_path)

    def load_data(self, filename):
        self.datafile = filename
        df = pd.read_csv(self.data_path + self.datafile, ',')
        df = df.drop(['Unnamed: 0'], axis=1)

        self.X_all = df[['Lat', 'Lon']].values
        self.y_all = df[['Temp']].values

    def partition_train_test(self, test_size=0.2):
        if self.verbose:
            print("Partition data into train and test samples")
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_all, 
                                                                                self.y_all, test_size=0.2)
        
        if self.verbose:
            print("Number of train samples: " + str(self.X_train.shape[0]))
            print("Number of test samples: " + str(self.X_test.shape[0]))
        
        
    def simple_interpolation(self, model, params=None):
        if not params is None:
            model.set_params(**params)
            
        score_train = model.train(self.X_train, self.y_train, eval_score=True)
        
        if self.verbose:
            print("Train MSE score: " + str(score_train))
            
        self.predictions, score_test = model.predict(self.X_test, self.y_test, eval_score=True)
        
        if self.verbose:
            print("Test MSE score: " + str(score_test))
            
        return score_train, score_test
            

    def simple_optimization(self, model, param_grid, cv=None, n_restart=5):
        if self.report:
            bestc, bests, CV_res = model.optimize(self.X_train, self.y_train, cv=cv, 
                                    n_splits=n_restart, info=True, **param_grid)
            CV_res.to_csv(self.report_path + '/CV_%s.csv' % model.get_name())
        else:
            bestc, bests = model.optimize(self.X_train, self.y_train, cv=cv, 
                                    n_splits=n_restart, info=True, **param_grid)
            
        return bestc, bests
            
    def benchmark(self, model_list, param_grid_list=None, optim=False, cv=None, n_restart=5):
        if self.report:
            save = pd.DataFrame(columns=['Model', 'Train Score', 'Test Score', 'Parameters', 'Database'])
            
        for i in range(0, len(model_list)):
            model = model_list[i]
            
            if optim:
                params = param_grid_list[i]
                print(params)
                model_parameters, score_train = self.simple_optimization(model, params, cv=cv, n_restart=n_restart)
                self.predictions, score_test = model.predict(self.X_test, self.y_test, eval_score=True)
                model_name = model.to_string()
                model_parameters = str(model_parameters)
                
            else:
                if not param_grid_list is None:
                    params = param_grid_list[i]
                else:
                    params = None
                score_train, score_test = self.simple_interpolation(model, params)
                model_name, model_parameters = model.to_string()
                
            if self.report:
                save.loc[i] = [model_name, score_train, score_test, model_parameters, self.datafile]
        
        if self.report:
            save.to_csv(self.report_path + '/Benchmark.csv')
            
            
    def load_models(self, file_path_name):
        df = pd.read_csv(file_path_name, ',')
        df['Parameters'] = df['Parameters'].apply(ast.literal_eval)
        df['Model'] = 'CustomModels.' + df['Model']
        for idx, model in df['Model'].iteritems():
            df.loc[idx, 'Model'] = eval(model)()
            df.loc[idx, 'Model'].set_params(**df.loc[idx, 'Parameters'])
            
        model_list = df['Model'].values
        
        return model_list

    def plot(self, model, nx=100, ny=100):
        plotter = Plotter(nx=nx, ny=ny)
        plotter.plot_map(model, self.X_all, self.y_all)
        
        
class Plotter:
    
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
        
    def _create_map(self, resolution = 'l', epsg = 4668):
        self.m = Basemap(llcrnrlon = self.lon_min, llcrnrlat = self.lat_min, urcrnrlon = self.lon_max, urcrnrlat = self.lat_max, 
                      resolution = resolution, epsg = epsg)

    def _create_grid(self, nx, ny):
        glons, glats = self.m.makegrid(nx, ny)
        self.gx, self.gy = self.m(glons, glats)
        self.grid = format_grid_for_prediction(glats, glons)

    def _predict_grid(self, model):
        preds = model.predict(self.grid)
        self.predict = format_prediction_to_grid(preds, self.predict.shape[0], self.predict.shape[1])
        
    def _draw_annotations(self, X, Y):
        lon = X[:,1]
        lat = X[:,0]
        temps = Y[:,0]
        x, y = self.m(lon, lat)
        
        for i in range(0, len(x)):
            if lon[i] > self.lon_min and lon[i] < self.lon_max and lat[i] > self.lat_min and lat[i] < self.lat_max:
                plt.text(x[i], y[i], "{0:.1f}".format(temps[i]),fontsize=10,fontweight='bold', ha='center',va='center',color='k')
        
    def _plot_contours(self):
        clevs = [-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20,22]
        cs = self.m.contourf(self.gx, self.gy, self.predict, clevs, cmap='Spectral_r')
        
        cbar = self.m.colorbar(cs,location='bottom',pad="5%")
        cbar.set_label('degrees Celsius')
        
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
 
    def plot_map(self, model, X, Y):
        self._predict_grid(model)
        
        self.fig, self.ax = plt.subplots(figsize=(24,24))
        
        self.m.drawmapboundary(fill_color='white')
        self.m.drawcoastlines()
        
        self._draw_annotations(X, Y)
        
        self._plot_contours()
        patch = self._mask_ocean()
        self.ax.add_patch(patch)
        plt.show()
        