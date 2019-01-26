# -*- coding: utf-8 -*-
"""
Custom Pipeline for applying models to meteo data and 
benchmarking models in this regard

@author: Tâm Le Minh
"""
import os
import numpy as np
import pandas as pd
import time, datetime
import kernels, models
from sklearn.model_selection import train_test_split
import ast
from plot import MeteoPlotter


##Formatting function
def format_for_learning(xlat, xlon, temp):
    X = np.column_stack((xlat, xlon))
    Y = np.asarray(temp).reshape(len(temp),1)
    return X, Y


##Pipeline class
class MeteoPipeline:
    
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

        
    ##Create the report folder with the current date and time
    def _create_report_folder(self):
        t = datetime.datetime.now()
        self.report_path = 'reports/{0:%Y_%m_%d-%H_%M_%S}'.format(t)
        os.mkdir(self.report_path)

        
    ##Load meteo data from a csv file
    def load_data(self, filename):
        self.datafile = filename
        df = pd.read_csv(self.data_path + self.datafile, ',')
        df = df.drop(['Unnamed: 0'], axis=1)

        self.X_all = df[['Lat', 'Lon']].values
        self.y_all = df[['Temp']].values
            
            
    ##Load models and parameters from a csv file
    def load_models(self, file_path_name):
        df = pd.read_csv(file_path_name, ',')
        df['Parameters'] = df['Parameters'].apply(ast.literal_eval)
        df['Model'] = 'models.' + df['Model']
        for idx, model in df['Model'].iteritems():
            df.loc[idx, 'Model'] = eval(model)()
            df.loc[idx, 'Model'].set_params(**df.loc[idx, 'Parameters'])
            
        model_list = df['Model'].values
        
        return model_list

        
    ##Partition data into train and test sets
    def partition_train_test(self, test_size=0.2):
        if self.verbose:
            print("Partition data into train and test samples")
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_all, 
                                                                                self.y_all, test_size=0.2)
        
        if self.verbose:
            print("Number of train samples: " + str(self.X_train.shape[0]))
            print("Number of test samples: " + str(self.X_test.shape[0]))
        
        
    ##Perform an interpolation with one model and parameters
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
        

    ##Optimize a model w.r.t. a parameter grid
    def simple_optimization(self, model, param_grid, cv=None, n_restart=5):
        if self.report:
            bestc, bests, CV_res = model.optimize(self.X_train, self.y_train, cv=cv, 
                                    n_splits=n_restart, info=True, **param_grid)
            CV_res.to_csv(self.report_path + '/CV_%s.csv' % model.get_name())
        else:
            bestc, bests = model.optimize(self.X_train, self.y_train, cv=cv, 
                                    n_splits=n_restart, info=True, **param_grid)
            
        return bestc, bests
            
        
    ##Perform a benchmark with or without optimizing the models
    def benchmark(self, model_list, param_grid_list=None, optim=False, cv=None, n_restart=5):
        if self.report:
            save = pd.DataFrame(columns=['Model', 'Train Score', 'Test Score', 'Parameters', 'Database'])
            
        for i in range(0, len(model_list)):
            model = model_list[i]
            print(model.__class__.__name__)
            if param_grid_list is None:
                print(model.get_params())
            else:
                print(param_grid_list[i])
            
            if optim:
                params = param_grid_list[i]
                model_parameters, score_train = self.simple_optimization(model, params, cv=cv, n_restart=n_restart)
                model.set_params(**model_parameters)
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

            
    ##Plot predictions for a model
    def plot(self, model, nx=100, ny=100):
        plotter = MeteoPlotter(nx=nx, ny=ny)
        plotter.plot_map(model, self.X_all, self.y_all)
        