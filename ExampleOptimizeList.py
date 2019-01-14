# -*- coding: utf-8 -*-
"""
Example script for optimizing a list of models and plotting the results 
for the best candidate from each model

@author: Tâm Le Minh
"""
from src.meteodr import Pipeline, Plotter
import src.CustomModels as CustomModels


##Prepare the data and the pipeline
datafile = "Temp-2019_01_04-15_47.csv"
pipe = MeteoPipeline(data_file=datafile, verbose=True, report=True)
pipe.partition_train_test(test_size=0.2)

##Prepare the model list
model_list = [
    CustomModels.NearestNeighbor(),
    CustomModels.InverseDistanceWeighting(),
    CustomModels.GaussianProcess(),
    CustomModels.GeographicallyWeightedRegressor(),
    CustomModels.RegressionTree(),
    CustomModels.RandomForest(),
    CustomModels.ExtraTrees(),
    CustomModels.SupportVectorRegression()
]

##Define the parameters associated with each model
param_grid_list = [
    {
        
    },
    {
        'radius' : [10, 100, 1000]
    },
    {
        
    },
    {
        
    },
    {
        'max_depth' : [9, 10, 11]
    },
    {
        'n_estimators' : [1000, 5000],
        'max_depth' : [9, 10, 11]
    },
    {
        'n_estimators' : [1000, 5000],
        'max_depth' : [9, 10, 11]
    },
    {
        'gamma' : [0.004, 0.02, 0.1],
        'C' : [1.0, 10, 1e2],
        'epsilon' : [0.0001, 0.001, 0.01, 0.1]
    },
    {
        'max_degree' : [3, 4, 5],
        'penalty' : [1.0, 3.0, 9.0]
    }
]

##Perform a benchmark with optimization and ShuffleSplit CV
pipe.benchmark(model_list, param_grid_list, optim=True, cv='ShuffleSplit', n_restart=2)

##Plot the results for each model best candidate
for model in model_list:
    pipe.plot(model)