# -*- coding: utf-8 -*-
"""
Example script for running a list of models and plotting the results

@author: Tâm Le Minh
"""
from src.meteodr import MeteoPipeline
import src.models as CustomModels


##Prepare the data and the pipeline
datafile = "Temp-2019_01_04-15_47.csv"
pipe = MeteoPipeline(data_file=datafile, verbose=True, report=True)
pipe.partition_train_test(test_size=0.2)

##Prepare the model list
model_list = [
    CustomModels.NearestNeighbor(),
    CustomModels.InverseDistanceWeighting(),
    CustomModels.RandomForest(),
]

##Define the parameters associated with each model
param_grid_list = [
    {
        
    },
    {
        'radius' : 10
    },
    {
        'n_estimators' : 1000,
        'max_depth' : 10
    }
]

##Perform a benchmark without optimization
pipe.benchmark(model_list, param_grid_list, optim=False, cv=None, n_restart=5)

##Plot the results for all the models
for model in model_list:
    pipe.plot(model)