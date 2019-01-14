# -*- coding: utf-8 -*-
"""
Example script for loading a list of models and parameters from 
a csv file and perform a benchmark

@author: Tâm Le Minh
"""
from src.meteodr import MeteoPipeline


##Prepare the data and the pipeline
datafile = "Temp-2019_01_04-15_47.csv"
pipe = MeteoPipeline(data_file=datafile, verbose=True, report=True)
pipe.partition_train_test(test_size=0.2)

##Import the models, initialized with the parameters from the csv
mlist = pipe.load_models("reports/2019_01_10-22_03_14/Benchmark.csv")

##Perform a benchmark (def: no optimization, verbose = True)
pipe.benchmark(mlist)

##Plot the results for each model
for model in mlist:
    pipe.plot(model)