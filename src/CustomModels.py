# -*- coding: utf-8 -*-
"""
Custom models based on sklearn models
Custom learning framework as a wrapper for sklearn models and custom models
List of models and example of parameters:
    NearestNeighbor - None
    InverseDistanceWeighting - Radius
    GaussianProcess - Kernel, alpha 
    GeographicallyWeightedRegressor - Kernel, alpha
    RegressionTree - Max_depth
    RandomForest - N_estimators, Max_features, Max_depth
    ExtraTrees - N_estimators, Max_features, Max_depth
    SupportVectorRegression - Gamma, Epsilon, C
    RegressionSplines - Max_degree, Penalty

@author: Tâm Le Minh
"""
import numpy as np
import pandas as pd
from CustomDistances import sq_distance, hv_distance
import CustomKernels
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, RationalQuadratic, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid, ShuffleSplit
from pyearth import Earth
            
            
##Helper class to check the CV argument in optimization functions
def checkCV(cv='warn', n_split=3):
        
    if cv is 'KFold':
        return KFold(n_split)
    elif cv is 'ShuffleSplit':
        return ShuffleSplit(n_split)
    else:
        raise ValueError("CV technique must be 'KFold' or 'ShuffleSplit'")
    
    
##Base class for sklearn's style custom models
class CustomModel:
    
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        pass
    
    def predict(self, X):
        pass
    

##NNI implementation as a custom model
class NearestNeighborRegressor(CustomModel):
    
    params = {}
    
    def __init__(self, **params):
        pass
    
    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        self.X_in = X
        self.Y_in = Y
        
    def predict(self, X_out):
        nsample = self.X_in.shape[0]
        npreds = X_out.shape[0]
        prediction = np.empty([npreds,1])
        
        for k in range(0, npreds):
            distance = hv_distance(X_out[k][0], X_out[k][1], self.X_in[:,0], self.X_in[:,1])
            prediction[k] = self.Y_in[np.argmin(distance)]
            
        ##Vectorized operation is slower than for-loop
        #distance = hv_distance(X_out[:,0,None], X_out[:,1,None], self.X_in[:,0], self.X_in[:,1])
        #print(distance)
        #prediction = self.Y_in[np.argmin(distance, axis=1)]
            
        return prediction
    
    def get_params(self):
        return self.params
    
    def set_params(self, **params):
        self.params = params

    
##IDW implementation as a custom model
class InverseDistanceWeightingRegressor(CustomModel): 
    
    params = {
        'radius' : 500
    }
    
    def __init__(self, **params):
        self.params['radius'] = params.pop('radius', 500)
        
    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        self.X_in = X
        self.Y_in = Y
        
    def predict(self, X_out):
        nsample = self.X_in.shape[0]
        npreds = X_out.shape[0]
        prediction = np.empty([npreds,1])
        
        for k in range(0, npreds):
            distance = hv_distance(X_out[k][0], X_out[k][1], self.X_in[:,0], self.X_in[:,1])
            idx0 = np.where(distance < 1)
            idx_in_radius = np.where(distance < self.params['radius'])
            if idx_in_radius[0].size == 0 or idx0[0].size > 0:
                prediction[k] = self.Y_in[np.argmin(distance)]
            else:
                numerator = np.sum(self.Y_in[:,None][idx_in_radius]/distance[idx_in_radius])
                denominator = np.sum(1/distance[idx_in_radius])
                prediction[k] = numerator/denominator
                
        return prediction
        
    def get_params(self):
        return self.params
    
    def set_params(self, **params):
        self.params = params
    
    
##Wrapper class to use sklearn models and custom models with Meteo-DR pipeline
class LearningFramework:
    
    def_params = {}
    current_params = {}
    
    def __init__(self, **params):
        pass
        
    ##Calculates an loss function score
    # Only MSE available currently
    def _score(self, Y, prediction):
        return -mean_squared_error(Y, prediction)
    
    ##Train the model, can return training score
    def train(self, X, Y, eval_score=False):
        self.model.fit(X, np.ravel(Y))
        if eval_score:
            prediction, score = self.predict(X, Y, eval_score=True)
            return score
        
    ##Predict with the model, can return test score
    def predict(self, X_out, Y_out=None, eval_score=False):
        prediction = self.model.predict(X_out)
        if eval_score:
            if Y_out is None:
                raise ValueError("Need Y_out (Y_test) to evaluate score")
            else:
                score = self._score(Y_out, prediction)
                return prediction, score
        else:
            return prediction
    
    ##Optimize the model, returns best candidate parameters, best score and dataframe containing 
    # the score for each parameter tested
    def optimize(self, X, Y, scoring='neg_mean_squared_error', cv='warn', n_splits=5, info=True, **params):
        
        param_grid = list(ParameterGrid(params))
        mean_scores = np.empty([len(param_grid), 1])
        splitter = checkCV(cv, n_splits)
        scores_candidate = np.empty([n_splits,1])
        if info:
            info_out = pd.DataFrame(columns = ['Model', 'Parameters', 'Mean CV Score'])
        
        for i in range(0, len(param_grid)):
            self.model.set_params(**param_grid[i])
            splitter.get_n_splits(X, Y)
            k = 0
            
            for train_index, test_index in splitter.split(X, Y):
                xtrain, xtest = X[train_index], X[test_index]
                ytrain, ytest = Y[train_index], Y[test_index]
                self.train(xtrain, ytrain)
                pred, scores_candidate[k] = self.predict(xtest, ytest, eval_score=True)
                k += 1
                
            mean_scores[i] = np.mean(scores_candidate)
            if info:
                info_out.loc[i] = [self.get_name(), param_grid[i], np.ravel(mean_scores)[i]]
            
        best_candidate = param_grid[np.argmax(mean_scores)]
        best_score = np.max(mean_scores)
        if info:
            return best_candidate, best_score, info_out
        else:
            return best_candidate, best_score
        
    
    def get_params(self):
        return self.model.get_params()
        
    def set_params(self, **params):
        self.model.set_params(**params)
        
    def get_name(self):
        return self.__class__.__name__
        
    def to_string(self):
        return self.__class__.__name__, str(self.get_params())
        

##NNI implementation of LearningFramework
class NearestNeighbor(LearningFramework):
    
    def_params = {}
    
    def __init__(self, **params):
        if params == {}:
            self.model = NearestNeighborRegressor(**self.def_params)
        else:
            self.model = NearestNeighborRegressor(**params)

            
##IDW implementation of LearningFramework   
class InverseDistanceWeighting(LearningFramework):
        
    def_params = {
        'radius': 100
    }
    
    def __init__(self, **params):
        if params == {}:
            self.model = InverseDistanceWeightingRegressor(**self.def_params)
        else:
            self.model = InverseDistanceWeightingRegressor(**params)
        

##Gaussian Process implementation of LearningFramework   
class GaussianProcess(LearningFramework):
         
    def_params = {
        'kernel' : C(1.0)*RBF(length_scale=[10.0, 10.0]) + WhiteKernel(0.1),
        'alpha' : 0,
        'optimizer' : None,
        'n_restarts_optimizer' : 0,
        'normalize_y' : True
    }
    
    def __init__(self, **params):
        if params == {}:
            self.model = GaussianProcessRegressor(**self.def_params)
        else:
            self.model = GaussianProcessRegressor(**params)


##GWR implementation of LearningFramework  
class GeographicallyWeightedRegressor(LearningFramework):
                
    def_params = {
        'kernel' : C(1.0)*CustomKernels.RBF(length_scale=100.0, metric='haversine') + WhiteKernel(0.1),
        'alpha' : 0,
        'optimizer' : None,
        'n_restarts_optimizer' : 0,
        'normalize_y' : True
    }
    
    def __init__(self, **params):
        if params == {}:
            self.model = GaussianProcessRegressor(**self.def_params)
        else:
            self.model = GaussianProcessRegressor(**params)
            

##Regression Tree implementation of LearningFramework  
class RegressionTree(LearningFramework):
                 
    def_params = {
        'max_depth' : 9
    }
    
    def __init__(self, **params):
        if params == {}:
            self.model = DecisionTreeRegressor(**self.def_params)
        else:
            self.model = DecisionTreeRegressor(**params)
            

##Random Forest implementation of LearningFramework  
class RandomForest(LearningFramework):
                     
    def_params = {
        'n_estimators' : 1000,
        'max_features' : 'auto',
        'max_depth' : 10
    }
    
    def __init__(self, **params):
        if params == {}:
            self.model = RandomForestRegressor(**self.def_params)
        else:
            self.model = RandomForestRegressor(**params)
    

##Extra Trees implementation of LearningFramework  
class ExtraTrees(LearningFramework):
                         
    def_params = {
        'n_estimators' : 1000,
        'max_features' : 'auto',
        'max_depth' : 10
    }
    
    def __init__(self, **params):
        if params == {}:
            self.model = ExtraTreesRegressor(**self.def_params)
        else:
            self.model = ExtraTreesRegressor(**params)


##SVR implementation of LearningFramework  
class SupportVectorRegression(LearningFramework):
                             
    def_params = {
        'gamma': 'auto',
        'C' : 10.0,
        'epsilon' : 5.0
    }
    
    def __init__(self, **params):
        if params == {}:
            self.model = SVR(**self.def_params)
        else:
            self.model = SVR(**params)
        

##Polynomial Splines Regression implementation of LearningFramework  
class RegressionSplines(LearningFramework):
                                
    def_params = {
        'max_degree' : 3,
        'penalty' : 3.0
    }
    
    def __init__(self, **params):
        if params == {}:
            self.model = Earth(**self.def_params)
        else:
            self.model = Earth(**params)
