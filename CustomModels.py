import numpy as np
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
    
    
class CustomModel:
    
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        pass
    
    def predict(self, X):
        pass
    
    
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
    
    
class LearningFramework:
    
    def __init__(self):
        pass
        
    def train(self, X, Y, eval_score=False):
        self.model.fit(X, np.ravel(Y))
        if eval_score:
            prediction, score = self.predict(X, Y, eval_score=True)
            return score
        
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
    
    def optimize(self, X, Y, scoring='neg_mean_squared_error', cv=5, **param_grid):
        scores = np.empty([len(list(ParameterGrid(param_grid))), 1])
        n_splits = 10
        k1 = 0
        mean_score_candidate = np.empty([n_splits,1])
        for candidate in list(ParameterGrid(param_grid)):
            k2 = 0
            self.model.set_params(**candidate)
            ss = ShuffleSplit(n_splits=n_splits)
            ss.get_n_splits(X, Y)
            for train_index, test_index in ss.split(X, Y):
                xtrain, xtest = X[train_index], X[test_index]
                ytrain, ytest = Y[train_index], Y[test_index]
                self.train(xtrain, ytrain)
                pred, score = self.predict(xtest, ytest, eval_score=True)
                mean_score_candidate[k2] = score
                k2 += 1
            scores[k1] = np.mean(mean_score_candidate)
            k1 += 1
        best_candidate = list(ParameterGrid(param_grid))[np.argmin(scores)]
        score_candidate = np.min(scores)
        return best_candidate, score_candidate, list(ParameterGrid(param_grid)), scores
    
    def _score(self, Y, prediction):
        return mean_squared_error(Y, prediction)
    
    def set_params(self, **params):
        self.model.set_params(**params)
        

class NearestNeighbor(LearningFramework):
    
    def __init__(self):
        self.model = NearestNeighborRegressor()

        
class InverseDistanceWeighting(LearningFramework):
    
    def __init__(self, **param):
        self.model = InverseDistanceWeightingRegressor(**param)
        
    def optimize(self, X, Y, scoring='neg_mean_squared_error', cv=5, **param_grid):
        scores = np.empty([len(list(ParameterGrid(param_grid))), 1])
        n_splits = 10
        k1 = 0
        mean_score_candidate = np.empty([n_splits,1])
        for candidate in list(ParameterGrid(param_grid)):
            k2 = 0
            self.model.set_params(**candidate)
            ss = ShuffleSplit(n_splits=n_splits)
            ss.get_n_splits(X, Y)
            for train_index, test_index in ss.split(X, Y):
                xtrain, xtest = X[train_index], X[test_index]
                ytrain, ytest = Y[train_index], Y[test_index]
                self.train(xtrain, ytrain)
                pred, score = self.predict(xtest, ytest, eval_score=True)
                mean_score_candidate[k2] = score
                k2 += 1
            scores[k1] = np.mean(mean_score_candidate)
            k1 += 1
        best_candidate = list(ParameterGrid(param_grid))[np.argmin(scores)]
        score_candidate = np.min(scores)
        return best_candidate, score_candidate, list(ParameterGrid(param_grid)), scores
    
    def set_params(self, **params):
        self.model.set_params(**params)
        
        
class GaussianProcess(LearningFramework):
    
    kernel = C(1.0)*RBF(length_scale=[10.0, 10.0]) + WhiteKernel(0.1)
    
    def __init__(self):
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha = 0, optimizer=None, n_restarts_optimizer=10, 
                                       normalize_y=True, random_state=0)
        #self.model = GaussianProcessRegressor(kernel=kernel, alpha = 0, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10, 
        #                               normalize_y=True, random_state=0).fit(X, Y)


class GeographicallyWeightedRegressor(LearningFramework):
    
    kernel = C(1.0)*CustomKernels.RBF(length_scale=100.0, metric='haversine') + WhiteKernel(0.1)
    
    def __init__(self):
        #self.model = GaussianProcessRegressor(kernel=self.kernel, alpha = 0, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10, 
        #                               normalize_y=True, random_state=0).fit(X, Y)
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha = 0, optimizer=None, n_restarts_optimizer=10, 
                                       normalize_y=True, random_state=0)
    

class RegressionTree(LearningFramework):
    
    max_depth = 9
    
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=self.max_depth)


class RandomForest(LearningFramework):
    
    param = {
        'n_estimators' : 1000,
        'max_features' : 'auto',
        'max_depth' : 10,
        'random_state' : 42
    }
    hyper = [
        'max_depth'
    ]
    
    n_estimators = 1000
    max_depth = 10
    random_state = 42
    
    def __init__(self):
        self.model = RandomForestRegressor(**self.param)
        #self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_features='auto', 
        #                            max_depth=self.max_depth, random_state=self.random_state) 


class ExtraTrees(LearningFramework):
    
    max_ntree = 1000
    max_depth = 10
    random_state = 42
    
    def __init__(self):
        self.model = ExtraTreesRegressor(n_estimators=self.max_ntree, max_features='auto', 
                                    max_depth=self.max_depth, random_state=self.random_state)
    

class SupportVectorRegression(LearningFramework):
    
    C = 10.0
    epsilon = 5.0
    
    def __init__(self):
        self.model = SVR(gamma='auto', C=self.C, epsilon=self.epsilon)

        
class RegressionSplines(LearningFramework):
    
    max_degree = 3
    penalty = 3.0
    
    def __init__(self):
        self.model = Earth(max_degree = self.max_degree, penalty = self.penalty)
            
