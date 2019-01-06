import numpy as np
from CustomDistances import sq_distance, hv_distance
    
    
class LearningModel:
    
    def __init__(self):
        pass
        
    def train(self, X, Y, eval_score=False):
        pass
        
    def predict(self, X_out, Y_out=None, eval_score=False):
        pass
    
    def _score(self, Y, prediction):
        assert Y.shape == prediction.shape
        E = (Y - prediction)**2
        score = np.sum(E)/E.size
        return score
    

class NearestNeighbor(LearningModel):
    
    def __init__(self):
        pass
    
    def train(self, X, Y, eval_score=False):
        assert X.shape[0] == Y.shape[0]
        self.X_in = X
        self.Y_in = Y
        if eval_score:
            prediction, score = self.predict(self.X_in, self.Y_in, eval_score=True)
            return score
        
    def predict(self, X_out, Y_out=None, eval_score=False):
        nsample = self.X_in.shape[0]
        npreds = X_out.shape[0]
        prediction = np.empty([npreds,1])
    
        ##Vectorized operation is slower than for-loop
        #distance = hv_distance(X_out[:,0,None], X_out[:,1,None], self.X_in[:,0], self.X_in[:,1])
        #print(distance)
        #prediction = self.Y_in[np.argmin(distance, axis=1)]
    
        for k in range(0, npreds):
            distance = hv_distance(X_out[k][0], X_out[k][1], self.X_in[:,0], self.X_in[:,1])
            prediction[k] = self.Y_in[np.argmin(distance)]
        
        if eval_score:
            if Y_out is None:
                raise ValueError("Need Y_out (Y_test) to evaluate score")
            else:
                score = self._score(Y_out, prediction)
                return prediction, score
        else:
            return prediction
    