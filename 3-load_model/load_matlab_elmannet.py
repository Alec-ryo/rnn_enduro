from scipy.io import loadmat
import numpy as np 

def sigmoid(X):
    return 1/(1+np.exp(-X))

class LoadElman():
    def __init__(self, model_path):
        data = loadmat(model_path, matlab_compatible=False, struct_as_record=False)
        net = data['net'][0][0]
        self.net = net
        
        self.U  = net.IW[0][0]
        self.W  = net.LW[0][0]
        self.V  = net.LW[1][0]
        self.bh = net.b[0][0].reshape(len(net.b[0][0]),)
        self.bo = net.b[1][0].reshape(len(net.b[1][0]),)
        
        self.prev_s = np.zeros(self.bh.shape) 

        print("U shape:", self.U.shape)
        print("W shape:", self.W.shape)
        print("V shape:", self.V.shape)
        print("bh shape:", self.bh.shape)
        print("bo shape:", self.bo.shape)
    
    def reset_prev_s(self):
        self.prev_s = np.zeros(self.bh.shape)
    
    def predict_step(self, X_train_t):
        # forward pass
        new_input = X_train_t
        mulu = np.dot(self.U, new_input)
        mulw = np.dot(self.W, self.prev_s)
        add = mulw + mulu + self.bh 
        s = sigmoid(add)
        mulv = np.dot(self.V, s) + self.bo
        out = sigmoid(mulv)
        self.prev_s = s

        return out