import math
import numpy as np

class AverageValueMerter(object):
    def __init__(self):
        super(AverageValueMerter, self).__init__()
        self.reset()
        self.val = 0

    def add(self,value,n=1):
        self.val=value
        self.sum+=value
        self.var+=value*value
        self.n+=n

        if self.n==0:
            self.mean,self.std=np.nan,np.nan

        elif self.n==1:
            self.mean,self.std=self.sum,np.inf
        else:
            self.mean=self.sum/self.n

    def value(self):
        return self.mean
    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
