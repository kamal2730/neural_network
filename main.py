import numpy as np
import nnfs
nnfs.init()
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        pass
    def forward(self,inputs):
        pass

X,y=spiral_data(samples=100,classes=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()