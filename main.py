import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(np.int64)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

def get_batches(X, y, batch_size):
    # Shuffle data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Yield batches
    for i in range(0, X.shape[0], batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        yield X_batch, y_batch

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons,weight_regularizer_l1=0,weight_regularizer_l2=0,bias_regularizer_l1=0,bias_regularizer_l2=0):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
        self.weight_regularizer_l1=weight_regularizer_l1
        self.weight_regularizer_l2=weight_regularizer_l2
        self.bias_regularizer_l1=bias_regularizer_l1
        self.bias_regularizer_l2=bias_regularizer_l2
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.dot(inputs,self.weights)+self.biases
    def backward(self,dvalues):
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dbiases=np.sum(dvalues,axis=0,keepdims=True)
        #regularization
        if self.weight_regularizer_l1>0:
            dL1=np.ones_like(self.weights)
            dL1[self.weights<1]=-1
            self.dweights+=self.weight_regularizer_l1*dL1
        if self.weight_regularizer_l2>0:
            self.dweights+=2*self.weight_regularizer_l2*self.weights
        if self.bias_regularizer_l1>0:
            dL1=np.ones_like(self.biases)
            dL1[self.biases<1]=-1
            self.dbiases+=self.bias_regularizer_l1*dL1
        if self.bias_regularizer_l2>0:
            self.dbiases+=2*self.bias_regularizer_l2*self.biases
        self.dinputs=np.dot(dvalues,self.weights.T)
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs<=0]=0
class Activation_Softmax:
    def forward(self,inputs):
        self.inputs=inputs
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probabilities
    def backward(self,dvalues):
        self.dinputs=np.empty_like(dvalues)
        for index,(single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output=single_output.reshape(-1,1)
            jacobian_matrix=np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index]=np.dot(jacobian_matrix,single_dvalues)
class Loss:
    def regularization_loss(self,layer):
        regularization_loss=0
        if layer.weight_regularizer_l1>0:
            regularization_loss+=layer.weight_regularizer_l1*np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2>0:
            regularization_loss+=layer.weight_regularizer_l2*np.sum(layer.weights*layer.weights)
        if layer.bias_regularizer_l1>0:
            regularization_loss+=layer.bias_regularizer_l1*np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2>0:
            regularization_loss+=layer.bias_regularizer_l2*np.sum(layer.biases*layer.biases)
        return regularization_loss
    
    def calculate(self,output,y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            corrected_confidences=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            corrected_confidences=np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihoods=-np.log(corrected_confidences)
        return negative_log_likelihoods
    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        labels=len(dvalues[0])
        if len(y_true.shape)==1:
            y_true=np.eye(labels)[y_true]
        self.dinputs=-y_true/dvalues
        self.dinputs=self.dinputs/samples
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=Loss_CategoricalCrossentropy()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output=self.activation.output
        return self.loss.calculate(self.output,y_true)
    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs=self.dinputs/samples

class Optimizer_Adam:
    def __init__(self,learning_rate=0.001,decay=0.,epsilon=1e-7,beta_1=0.9,beta_2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iterations))
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_momentums=np.zeros_like(layer.weights)
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_momentums=np.zeros_like(layer.biases)
            layer.bias_cache=np.zeros_like(layer.biases)


        layer.weight_momentums=self.beta_1*layer.weight_momentums+(1-self.beta_1)*layer.dweights
        layer.bias_momentums=self.beta_1*layer.bias_momentums+(1-self.beta_1)*layer.dbiases

        weight_momentums_corrected=layer.weight_momentums/(1-self.beta_1**(self.iterations+1))
        bias_momentums_corrected=layer.bias_momentums/(1-self.beta_1**(self.iterations+1))

        layer.weight_cache=self.beta_2*layer.weight_cache+(1-self.beta_2)*layer.dweights**2
        layer.bias_cache=self.beta_2*layer.bias_cache+(1-self.beta_2)*layer.dbiases**2

        weight_cache_corrected=layer.weight_cache/(1-self.beta_2**(self.iterations+1))
        bias_cache_corrected=layer.bias_cache/(1-self.beta_2**(self.iterations+1) )

        layer.weights+=-self.current_learning_rate*weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.biases+=-self.current_learning_rate*bias_momentums_corrected/(np.sqrt(bias_cache_corrected)+self.epsilon)
    def post_update_params(self):
        self.iterations+=1
class Layer_Droput:
    def __init__(self,rate):
        self.rate=1-rate
    def forward(self,inputs):
        self.inputs=inputs
        self.binary_mask=np.random.binomial(1,self.rate,size=inputs.shape)/self.rate
        self.output=inputs*self.binary_mask
    def backward(self,dvalues):
        self.dinputs=dvalues*self.binary_mask


dense1=Layer_Dense(784,512,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
activation1=Activation_ReLU()
dropout1=Layer_Droput(0.1)
dense2=Layer_Dense(512,10)
loss_activation=Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer=Optimizer_Adam(learning_rate=0.05,decay=5e-5)
for epoch in range(30):
    for X_batch, y_batch in get_batches(X_train, y_train, 32):
        dense1.forward(X_batch)
        activation1.forward(dense1.output)
        dropout1.forward(activation1.output)
        dense2.forward(dropout1.output)
        data_loss=loss_activation.forward(dense2.output,y_batch)
        regularization_loss=loss_activation.loss.regularization_loss(dense1)+loss_activation.loss.regularization_loss(dense2)
        loss=data_loss+regularization_loss

        predictions=np.argmax(loss_activation.output,axis=1)
        if len(y_batch.shape)==2:
            y_batch=np.argmax(y_batch,axis=1)
        accuracy=np.mean(predictions==y_batch)
        #backward pass
        loss_activation.backward(loss_activation.output,y_batch)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    #testing 
    print(f'epoch:{epoch},acc:{accuracy:.3f},loss:{loss:.3f},data_loss:{data_loss},reg_loss:{regularization_loss},lr:{optimizer.current_learning_rate}')
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss=loss_activation.forward(dense2.output,y_test)

    predictions=np.argmax(loss_activation.output,axis=1)
    if len(y_test.shape)==2:
        y_test=np.argmax(y_test,axis=1)
    accuracy=np.mean(predictions==y_test)

    print('validation accuracy:',accuracy)