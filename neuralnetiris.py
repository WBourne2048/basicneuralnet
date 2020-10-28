"""
Implementation of a 1 hidden layer NN to classify the iris dataset
Note: there is no test/train split in this code so the classification accuracy 
will be very high as a result of overtraining if the number of iterations is increased 
"""
import sklearn 
import numpy as np  
from sklearn import datasets
np.random.seed(0)

#load the data and labels from sklearn
iris = datasets.load_iris()
X = iris.data
#generate  target labels for the iris dataset 
a = np.array([(1,0,0)]*50)
b = np.array([(0,1,0)]*50)
c = np.array([(0,0,1)]*50)
Y = np.concatenate((a,b,c), axis = 0)

#define the shape of our neural network

inputneurons = 4 #iris has 4 input variables
hiddenneurons = 10 #hyperparameter, this can be altered 
outputneurons = 3 # we have 3 classes to output towards
firstweights = np.random.randn(inputneurons,hiddenneurons) / 10 #random 4x10 array of weights for our network 
secondweights = np.random.randn(hiddenneurons,outputneurons) / 10
alpha = 0.01 #learning rate
iters = 10000 #iterations

#implement our main functions for the NN including activity. forward pass and backpropogation 

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoidprime(x):
	return x * (1-x)

#forwardpass will input data into our input layer and then produce an output activity for the activity layer
def forwardpass(X):
	hiddenrawactivity = np.dot(X,firstweights)
	hiddenactivity = sigmoid(hiddenrawactivity)
	outputrawactivity = np.dot(hiddenactivity,secondweights)
	outputactivity = sigmoid(outputrawactivity)
	return outputactivity

#given our forward pass, we want to change the weights based on the gradient descent method
def backpropogation(X,Y,secondweights,firstweights):
	outputerror = Y - forwardpass(X)
	outputdelta = outputerror * sigmoidprime(forwardpass(X))

	hiddenerror = np.dot(outputdelta , secondweights.T)
	hiddendelta = hiddenerror * sigmoidprime(sigmoid(np.dot(X,firstweights)))

	firstweights += alpha * np.dot(X.T,hiddendelta)
	secondweights += alpha * np.dot(sigmoid(np.dot(X,firstweights).T),outputdelta)

def train(X,Y):
	output = forwardpass(X)
	backpropogation(X,Y,secondweights,firstweights)

for i in range(1,iters):
	train(X,Y)

def predict(X):
	predictedclass = []
	for i in range(0,len(forwardpass(X))):
			maxpos = np.where(forwardpass(X)[i] == np.amax(forwardpass(X)[i]))
			predictedclass.append(maxpos)
	return np.array(predictedclass, dtype = int).T

#print labels for the predicted classes and the original labels
print(predict(X))
print(iris.target)

a = np.reshape(predict(X),(1,150))

def accuracy(X):
	count = 0
	for i in range(0,len(X)):
		if int(a[0][i]) == int(iris.target[i]):
			count +=1
	return count /150

#final accuracy of the model 
print(accuracy(X))