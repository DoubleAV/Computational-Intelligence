import numpy as np
import sys

# Python Numpy Versions
# print(sys.version)
# print("Numpy Version: ", np.__version__)


# ============
# Read In Data
# ============
# Hidden Layer Biases (11)
b1 = np.loadtxt(open("Data/b1 (11 nodes).csv", "rb"), delimiter=",")
# Output node biases (2)
b2 = np.loadtxt(open("Data/b2 (2 output nodes).csv", "rb"), delimiter=",")
# Cross Data - Three inputs, Two outputs
init_training_data = np.loadtxt(open("Data/cross_data (3 inputs - 2 outputs).csv", "rb"), delimiter=",")
data = init_training_data[:, :3] # First three columns of data
desired_outputs = init_training_data[:, [3,4]] # Last two columns of data
# Input to Hidden Layer Weights
w1 = np.loadtxt(open("Data/w1 (3 inputs - 11 nodes).csv", "rb"), delimiter=",")
# Hidden Layer to Output layer weights
w2 = np.loadtxt(open("Data/w2 (from 11 to 2).csv", "rb"), delimiter=",")

# Learning Rate
N = 0.7
# Momentum Term
M = 0.3


class Neural_Network(object):
    def __init__(self):
        # Params
        self.inputSize = 3
        self.hiddenSize = 11
        self.outputSize = 2
        # Biases
        self.b1 = b1 # 11x1
        self.b2 = b2 # 2x1
        # Weights
        self.w1 = w1 # 11x3
        self.w2 = w2 # 2x11

        # Others
        self.learning_rate = N
        self.first_pass = True
        if self.first_pass:
            self.momentum = 0
        else:
            self.momentum = M
        

    '''
    Sigmoid function of any value or list of values
    handles derivative with flag.
    '''
    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    '''
    Handles forward pass
    '''
    def forward_pass(self, x):
        self.z = np.dot(x, self.w1.T) + self.b1 # Dot product of X (inputs) and first set of 11rowsx3cols weights plus biases
        self.z2 = self.sigmoid(self.z) # Activation function
        self.z3 = np.dot(self.z2, self.w2.T) + self.b2 # Dot product of hidden layer (z2) and second set of 2x11 weights + biases
        o = self.sigmoid(self.z3) # Final Activation Function
        return o

    '''
    Handles backward Pass
    x = data, y = desired outputs, o = forward pass outputs
    '''
    def backward_pass(self, x, desired_o, output):
        self.o_error = desired_o - output # error in output
        self.o_delta = self.o_error * self.sigmoid(output, True) # applying derivative of sigmoid to error
        
        self.z2_error = self.o_delta.dot(self.w2)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, True) # applying derivative of sigmoid to z2 error

        self.w1 += self.learning_rate * x.T.dot(self.z2_delta).T # adjusting first set (input --> hidden) weights
        self.w2 += self.learning_rate * self.z2.T.dot(self.o_delta).T # adjusting second set (hidden --> output) weights

        self.b2 += self.learning_rate * self.o_delta.T * 1 # Corresponding input for a bias is always 1'
        self.b1 += self.learning_rate * self.z2_delta * 1 # Update biases

        self.first_pass = False

        

    '''
    Training Function
    '''
    def train(self, x, desired_outputs):
        output = self.forward_pass(x)
        self.backward_pass(x, desired_outputs, output)
    
NN = Neural_Network()

sse = np.sum(np.square(desired_outputs - NN.forward_pass(data))).round(4) # Sum of Squared errors
mse = np.mean(np.square(desired_outputs - NN.forward_pass(data))).round(4) # Mean of squared errors
sse_tot = np.square(desired_outputs - NN.forward_pass(data)).round(4) # SSE for every sample
NN.train(data, desired_outputs)
updated_w1 = NN.w1.round(4)
updated_w2 = NN.w2.round(4)
