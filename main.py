import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return np.exp(x)/np.sum(np.exp(x),axis=1, keepdims=True)


df = pd.read_csv("adult.data.txt", names=["age","workclass","fnlwgt","education","education-num","marital-status"
    ,"occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","class"])
dx = pd.read_csv("adult.test.txt", names=["age","workclass","fnlwgt","education","education-num","marital-status"
    ,"occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","class"])

for lf in df:
    if df[lf].dtype == "object":
        df[lf] = df[lf].astype("category").cat.codes
        dx[lf] = dx[lf].astype("category").cat.codes
    else :
        df[lf] = (df[lf] - df[lf].mean())/(df[lf].max() - df[lf].min())
        dx[lf] = (dx[lf] - dx[lf].mean()) / (dx[lf].max() - dx[lf].min())

x = df.drop(columns=["class"])
y = df["class"].values
x_test = dx.drop(columns=["class"])
y_test = dx["class"].values

multi_y = np.zeros((y.size, y.max()+1))
multi_y[np.arange(y.size), y] = 1
multi_y_test = np.zeros((y_test.size, y_test.max()+1))
multi_y_test[np.arange(y_test.size), y_test] = 1

inputSize = len(x.columns)
numberOfNodes = 150
numberOfClass = y.max() + 1
numberOfExamples = x.shape[0]

w1 = np.random.random_sample(size=(inputSize, numberOfNodes))
b1 = np.random.random_sample(numberOfNodes)
w2 = np.random.random_sample(size=(numberOfNodes, numberOfClass))
b2 = np.random.random_sample(numberOfClass)

batchSize = 32
trainNum = 150
learningRate = 0.01

# Start Training
for k in range(trainNum + 1):
    cost = 0
    accuracy = 0
    for i in range(int(numberOfExamples/batchSize)):
        # Forward-Propagation
        z = x[i * batchSize : (i+1) * batchSize]
        z_y = multi_y[i * batchSize : (i+1) * batchSize]
        layer1 = np.matmul(z, w1) + b1
        sig_layer1 = sigmoid(layer1)
        layer2 = np.matmul(sig_layer1, w2) + b2
        soft_layer2 = softmax(layer2)
        pred = np.argmax(soft_layer2, axis=1)
        # Cost Function: Cross-Entropy loss
        cost += -(z_y * np.log(soft_layer2 + 1e-9) + (1-z_y) * np.log(1 - soft_layer2 + 1e-9)).sum()
        accuracy += (pred == y[i * batchSize : (i + 1) * batchSize]).sum()

        # Back-Propagation
        dlayer2 = soft_layer2 - multi_y[i * batchSize : (i+1) * batchSize]
        dw2 = np.matmul(sig_layer1.T, dlayer2) / batchSize
        db2 = dlayer2.mean(axis=0)
        dsig_layer1 = (dlayer2.dot(w2.T))
        dlayer1 = sigmoid(layer1) * (1 - sigmoid(layer1)) * dsig_layer1
        dw1 = np.matmul(z.T, dlayer1) / batchSize
        db1 = dlayer1.mean(axis=0)

        w2 -= learningRate * dw2
        w1 -= learningRate * dw1
        b2 -= learningRate * db2
        b1 -= learningRate * db1
    if k % 10 == 0 :
        print("-------- # : {} ---------".format(k))
        print("cost: {}".format(cost/numberOfExamples))
        print("accuracy: {} %".format(accuracy/numberOfExamples * 100))

# Test the trained model
test_cost = 0
test_accuracy = 0
# Forward-Propagation
layer1 = np.matmul(x_test, w1) + b1
sig_layer1 = sigmoid(layer1)
layer2 = np.matmul(sig_layer1, w2) + b2
soft_layer2 = softmax(layer2)
pred = np.argmax(soft_layer2, axis=1)
# Cost Function: Cross-Entropy loss
test_cost += -(multi_y_test * np.log(soft_layer2 + 1e-9) + (1-multi_y_test) * np.log(1 - soft_layer2 + 1e-9)).sum()
test_accuracy += (pred == y_test).sum()

print("---- Result of applying test data to the trained model")
print("cost: {}".format(test_cost/numberOfExamples))
print("accuracy: {} %".format(test_accuracy/numberOfExamples * 100))