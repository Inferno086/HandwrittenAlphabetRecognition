import numpy as np
import pandas as pd

df = pd.read_csv (r'C:\Users\ckshu\Desktop\Python\archive\A_Z Handwritten Data.csv')

data = df.to_numpy()
np.random.shuffle(data)
np.random.shuffle(data)
np.random.shuffle(data)

X = data[:,1:]
#print(X)
#X.shape

y = data[:,0]
y = y.reshape(X.shape[0],1)
#print(y)
#y.shape

X_bias = np.ones((X.shape[0],1))
X = np.concatenate((X_bias,X),1)
#X.shape
#print(X) #Make sure to restart the entire thing again. 

X_train = X[0:100000,:]
y_train = y[0:100000,:]
X_test = X[300000:,:]
y_test = y[300000:,:]
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

m = X_train.shape[0]
K = 26
Y1 = np.arange(K)
Y = np.tile(Y1,(m,1))
for i in range(m):
    Y[i,:] = np.int32(Y[i,:] == y[i])

Y_train = Y[0:100000,:]
Y_test = Y[300000:,:]

"""
We have a data set with 300,000 stimuli or training examples.
Our target is to make a neural network which classifies a stimuli as a letter between A-Z.
The input layer has 784 features without bias unit. In total 785 features.

Our network will have 4 layers. 1 input, 2 hidden and 1 output layer.
Input = 784 units
Hidden layer 1 = 40 activation units
Hidden layer 2 = 40 activation units
Output layer = 26 units

Theta1 has dimentions (40 x 785)
Theta2 has dimentions (40 x 41)
Theta3 has dimentions (26 x 41)
"""

def sigmoid(K):
    """This returns the sigmoid of every element of a matrix"""
    sig = 1/(1 + np.exp(K))
    return sig

Theta1 = np.random.rand(40,785)
Theta2 = np.random.rand(40,41)
Theta3 = np.random.rand(26,41)
Lambda = 1
alpha = 50000
iterations = 150


def computeCost(X,y,theta1,theta2,theta3,Lambda):
    """This function computes the cost for a given theta on the dataset"""
    K = theta3.shape[0]
    m = y.shape[0]
    
    # Y1 = np.arange(K)
    # Y = np.tile(Y1,(m,1))
    # for i in range(m):
    #     Y[i,:] = np.int32(Y[i,:] == y[i])
        
    # Forward propogation to find final output from input
    a1 = X  # a1 has input units for 300,000 stimuli. a1.shape = (300000,785)
    z2 = np.matmul(a1,theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((z2.shape[0],1)),a2),1)
    z3 = np.matmul(a2,theta2.T)
    a3 = sigmoid(z3)
    a3 = np.concatenate((np.ones((z3.shape[0],1)),a3),1)
    z4 = np.matmul(a3,theta3.T)
    a4 = sigmoid(z4)
    h = a4
    
    # Regularisation
    reg1 = np.sum(theta1[:,1:] * theta1[:,1:])
    reg2 = np.sum(theta2[:,1:] * theta2[:,1:])
    reg3 = np.sum(theta3[:,1:] * theta3[:,1:])
    reg = reg1 + reg2 + reg3
    
    J = (-1/m) * np.sum( y * np.log(h) + (1 - y) * np.log(1 - h) ) + (Lambda/(2*m)) * reg
    
    return J
    
    #print(Y.shape)
    #print(Y[200000:200010,:])

J = computeCost(X_train,Y_train,Theta1,Theta2,Theta3,Lambda)
print(J)

def gradient(X,y,theta1,theta2,theta3,Lambda):
    """This function uses Back Propogation to calculate gradient for each theta"""
    K = theta3.shape[0]
    m = y.shape[0]
    
    # Y1 = np.arange(K)
    # Y = np.tile(Y1,(m,1))
    # for i in range(m):
    #     Y[i,:] = np.int32(Y[i,:] == y[i])
        
    # Forward propogation to find final output from input
    a1 = X  # a1 has input units for 300,000 stimuli. a1.shape = (300000,785)
    z2 = np.matmul(a1,theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((z2.shape[0],1)),a2),1)
    z3 = np.matmul(a2,theta2.T)
    a3 = sigmoid(z3)
    a3 = np.concatenate((np.ones((z3.shape[0],1)),a3),1)
    z4 = np.matmul(a3,theta3.T)
    a4 = sigmoid(z4)
    h = a4
    
    # Back Propogation to find gradient
    d4 = a4 - y
    d3 = np.matmul(d4,theta3) * a3 * (1 - a3)
    d3 = d3[:,1:]  # d3 becomes 300000 x 40
    d2 = np.matmul(d3,theta2) * a2 * (1 - a2)
    d2 = d2[:,1:]  # d2 becomes 300000 x 40
    
    Delta1 = np.matmul(d2.T , a1)
    Delta2 = np.matmul(d3.T , a2)
    Delta3 = np.matmul(d4.T , a3)
    
    Theta1new = theta1;
    Theta1new = np.concatenate((np.zeros((theta1.shape[0],1)).reshape(theta1.shape[0],1), Theta1new[:,1:]),1);
    Theta2new = theta2;
    Theta2new = np.concatenate((np.zeros((theta2.shape[0],1)).reshape(theta2.shape[0],1), Theta2new[:,1:]),1);
    Theta3new = theta3;
    Theta3new = np.concatenate((np.zeros((theta3.shape[0],1)).reshape(theta3.shape[0],1), Theta3new[:,1:]),1);

    Theta1_grad = (1/m)*Delta1 + (Lambda/m) * Theta1new;
    Theta2_grad = (1/m)*Delta2 + (Lambda/m) * Theta2new;
    Theta3_grad = (1/m)*Delta3 + (Lambda/m) * Theta3new;
    
    return Theta1_grad, Theta2_grad, Theta3_grad

#gradient(X_train,y_train,Theta1,Theta2,Theta3,Lambda)



def predict(X,y,theta1,theta2,theta3):
    """Gives prediction and number of correct predictions"""
    """Here the X and y are test dataset"""
    m = y.shape[0]

    a1 = X  # a1 has input units for 100,000 stimuli. a1.shape = (100000,785)
    z2 = np.matmul(a1,theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((z2.shape[0],1)),a2),1)
    z3 = np.matmul(a2,theta2.T)
    a3 = sigmoid(z3)
    a3 = np.concatenate((np.ones((z3.shape[0],1)),a3),1)
    z4 = np.matmul(a3,theta3.T)
    a4 = sigmoid(z4)

    print(a4)
    result = np.argmax(a4,1)  # This is an 100000 dimentional vector
    result.reshape(1,m)
    return result
    # num = 0
    # for i in range(m):
    #     if result[i] == y[i]:
    #         num += 1

    # accu = (num/m) * 100


def accuracy(X,y,theta1,theta2,theta3):
    """This function gives accuracy of the model"""
    """Here the X and y are test dataset"""
    m = y.shape[0]
    result = predict(X,y,theta1,theta2,theta3)

    num = 0
    for i in range(m):
        if result[i] == y[i]:
            num += 1

    accu = np.mean(np.double(num/m)) * 100
    return accu


def gradientDescent(X,y,X1,y1,theta1,theta2,theta3,alpha,iterations,Lambda):
    """This function returns final weights using descent"""
    m = y.shape[0]
    iter = 0
    for x in range(iterations):
        iter += 1
        grad1, grad2, grad3 = gradient(X,y,theta1,theta2,theta3,Lambda)
        thetatemp1 = theta1 - (alpha/m) * grad1
        thetatemp2 = theta2 - (alpha/m) * grad2
        thetatemp3 = theta3 - (alpha/m) * grad3
        theta1 = thetatemp1
        theta2 = thetatemp2
        theta3 = thetatemp3
        if iter % 5 == 0:
            print("\nIteration = ",iter)
            print("Coost = ",computeCost(X,y,theta1,theta2,theta3,Lambda))

            m = y1.shape[0]
            result = predict(X,y,theta1,theta2,theta3)
            print("Result = ",result)

            num = 0
            for i in range(m):
                if result[i] == y1[i]:
                    num += 1

            accu = np.mean(np.double(num/m)) * 100
            print("Accuracy = ",accu)
        
    return theta1,theta2,theta3


print(gradientDescent(X_train,Y_train,X_test,y_test,Theta1,Theta2,Theta3,alpha,iterations,Lambda))



