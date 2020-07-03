import numpy
import math
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
# The matrix facorization script, the path are set to read each fold cross-validation data. 
# So the first time set the path to u1.base and u1.test, get the result and then move to u2.base and u2.test
# After five time running, the average 5 fold cross validation score, and the learning curve can be generated.
# Different K score can also be tested via this script
curve = []
def matrix_factorization(R, P, Q, K, steps=50, alpha=0.002, beta=0.02):
    Q = Q.T
    for step in tqdm(range(steps)):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
        curve.append(e)
    return P, Q.T

R = numpy.zeros([943,1682])
with open("ml-100k/u5.base",encoding = 'ISO-8859-1') as f:# Set the training path here
		info = f.readlines()
		for message in info:
			message = message.split("\t")
			message = message[0:3]
			R[int(message[0])-1][int(message[1])-1] = int(message[2])

R = numpy.array(R)
N = len(R)
M = len(R[0])

K = 5#<--adjust the K value here
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)
nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)
groundTruth = []
predict = []
with open("ml-100k/u5.test",encoding = 'ISO-8859-1') as f:# Set the testing path here
	info = f.readlines()
	for message in info:
		message = message.split("\t")
		message = message[0:3]
		predict.append(nR[int(message[0])-1][int(message[1])-1])
		groundTruth.append(int(message[2]))

mae = mean_absolute_error(groundTruth,predict)
rmse = math.sqrt(mean_squared_error(groundTruth,predict))
print('mae: ',mae)
print('rmse: ',rmse)
plt.plot(curve)
plt.show()