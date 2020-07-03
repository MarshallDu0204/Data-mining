import numpy
import math
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tqdm import tqdm

from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import cross_validate

#The comparsion script, compare the Matrix Factorization with the Random, KNN, and SVD
# The training and prediction(without rating) is using full training set.

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
	return P, Q.T

R = numpy.zeros([943,1682])
with open("ml-100k/u.data",encoding = 'ISO-8859-1') as f:#<-- use full set
		info = f.readlines()
		for message in info:
			message = message.split("\t")
			message = message[0:3]
			R[int(message[0])-1][int(message[1])-1] = int(message[2])

R = numpy.array(R)
N = len(R)
M = len(R[0])

K = 5
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)
nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = KNNBasic(user_based = False,k = 17)
algo.fit(trainset)
algo1 = SVD()
algo1.fit(trainset)
algo2 = NormalPredictor()
algo2.fit(trainset)
mf = []
knn = []
svd = []
random = []

with open("ml-100k/u.data",encoding = 'ISO-8859-1') as f:#<-- use full set
        info = f.readlines()
        infoList = tqdm(info)
        for message in infoList:
            message = message.split("\t")
            message = message[0:3]
            mf.append(int(nR[int(message[0])-1][int(message[1])-1]))
            knnPred = algo.predict(message[0],message[1])
            svdPred = algo1.predict(message[0],message[1])
            randPred = algo2.predict(message[0],message[1])
            knn.append(int(knnPred.est))
            svd.append(int(svdPred.est))
            random.append(int(randPred.est))
            

print('mf-->knn',math.sqrt(mean_squared_error(mf,knn)))
print('mf-->svd',math.sqrt(mean_squared_error(mf,svd)))
print('mf-->random',math.sqrt(mean_squared_error(mf,random)))
print('mf<--knn',math.sqrt(mean_squared_error(knn,mf)))
print('mf<--svd',math.sqrt(mean_squared_error(svd,mf)))
print('mf<--random',math.sqrt(mean_squared_error(random,mf)))