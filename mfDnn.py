import tensorflow as tf
import numpy as np
import keras

from keras import Input,Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Flatten,Embedding,Reshape,Concatenate,Dot,Add
from keras.callbacks import ModelCheckpoint

import math
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tqdm import tqdm

#The comparsion script, compare the Matrix Factorization with the Neural Network
# The training and prediction(without rating) is using full training set.

def getData():
	user = []
	movie = []
	rating = []
	with open("ml-100k/u.data",encoding = 'ISO-8859-1') as f:#<-- use full set
		info = f.readlines()
		for message in info:
			message = message.split("\t")
			message = message[0:3]
			user.append(message[0])
			movie.append(message[1])
			rating.append(message[2])
	user = np.array(user)
	movie = np.array(movie)
	rating = np.array(rating)
	return user,movie,rating

def getTest():
	user = []
	movie = []
	rating = []
	with open("ml-100k/u2.test",encoding = 'ISO-8859-1') as f:#<--ignore the test data here
		info = f.readlines()
		for message in info:
			message = message.split("\t")
			message = message[0:3]
			user.append(message[0])
			movie.append(message[1])
			rating.append(message[2])
	user = np.array(user)
	movie = np.array(movie)
	rating = np.array(rating)
	return user,movie,rating


def network(pretrained_weights = None):

	userInput = Input(shape = (1,))
	user = Embedding(944,50)(userInput)
	user = Reshape((50,))(user)

	movieInput = Input(shape = (1,))
	movie = Embedding(1684,50)(movieInput)
	movie = Reshape((50,))(movie)
	
	#---------------V1-----------
	user1 = Embedding(944,1)(userInput)
	user1 = Reshape((1,))(user1)

	movie1 = Embedding(1684,1)(movieInput)
	movie1 = Reshape((1,))(movie1)
	
	x = Dot(axes = 1)([user,movie])
	x = Add()([x,user1,movie1])
	#------------v1-------------
	'''#-------V2--------------
	x = Concatenate()([user,movie])
	x = Dropout(0.05)(x)

	x = Dense(10,activation = 'relu')(x)
	x = Dropout(0.5)(x)
	#-------V2--------------'''
	result = Dense(1,activation = 'linear')(x)

	model = Model(inputs = [userInput,movieInput],outputs = result)

	model.compile(
			optimizer = Adam(lr = 0.001),
			loss = tf.keras.losses.MAE,
			metrics = [tf.keras.metrics.RootMeanSquaredError(name='rmse')]
		)

	if (pretrained_weights):
			model.load_weights(pretrained_weights)

	return model

def train(model,user,movie,rating,userTest,movieTest,ratingTest):

	history = model.fit(
		x = [user,movie],
		y = rating,
		#validation_data = [[userTest,movieTest],ratingTest],
		batch_size = 100,
		epochs = 10
	)

user,movie,rating = getData()
userTest,movieTest,ratingTest = getTest()
model = network()
train(model,user,movie,rating,userTest,movieTest,ratingTest)


def matrix_factorization(R, P, Q, K, steps=50, alpha=0.002, beta=0.02):
	Q = Q.T
	for step in tqdm(range(steps)):
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - np.dot(P[i,:],Q[:,j])
					for k in range(K):
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		eR = np.dot(P,Q)
		e = 0
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
					for k in range(K):
						e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
			break
	return P, Q.T

R = np.zeros([943,1682])
with open("ml-100k/u.data",encoding = 'ISO-8859-1') as f:#<-- use full set
		info = f.readlines()
		for message in info:
			message = message.split("\t")
			message = message[0:3]
			R[int(message[0])-1][int(message[1])-1] = int(message[2])

R = np.array(R)
N = len(R)
M = len(R[0])

K = 5
P = np.random.rand(N,K)
Q = np.random.rand(M,K)
nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)
mf = []
dnn = []
with open("ml-100k/u.data",encoding = 'ISO-8859-1') as f:#<-- use full set
        info = f.readlines()
        infoList = tqdm(info)
        for message in infoList:
            message = message.split("\t")
            message = message[0:3]
            mf.append(int(nR[int(message[0])-1][int(message[1])-1]))
            u = message[0]
            m = message[1]
            u = np.array([u])
            m = np.array([m])
            dnn.append(int(model.predict([u,m])))
            
print('mf-->dnn',math.sqrt(mean_squared_error(mf,dnn)))
print('mf<--dnn',math.sqrt(mean_squared_error(dnn,mf)))