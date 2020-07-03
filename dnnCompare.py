import tensorflow as tf
import numpy as np
import keras

from keras import Input,Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Flatten,Embedding,Reshape,Concatenate,Dot,Add
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error,mean_squared_error
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import cross_validate
from tqdm import tqdm
import math
#The comparsion script, compare the Random, KNN, and SVD with the Neural Network
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
	with open("ml-100k/u2.test",encoding = 'ISO-8859-1') as f:#<-- ignore the test set
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
		validation_data = [[userTest,movieTest],ratingTest],
		batch_size = 100,
		epochs = 10
	)

user,movie,rating = getData()
userTest,movieTest,ratingTest = getTest()
model = network()
train(model,user,movie,rating,userTest,movieTest,ratingTest)

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = KNNBasic(user_based = False,k = 17)
algo.fit(trainset)
algo1 = SVD()
algo1.fit(trainset)
algo2 = NormalPredictor()
algo2.fit(trainset)

dnn = []
random = []
knn = []
svd = []

with open("ml-100k/u.data",encoding = 'ISO-8859-1') as f:#<-- use full set
        info = f.readlines()
        infoList = tqdm(info)
        for message in infoList:
            message = message.split("\t")
            message = message[0:3]
            u = message[0]
            m = message[1]
            knnPred = algo.predict(str(u),str(m))
            svdPred = algo1.predict(str(u),str(m))
            randPred = algo2.predict(str(u),str(m))
            knn.append(int(knnPred.est))
            svd.append(int(svdPred.est))
            random.append(int(randPred.est))
            u = np.array([u])
            m = np.array([m])
            dnn.append(int(model.predict([u,m])))
            
print('dnn-->knn',math.sqrt(mean_squared_error(dnn,knn)))
print('dnn-->svd',math.sqrt(mean_squared_error(dnn,svd)))
print('dnn-->random',math.sqrt(mean_squared_error(dnn,random)))
print('dnn<--knn',math.sqrt(mean_squared_error(knn,dnn)))
print('dnn<--svd',math.sqrt(mean_squared_error(svd,dnn)))
print('dnn<--random',math.sqrt(mean_squared_error(random,dnn)))