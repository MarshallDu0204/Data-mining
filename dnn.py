import tensorflow as tf
import numpy as np
import keras

from keras import Input,Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Flatten,Embedding,Reshape,Concatenate,Dot,Add
from keras.callbacks import ModelCheckpoint

# The Neural Network script, there are two model, choose one before running. the path are set to read each fold cross-validation data. 
# So the first time set the path to u1.base and u1.test, get the result and then move to u2.base and u2.test
# After five time running, the average 5 fold cross validation score through the validation score.
def getData():
	user = []
	movie = []
	rating = []
	with open("ml-100k/u2.base",encoding = 'ISO-8859-1') as f:# Set the training path here
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
	with open("ml-100k/u2.test",encoding = 'ISO-8859-1') as f:# Set the testing path here
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
		epochs = 10#<--set the epoch number here
	)

user,movie,rating = getData()
userTest,movieTest,ratingTest = getTest()
model = network()
train(model,user,movie,rating,userTest,movieTest,ratingTest)