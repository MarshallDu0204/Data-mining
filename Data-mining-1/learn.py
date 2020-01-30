import csv
import numpy as np
import keras
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

imgWidth = 28
imgHeight = 28

def readTrainingData(path = ""):
	trainingData = np.loadtxt("training.csv",delimiter = ",",skiprows = 1)

	label = trainingData[:,0]
	data = trainingData[:,1:785]

	return data,label

def preprocessing(data,label):
	global imgHeight
	global imgWidth

	# add more preprocessing steps here

	data = data/255

	data = data.reshape(len(data),imgWidth,imgHeight,1)

	label = keras.utils.to_categorical(label,num_classes = 10)

	return data,label

def CNN(pretrained_weights = None,input_shape = (imgHeight,imgWidth,1)):
	
	model = Sequential()

	model.add(Conv2D(32,(3,3),activation = 'relu',input_shape = input_shape))
	model.add(Conv2D(32,(3,3),activation = 'relu'))

	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Dropout(0.25))

	model.add(Conv2D(64,(3,3),activation = 'relu'))
	model.add(Conv2D(64,(3,3),activation = 'relu'))

	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(256,activation = 'relu'))

	model.add(Dropout(0.5))

	model.add(Dense(10,activation = 'softmax'))

	model.compile(optimizer = Adam(lr = 1e-3),loss = 'categorical_crossentropy',metrics = ['accuracy'])

	return model

def trainCNN(model,data,label):

	model_checkpoint = ModelCheckpoint('CNN.hdf5',monitor = 'loss',verbose = 1,save_best_only = True)

	model.fit(data,label,epochs = 5,batch_size = 16,callbacks = [model_checkpoint])

	return model


def predictResult(model,path = ""):
	global imgHeight
	global imgWidth

	testingData = np.loadtxt("testing.csv",delimiter = ",",skiprows = 1)

	data = testingData/255

	data = data.reshape(len(data),imgWidth,imgHeight,1)

	result = model.predict(data,batch_size = 16)

	i = 0

	resultList = []

	while i!=len(result):
		resultList.append(np.argmax(result[i]))
		i+=1

	with open("result.csv","w",newline = "") as file:
		writer = csv.writer(file)
		writer.writerow(["TestLabel"])
		for result in resultList:
			writer.writerow([result])
		
		file.close()

def dataToImg(index):
	data = np.loadtxt("testing.csv",delimiter = ",",skiprows = 1)
	img = data[index]
	img = img.reshape(28,28,1)
	newImg = []
	for x in img:
		tempX = []
		for y in x:
			temp = [y[0],y[0],y[0]]
			temp = np.array(temp,dtype = 'uint8')
			tempX.append(temp)
		tempX = np.array(tempX)
		newImg.append(tempX)
	newImg = np.array(newImg)

	img = Image.fromarray(newImg)
	img.show()

def execCNN(data,label):
	data,label = preprocessing(data,label)
	model = CNN()
	model = trainCNN(model,data,label)
	predictResult(model)

data,label = readTrainingData()
execCNN(data,label)
