import csv
import numpy as np
import keras
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn import tree

OutputPath = "result2.csv"

class CNN():
	global OutputPath

	imgWidth = 28
	imgHeight = 28

	def __init__(self,imgWidth = 28,imgHeight = 28):
		self.imgWidth = imgWidth
		self.imgHeight = imgHeight
		
	def preprocessing(self,data,label):

		# add more preprocessing steps here

		data = data/255

		data = data.reshape(len(data),self.imgWidth,self.imgHeight,1)

		label = keras.utils.to_categorical(label,num_classes = 10)

		return data,label

	def CNNmodel(self,pretrained_weights = None,input_shape = (imgHeight,imgWidth,1)):
		
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

	def trainCNN(self,model,data,label):

		model_checkpoint = ModelCheckpoint('CNN.hdf5',monitor = 'loss',verbose = 1,save_best_only = True)

		model.fit(data,label,epochs = 5,batch_size = 16,callbacks = [model_checkpoint])

		return model


	def predictResult(self,model,path = ""):

		testingData = np.loadtxt("testing.csv",delimiter = ",",skiprows = 1)

		data = testingData/255

		data = data.reshape(len(data),self.imgWidth,self.imgHeight,1)

		result = model.predict(data,batch_size = 16)

		i = 0

		resultList = []

		while i!=len(result):
			resultList.append(np.argmax(result[i]))
			i+=1

		with open(OutputPath,"w",newline = "") as file:
			writer = csv.writer(file)
			writer.writerow(["TestLabel"])
			for result in resultList:
				writer.writerow([result])
			
			file.close()

	def dataToImg(self,index):
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

class decisionTree():
	global OutputPath

	def preprocessing(self,data,label):
		#add preprocessing steps here
		pass

	def training(self,data,label):
		classTree = tree.DecisionTreeClassifier()
		classTree = classTree.fit(data,label)
		return classTree 

	def predict(self,classTree,path = ""):
		testingData = np.loadtxt("testing.csv",delimiter = ",",skiprows = 1)

		resultList = classTree.predict(testingData)

		with open(OutputPath,"w",newline = "") as file:
			writer = csv.writer(file)
			writer.writerow(["TestLabel"])
			for result in resultList:
				writer.writerow([result])
			
			file.close()

def readTrainingData(path = ""):
	trainingData = np.loadtxt("training.csv",delimiter = ",",skiprows = 1)

	label = trainingData[:,0]
	data = trainingData[:,1:785]

	return data,label

def calculateScore(path1,path2):
	result1 = np.loadtxt(path1,delimiter = ",",skiprows = 1)
	result2 = np.loadtxt(path2,delimiter = ",",skiprows = 1)
	i = 0
	k = 0
	while i!=len(result1):
		if result1[i] == result2[i]:
			k+=1
		i+=1

	score = k/len(result1)

	print("Score",score)

def execCNN(data,label):
	cnn = CNN()
	data,label = cnn.preprocessing(data,label)
	model = cnn.CNNmodel()
	model = cnn.trainCNN(model,data,label)
	cnn.predictResult(model)

def execDecisionTree(data,label):
	tree = decisionTree()
	model = tree.training(data,label)
	tree.predict(model)

def execBayes(data,label):
	bayesClassifier = naiveBayes()
	model = bayesClassifier.training(data,label)
	bayesClassifier.predict(model)

data,label = readTrainingData()
#execCNN(data,label)
#execDecisionTree(data,label)
calculateScore("result.csv","result2.csv")
