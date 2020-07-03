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

#code of Convolution Neural Network
class CNN():

	imgWidth = 28#the size of each image
	imgHeight = 28

	num_epochs = 7
	batch_size =16

	def __init__(self,imgWidth = 28,imgHeight = 28):
		self.imgWidth = imgWidth
		self.imgHeight = imgHeight
		
	def preprocessing(self,data,label):

		# add more preprocessing steps here

		data = data/255#scale the pixel grey scale value to 0-1

		data = data.reshape(len(data),self.imgWidth,self.imgHeight,1)# reshape the 784 pixel to 28*28

		label = keras.utils.to_categorical(label,num_classes = 10)#one hot label of the data from 0-9

		return data,label

	def CNNmodel(self,pretrained_weights = None,input_shape = (imgHeight,imgWidth,1)):
		
		model = Sequential()#init the model

		model.add(Conv2D(32,(3,3),activation = 'relu',input_shape = input_shape))
		model.add(Conv2D(32,(3,3),activation = 'relu'))#first convolution layer

		model.add(MaxPooling2D(pool_size = (2,2)))#max pooling layer

		model.add(Dropout(0.25))#give up 25% of weight value

		model.add(Conv2D(64,(3,3),activation = 'relu'))#convolution layer
		model.add(Conv2D(64,(3,3),activation = 'relu'))

		model.add(MaxPooling2D(pool_size = (2,2)))#max pooling layer

		model.add(Dropout(0.25))

		model.add(Flatten())#flat the data to the fully connect layer

		model.add(Dense(512,activation = 'relu'))# fully connect layer

		model.add(Dropout(0.5))#random giveup 50 precent of weight to prevent overfitting

		model.add(Dense(10,activation = 'softmax'))#the softmax layer

		model.compile(optimizer = Adam(lr = 1e-3),loss = 'categorical_crossentropy',metrics = ['accuracy'])

		return model

	def trainCNN(self,model,data,label):

		model_checkpoint = ModelCheckpoint('CNN.hdf5',monitor = 'loss',verbose = 1,save_best_only = True)#store the model in each epoch

		model.fit(data,label,epochs = self.num_epochs,batch_size = self.batch_size,callbacks = [model_checkpoint])#training the data

		return model


	def predictResult(self,model,testDataPath = "testing.csv",outputPath = "result.csv"):

		testingData = np.loadtxt(testDataPath,delimiter = ",",skiprows = 1)#load the testing data

		data = testingData/255

		data = data.reshape(len(data),self.imgWidth,self.imgHeight,1)

		result = model.predict(data,batch_size = self.batch_size)#predict the result

		i = 0

		resultList = []

		while i!=len(result):
			resultList.append(np.argmax(result[i]))#argmax to extract the max possibility of given 10 result of softmax layer
			i+=1

		with open(outputPath,"w",newline = "") as file:#write the result to file
			writer = csv.writer(file)
			writer.writerow(["TestLabel"])
			for result in resultList:
				writer.writerow([result])
			
			file.close()

	def dataToImg(self,index):#convert a given line of trainig data to image
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

	def preprocessing(self,data,label):
		#add preprocessing steps here
		pass

	def training(self,data,label):
		classTree = tree.DecisionTreeClassifier()
		classTree = classTree.fit(data,label)
		return classTree 

	def predict(self,classTree,testDataPath = "testing.csv",outputPath = "result1.csv"):
		testingData = np.loadtxt(testDataPath,delimiter = ",",skiprows = 1)

		resultList = classTree.predict(testingData)

		with open(outputPath,"w",newline = "") as file:
			writer = csv.writer(file)
			writer.writerow(["TestLabel"])
			for result in resultList:
				writer.writerow([result])
			
			file.close()

# code of the example classifycation method
class dmExample():

	def preprocessing(self,data,label):
		
		return data,label

	def training(self,data,label):
		
		return model

	def predict(self,model,testDataPath = "testing.csv",outputPath = ""):
		testingData = np.loadtxt(testDataPath,delimiter = ",",skiprows = 1)

		#result = model.predict(testingData)

		with open(outputPath,"w",newline = "") as file:
			writer = csv.writer(file)
			writer.writerow(["TestLabel"])
			for result in resultList:
				writer.writerow([result])
			
			file.close()


def readTrainingData(path = "training.csv"):
	trainingData = np.loadtxt(path,delimiter = ",",skiprows = 1)#skip the first row of string

	label = trainingData[:,0]#split the label and training data
	data = trainingData[:,1:785]

	return data,label

def calculateScore(path1 = "result.csv",path2 = "result1.csv"):#compare the similarity of any two result 
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

#code of execute the example class
def execDmExample():
	model = dmExample()
	#model = model.training(data,label)
	#model.predict()
	

data,label = readTrainingData()
execCNN(data,label)
#execDecisionTree(data,label)
#calculateScore("result.csv","result1.csv")
