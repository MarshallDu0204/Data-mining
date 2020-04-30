from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tqdm import tqdm
import math
#The comparsion script, compare the Random, KNN, and SVD 
# The training and prediction(without rating) is using full training set.

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = KNNBasic(user_based = False,k = 17)
algo.fit(trainset)
algo1 = SVD()
algo1.fit(trainset)
algo2 = NormalPredictor()
algo2.fit(trainset)

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
            
            
print('rand-->knn',math.sqrt(mean_squared_error(random,knn)))
print('rand-->svd',math.sqrt(mean_squared_error(random,svd)))
print('knn-->rand',math.sqrt(mean_squared_error(knn,random)))
print('knn-->svd',math.sqrt(mean_squared_error(knn,svd)))
print('svd-->knn',math.sqrt(mean_squared_error(svd,knn)))
print('svd-->rand',math.sqrt(mean_squared_error(svd,random)))