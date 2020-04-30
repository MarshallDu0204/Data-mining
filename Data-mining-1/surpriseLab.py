from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import cross_validate

# After execute the testFactor to derive the approprate factor, run the surpriseLab to gain the result of 5 fold cross-validation
#from the Random, KNN, and SVD The parameter set is get according to the testFactor.

dataDir = ("ml-100k/")
reader = Reader('ml-100k')
train_file = dataDir + 'u%d.base'
test_file = dataDir + 'u%d.test'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()

algo3 = SVD()
algo2 = KNNBasic(user_based = False,k = 17)
algo1 = NormalPredictor()
i = 0
for trainset, testset in pkf.split(data):
    i+=1
    print("Random",i)

    # train and test algorithm.
    algo1.fit(trainset)
    predictions = algo1.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)
    
    print("KNN",i)
    
    # train and test algorithm.
    algo2.fit(trainset)
    predictions = algo2.test(testset)
    
    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)
    
    print("SVD",i)
    
     # train and test algorithm.
    algo3.fit(trainset)
    predictions = algo3.test(testset)
    
     # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)