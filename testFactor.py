from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import cross_validate
#This script is mainly test on different parameter of KNN and SVD to get the best result


data = Dataset.load_builtin('ml-100k')

# iterate the k in the KNN, to get the optimal MAE score
i = 3
while i!=20:
	print("K = ",i)
	# Use the famous knn algorithm
	algo = KNNBasic(user_based = False,k = i)

	# Run 5-fold cross-validation and print results.
	cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
	i+=1

# iterate the n_factor in the KNN, to get the optimal MAE score

i = [50,60,70,80,90,100,110,120,130,140,150]
for index in i:
	print("n_factor = ",index)
	# Use the famous knn algorithm
	algo = SVD(n_factors=index)

	# Run 5-fold cross-validation and print results.
	cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)